/* Pure-integer C99 packed 4-bit GRU decoder kernel for implantable BCI microcontrollers.
 *
 * Zero floating-point operations inside gru_int4_step() or gru_int4_forward_seq():
 *   - Weights stored packed 2 per byte (signed 4-bit two's complement [-8, +7]):
 *       W_ih: (3*H x I) -> 3*H*I/2 bytes  (33,792 B for I=176, H=128)
 *       W_hh: (3*H x H) -> 3*H*H/2 bytes  (24,576 B for H=128)
 *       W_out: (O x H)  -> O*H/2 bytes    (   448 B for O=7, H=128)
 *       Total packed weight ROM: 58,816 bytes (58.8 kB)
 *   - Input activations x_t and recurrent hidden state h_t quantized to int8_t (Q7 for h_t).
 *   - Matrix-vector products accumulate int4 x int8 in int32_t with exact fixed-point
 *     (mult, shift) requantization to Q12 pre-activations.
 *   - Sigmoid and tanh computed via 257-entry Q15 lookup tables (1.0 kB ROM) with linear interpolation.
 *   - Per-channel affine input adapter (1% parameters) runs in Q12 integer fixed-point so
 *     cross-day recalibration updates only ~1.4 kB while the 58.8 kB int4 core stays read-only.
 */
#include <stdint.h>
#include <stddef.h>

#define Q12_ONE 4096
#define Q15_ONE 32767
#define LUT_SIZE 257
#define LUT_MIN_Q12 (-32768) /* -8.0 in Q12 */
#define LUT_MAX_Q12 (32767)  /* +8.0 in Q12 */

typedef struct {
    int32_t input_size;   /* I (must be even, e.g. 176) */
    int32_t hidden_size;  /* H (must be even, e.g. 128) */
    int32_t output_size;  /* O (e.g. 7) */

    /* Packed int4 weight matrices (two signed 4-bit weights per byte, row-major) */
    const uint8_t *w_ih_packed;   /* (3*H) * (I/2) bytes, PyTorch gate order [r, z, n] */
    const uint8_t *w_hh_packed;   /* (3*H) * (H/2) bytes, PyTorch gate order [r, z, n] */
    const uint8_t *w_out_packed;  /* O * (H/2) bytes */

    /* Biases in Q12 fixed-point */
    const int32_t *b_ih_q12;      /* 3*H */
    const int32_t *b_hh_q12;      /* 3*H */
    const int32_t *b_out_q12;     /* O */

    /* Exact integer requantization multipliers: acc * mult >> shift -> Q12 */
    int32_t ih_mult, ih_shift;
    int32_t hh_mult, hh_shift;
    int32_t out_mult, out_shift;

    /* 257-entry Q15 LUTs over [-8.0, +8.0] (step 0.0625 = 256 in Q12) */
    const int16_t *sigmoid_lut;   /* [0, 32767] */
    const int16_t *tanh_lut;      /* [-32767, 32767] */
} Int4GRUModel;

static inline int32_t requant_q12(int32_t acc, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)acc * (int64_t)mult;
    int64_t round_add = (shift > 0) ? (1LL << (shift - 1)) : 0;
    if (prod >= 0) {
        return (int32_t)((prod + round_add) >> shift);
    } else {
        return -(int32_t)(((-prod) + round_add) >> shift);
    }
}

static inline int32_t lut_interp_q15(const int16_t *lut, int32_t x_q12) {
    if (x_q12 <= LUT_MIN_Q12) return (int32_t)lut[0];
    if (x_q12 >= LUT_MAX_Q12) return (int32_t)lut[LUT_SIZE - 1];
    uint32_t u = (uint32_t)(x_q12 - LUT_MIN_Q12); /* [0, 65535] */
    uint32_t idx = u >> 8;                        /* [0, 255] */
    uint32_t frac = u & 255U;                     /* [0, 255] */
    int32_t y0 = (int32_t)lut[idx];
    int32_t y1 = (int32_t)lut[idx + 1];
    return y0 + (((y1 - y0) * (int32_t)frac) >> 8);
}

static inline int32_t dot_int4_int8(const uint8_t *w_row_packed, const int8_t *x, int32_t n) {
    int32_t acc = 0;
    int32_t pairs = n >> 1;
    for (int32_t p = 0; p < pairs; ++p) {
        uint8_t b = w_row_packed[p];
        int32_t w0 = (int32_t)((b & 0x0FU) ^ 8U) - 8;
        int32_t w1 = (int32_t)(((b >> 4) & 0x0FU) ^ 8U) - 8;
        acc += w0 * (int32_t)x[2 * p] + w1 * (int32_t)x[2 * p + 1];
    }
    return acc;
}

static inline int8_t clamp_i8(int32_t v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return (int8_t)v;
}

/* Single 20 ms causal step: updates h_q7[H] in place and writes out_q12[O].
 * Scratch buffer `tmp_ih_hh` must hold at least 6 * H int32_t elements (3 kB for H=128).
 */
void gru_int4_step(
    const Int4GRUModel *m,
    const int8_t *x_q8,
    int8_t *h_q7,
    int32_t *out_q12,
    int32_t *scratch_6h
) {
    const int32_t I = m->input_size;
    const int32_t H = m->hidden_size;
    const int32_t O = m->output_size;
    const int32_t ih_stride = I >> 1;
    const int32_t hh_stride = H >> 1;

    int32_t *gi = scratch_6h;           /* 3*H: W_ih * x + b_ih in Q12 */
    int32_t *gh = scratch_6h + 3 * H;   /* 3*H: W_hh * h + b_hh in Q12 */

    for (int32_t row = 0; row < 3 * H; ++row) {
        int32_t acc_i = dot_int4_int8(m->w_ih_packed + (size_t)row * ih_stride, x_q8, I);
        int32_t acc_h = dot_int4_int8(m->w_hh_packed + (size_t)row * hh_stride, h_q7, H);
        gi[row] = requant_q12(acc_i, m->ih_mult, m->ih_shift) + m->b_ih_q12[row];
        gh[row] = requant_q12(acc_h, m->hh_mult, m->hh_shift) + m->b_hh_q12[row];
    }

    for (int32_t j = 0; j < H; ++j) {
        int32_t r_q15 = lut_interp_q15(m->sigmoid_lut, gi[j] + gh[j]);
        int32_t z_q15 = lut_interp_q15(m->sigmoid_lut, gi[H + j] + gh[H + j]);
        /* r_t * (W_hn h_{t-1} + b_hn): Q15 * Q12 >> 15 -> Q12 */
        int32_t rh_q12 = (int32_t)(((int64_t)r_q15 * (int64_t)gh[2 * H + j]) >> 15);
        int32_t n_q15 = lut_interp_q15(m->tanh_lut, gi[2 * H + j] + rh_q12);
        /* h_t = (1 - z) * n + z * h_{t-1}, with h_{t-1} in Q7 -> Q15 via << 8 */
        int32_t h_prev_q15 = (int32_t)h_q7[j] << 8;
        int32_t h_new_q15 = (int32_t)((((int64_t)(Q15_ONE - z_q15) * (int64_t)n_q15) +
                                       ((int64_t)z_q15 * (int64_t)h_prev_q15)) >> 15);
        /* Round Q15 -> Q7 int8_t */
        int32_t h_q7_val = (h_new_q15 >= 0) ? ((h_new_q15 + 128) >> 8) : -(((-h_new_q15) + 128) >> 8);
        h_q7[j] = clamp_i8(h_q7_val);
    }

    for (int32_t o = 0; o < O; ++o) {
        int32_t acc_o = dot_int4_int8(m->w_out_packed + (size_t)o * hh_stride, h_q7, H);
        out_q12[o] = requant_q12(acc_o, m->out_mult, m->out_shift) + m->b_out_q12[o];
    }
}

/* Run an entire trial of T steps (20 ms bins) starting from zero hidden state. */
void gru_int4_forward_seq(
    const Int4GRUModel *m,
    const int8_t *x_seq_q8,   /* T * I */
    int32_t T,
    int32_t *out_seq_q12,     /* T * O */
    int8_t *h_buf,            /* H */
    int32_t *scratch_6h       /* 6 * H */
) {
    for (int32_t j = 0; j < m->hidden_size; ++j) h_buf[j] = 0;
    for (int32_t t = 0; t < T; ++t) {
        gru_int4_step(
            m,
            x_seq_q8 + (size_t)t * m->input_size,
            h_buf,
            out_seq_q12 + (size_t)t * m->output_size,
            scratch_6h
        );
    }
}
