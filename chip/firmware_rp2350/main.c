/* Bare-metal ARM Cortex-M33 (Raspberry Pi Pico 2 / RP2350) & Host-Testable Firmware Benchmark
 * for the Threebits Pure-Integer 4-Bit Human Implant GRU Decoder (66.35 kB total ROM + SRAM).
 *
 * Host verification build (macOS/Linux):
 *   cc -O3 -Wall -Wextra -DHOST_TEST chip/firmware_rp2350/main.c chip/int4_gru.c -o chip/firmware_rp2350/host_bench
 *   ./chip/firmware_rp2350/host_bench
 *
 * Raspberry Pi Pico 2 (RP2350, Cortex-M33 @ 150 MHz) build with Pico SDK:
 *   mkdir build && cd build && cmake -DPICO_BOARD=pico2 .. && make -j
 *   Flash `threebits_int4_gru.uf2` over USB and measure live USB current/milliwatts!
 */
#include <stdint.h>
#include <stdio.h>

#ifdef HOST_TEST
#include <time.h>
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}
#else
#include "pico/stdlib.h"
static uint64_t now_us(void) { return time_us_64(); }
#endif

#include "falcon_h1_weights.h"

/* Declarations from chip/int4_gru.c */
typedef struct {
    int32_t input_size, hidden_size, output_size;
    const uint8_t *w_ih_packed, *w_hh_packed, *w_out_packed;
    const int32_t *b_ih_q12, *b_hh_q12, *b_out_q12;
    int32_t ih_mult, ih_shift, hh_mult, hh_shift, out_mult, out_shift;
    const int16_t *sigmoid_lut, *tanh_lut;
} Int4GRUModel;

void gru_int4_step(const Int4GRUModel *m, const int8_t *x_q8, int8_t *h_q7,
                   int32_t *out_q12, int32_t *scratch_6h);

static int8_t g_hidden[TB_HIDDEN_SIZE];
static int32_t g_scratch[6 * TB_HIDDEN_SIZE];
static int32_t g_output[TB_OUTPUT_SIZE];
static int8_t g_input[TB_INPUT_SIZE];

int main(void) {
#ifndef HOST_TEST
    stdio_init_all();
    sleep_ms(1500);
#endif

    const Int4GRUModel model = {
        TB_INPUT_SIZE, TB_HIDDEN_SIZE, TB_OUTPUT_SIZE,
        TB_W_IH_PACKED, TB_W_HH_PACKED, TB_W_OUT_PACKED,
        TB_B_IH_Q12, TB_B_HH_Q12, TB_B_OUT_Q12,
        TB_IH_MULT, TB_IH_SHIFT, TB_HH_MULT, TB_HH_SHIFT, TB_OUT_MULT, TB_OUT_SHIFT,
        TB_SIGMOID_LUT, TB_TANH_LUT
    };

    /* Deterministic synthetic spike frame for reproducible cycle/power benchmarking */
    for (int i = 0; i < TB_INPUT_SIZE; ++i) g_input[i] = (int8_t)((i * 37) % 61 - 30);
    for (int j = 0; j < TB_HIDDEN_SIZE; ++j) g_hidden[j] = 0;

    const int N_STEPS = 2000; /* 40 seconds of real-time 50 Hz (20 ms/bin) neural decoding */
    uint64_t t0 = now_us();
    int64_t checksum = 0;
    for (int s = 0; s < N_STEPS; ++s) {
        g_input[s % TB_INPUT_SIZE] = (int8_t)((s * 13) % 53 - 26);
        gru_int4_step(&model, g_input, g_hidden, g_output, g_scratch);
        checksum += g_output[0];
    }
    uint64_t dt_us = now_us() - t0;
    double us_per_step = (double)dt_us / (double)N_STEPS;
    double duty_pct = (us_per_step / 20000.0) * 100.0;

    printf("=== Threebits Pure-Integer 4-Bit Human Implant GRU Firmware ===\n");
    printf("Config:   I=%d -> H=%d -> O=%d | Packed 4-bit Flash ROM: %zu B | Static SRAM: %zu B\n",
           TB_INPUT_SIZE, TB_HIDDEN_SIZE, TB_OUTPUT_SIZE,
           sizeof(TB_W_IH_PACKED) + sizeof(TB_W_HH_PACKED) + sizeof(TB_W_OUT_PACKED)
           + sizeof(TB_B_IH_Q12) + sizeof(TB_B_HH_Q12) + sizeof(TB_B_OUT_Q12)
           + sizeof(TB_SIGMOID_LUT) + sizeof(TB_TANH_LUT),
           sizeof(g_hidden) + sizeof(g_scratch) + sizeof(g_output) + sizeof(g_input));
    printf("Timing:   %d steps in %llu us -> %.2f us / 20 ms bin (%.3f%% CPU duty @ 50 Hz)\n",
           N_STEPS, (unsigned long long)dt_us, us_per_step, duty_pct);
    printf("Checksum: %lld | last Q12 out[0..2] = [%d, %d, %d]\n",
           (long long)checksum, g_output[0], g_output[1], g_output[2]);

#ifndef HOST_TEST
    /* Continuous 50 Hz real-time duty-cycle loop for USB power meter measurement */
    while (1) {
        uint64_t s0 = now_us();
        gru_int4_step(&model, g_input, g_hidden, g_output, g_scratch);
        uint64_t spent = now_us() - s0;
        if (spent < 20000ULL) sleep_us(20000ULL - spent);
    }
#endif
    return 0;
}
