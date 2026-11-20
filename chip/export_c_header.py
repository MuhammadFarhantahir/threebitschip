"""Export falcon_submit/h1_int4_bundle.pkl into a self-contained C99 Flash-ROM header
(chip/firmware_rp2350/falcon_h1_weights.h) ready to flash onto a Raspberry Pi Pico 2 (RP2350)
or STM32H7 ARM Cortex-M microcontroller.

Run:
    .venv/bin/python chip/export_c_header.py
"""
import pickle
from pathlib import Path

import numpy as np

from bench_int4_gru import build_luts, float_to_mult_shift

ROOT = Path(__file__).resolve().parent.parent
PKL = ROOT / "falcon_submit" / "h1_int4_bundle.pkl"
OUT_H = Path(__file__).resolve().parent / "firmware_rp2350" / "falcon_h1_weights.h"


def fmt_arr(name, ctype, arr, per_line=16):
    flat = np.asarray(arr).ravel().tolist()
    lines = [f"static const {ctype} {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        lines.append("    " + ", ".join(str(int(v)) for v in flat[i:i + per_line]) + ",")
    lines.append("};\n")
    return "\n".join(lines)


def main():
    with open(PKL, "rb") as f:
        payload = pickle.load(f)
    core = payload["core"]
    ad = payload["adapters"]["default"]
    sig_lut, th_lut = build_luts()

    x_scale = core["x_scale"]
    h_scale = 1.0 / 128.0
    ih_m, ih_s = float_to_mult_shift(core["s_ih"] * x_scale * 4096.0)
    hh_m, hh_s = float_to_mult_shift(core["s_hh"] * h_scale * 4096.0)
    out_m, out_s = float_to_mult_shift(ad["s_out"] * h_scale * 4096.0)

    OUT_H.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "/* Auto-generated from falcon_submit/h1_int4_bundle.pkl — Threebits 4-bit Human Implant GRU */\n",
        "#ifndef THREEBITS_FALCON_H1_WEIGHTS_H\n#define THREEBITS_FALCON_H1_WEIGHTS_H\n#include <stdint.h>\n\n",
        f"#define TB_INPUT_SIZE  {core['I']}\n",
        f"#define TB_HIDDEN_SIZE {core['H']}\n",
        f"#define TB_OUTPUT_SIZE {core['O']}\n",
        f"#define TB_IH_MULT     {ih_m}\n#define TB_IH_SHIFT    {ih_s}\n",
        f"#define TB_HH_MULT     {hh_m}\n#define TB_HH_SHIFT    {hh_s}\n",
        f"#define TB_OUT_MULT    {out_m}\n#define TB_OUT_SHIFT   {out_s}\n\n",
        fmt_arr("TB_W_IH_PACKED", "uint8_t", core["w_ih_packed"]),
        fmt_arr("TB_W_HH_PACKED", "uint8_t", core["w_hh_packed"]),
        fmt_arr("TB_W_OUT_PACKED", "uint8_t", ad["w_out_packed"]),
        fmt_arr("TB_B_IH_Q12", "int32_t", core["b_ih_q12"], per_line=12),
        fmt_arr("TB_B_HH_Q12", "int32_t", core["b_hh_q12"], per_line=12),
        fmt_arr("TB_B_OUT_Q12", "int32_t", ad["b_out_q12"], per_line=8),
        fmt_arr("TB_SIGMOID_LUT", "int16_t", sig_lut, per_line=16),
        fmt_arr("TB_TANH_LUT", "int16_t", th_lut, per_line=16),
        "#endif /* THREEBITS_FALCON_H1_WEIGHTS_H */\n",
    ]
    OUT_H.write_text("".join(parts))
    rom_b = (core["w_ih_packed"].nbytes + core["w_hh_packed"].nbytes + ad["w_out_packed"].nbytes
             + core["b_ih_q12"].nbytes + core["b_hh_q12"].nbytes + ad["b_out_q12"].nbytes
             + sig_lut.nbytes + th_lut.nbytes)
    print(f"Exported {OUT_H} ({OUT_H.stat().st_size/1e3:.1f} kB source | {rom_b:,} B = {rom_b/1e3:.2f} kB Flash ROM)")


if __name__ == "__main__":
    main()
