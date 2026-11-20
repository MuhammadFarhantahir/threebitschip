"""FALCON H1 EvalAI submission decoder: 4-bit packed-weight causal GRU with 1% L2-anchored per-day adapter.

Implements `falcon_challenge.interface.BCIDecoder` (`reset`, `on_done`, `observe`, `predict`), running
either the pure-integer C99 packed int4xint8 kernel (`int4_gru.c`) or PyTorch quantized inference.

Usage:
  1. Train & bundle weights (held-in + per-day held-out 1% anchored adapters):
       .venv/bin/python falcon_submit/decoder.py --mode train --data-dir data/falcon_h1 --out falcon_submit/h1_int4_bundle.pkl
  2. Build & submit Docker container (once Docker Desktop + EvalAI CLI are installed):
       docker build -t threebits-h1-int4 -f falcon_submit/Dockerfile .
       evalai push threebits-h1-int4:latest --phase few-shot-minival-2319 --private
"""
import argparse
import copy
import ctypes
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chip.bench_int4_gru import Int4GRUModelC, build_luts, export_int4_c_bundle, float_to_mult_shift, pack_int4_matrix


def _load_c_lib():
    for cand in (Path(__file__).with_name("libint4_gru.so"), ROOT / "chip" / "libint4_gru.dylib", ROOT / "chip" / "libint4_gru.so"):
        if cand.exists():
            lib = ctypes.CDLL(str(cand))
            lib.gru_int4_step.argtypes = [
                ctypes.POINTER(Int4GRUModelC), ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ]
            return lib
    raise FileNotFoundError("Compiled libint4_gru shared library not found; compile chip/int4_gru.c first.")


def _make_runtime_struct(core, adapter):
    """Build ctypes Int4GRUModelC from frozen 58.4 kB int4 GRU core + per-day 1% adapter."""
    w_ih_p, w_hh_p = core["w_ih_packed"], core["w_hh_packed"]
    b_ih, b_hh = core["b_ih_q12"], core["b_hh_q12"]
    w_out_p, b_out = adapter["w_out_packed"], adapter["b_out_q12"]
    sig_lut, th_lut = build_luts()
    x_scale = core["x_scale"]
    h_scale = 1.0 / 128.0
    ih_m, ih_s = float_to_mult_shift(core["s_ih"] * x_scale * 4096.0)
    hh_m, hh_s = float_to_mult_shift(core["s_hh"] * h_scale * 4096.0)
    out_m, out_s = float_to_mult_shift(adapter["s_out"] * h_scale * 4096.0)
    st = Int4GRUModelC(
        input_size=core["I"], hidden_size=core["H"], output_size=core["O"],
        w_ih_packed=w_ih_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w_hh_packed=w_hh_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w_out_packed=w_out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        b_ih_q12=b_ih.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        b_hh_q12=b_hh.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        b_out_q12=b_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ih_mult=ih_m, ih_shift=ih_s, hh_mult=hh_m, hh_shift=hh_s, out_mult=out_m, out_shift=out_s,
        sigmoid_lut=sig_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        tanh_lut=th_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
    )
    return {"c": st, "keep": (w_ih_p, w_hh_p, w_out_p, b_ih, b_hh, b_out, sig_lut, th_lut), **adapter}


def train_bundle(data_dir: str, out_path: str):
    """Train 4-bit packed-weight GRU on held-in-calib + per-day 1% anchored adapters for held-out-calib."""
    from falcon_anchored_recal import recalibrate_anchored
    from falcon_precision_drift import CausalGRU, _load_all, fit_model, pad_trials, split_trials
    from precision_sweep import Rounded

    calib_in, _, ho_days, x_mean, x_std, y_std = _load_all(data_dir)
    all_trials = [tr for s in calib_in for tr in split_trials(s)]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(all_trials))
    n_tune = len(all_trials) * 15 // 100
    Xf, Yf, Mf = pad_trials([all_trials[i] for i in order[n_tune:]], x_mean, x_std, y_std)
    Xt, Yt, Mt = pad_trials([all_trials[i] for i in order[:n_tune]], x_mean, x_std, y_std)

    torch.manual_seed(0)
    base, _ = fit_model(CausalGRU(Xf.shape[-1]), Xf, Yf, Mf, Xt, Yt, Mt, lr=2e-3)
    torch.manual_seed(4)
    q4, _ = fit_model(Rounded(copy.deepcopy(base), 4), Xf, Yf, Mf, Xt, Yt, Mt, lr=1e-3, max_epochs=100, patience=12)

    b0 = export_int4_c_bundle(q4)
    steps0 = {n: float(ls.exp().item()) for n, ls in zip(q4.names, q4.log_steps)}
    core = {
        "I": b0["I"], "H": b0["H"], "O": b0["O"], "x_scale": b0["x_scale"],
        "w_ih_packed": b0["keep"][0], "w_hh_packed": b0["keep"][1],
        "b_ih_q12": b0["keep"][3], "b_hh_q12": b0["keep"][4],
        "s_ih": steps0["gru.weight_ih_l0"], "s_hh": steps0["gru.weight_hh_l0"],
        "x_std": x_std, "y_std": y_std,
    }

    def pack_adapter(model, day_mean):
        bd = export_int4_c_bundle(model)
        st = {n: float(ls.exp().item()) for n, ls in zip(model.names, model.log_steps)}
        return {
            "x_mean": day_mean, "ch_scale": bd["ch_scale"], "ch_bias": bd["ch_bias"],
            "w_out_packed": bd["keep"][2], "b_out_q12": bd["keep"][5], "s_out": st["readout.weight"],
        }

    adapters = {"default": pack_adapter(q4, x_mean)}
    for day, slist in sorted(ho_days.items()):
        # Combine available held-out calibration sessions for this day
        merged = {
            "spikes": np.concatenate([s["spikes"] for s in slist], axis=0),
            "vel": np.concatenate([s["vel"] for s in slist], axis=0),
            "mask": np.concatenate([s["mask"] for s in slist], axis=0),
            "tnum": np.concatenate([s["tnum"] + 1000 * i for i, s in enumerate(slist)], axis=0),
        }
        ma, dm = recalibrate_anchored(q4, merged, x_mean, x_std, y_std, mode="adapter")
        adapters[day] = pack_adapter(ma, dm)

    payload = {"core": core, "adapters": adapters}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {out_path} ({Path(out_path).stat().st_size / 1e3:.1f} kB) with {len(adapters)-1} held-out day adapters.")


try:
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from falcon_challenge.interface import BCIDecoder
except ImportError:
    BCIDecoder = object  # allows running --mode train locally without falcon_challenge installed


class Int4AnchoredGRUDecoder(BCIDecoder):
    """Pure-integer 4-bit causal GRU decoder compatible with FALCON EvalAI's BCIDecoder interface."""

    def __init__(self, task_config, model_path: str, batch_size: int = 1):
        if BCIDecoder is not object:
            super().__init__(task_config=task_config, batch_size=batch_size)
        self._task_config = task_config
        self.batch_size = batch_size
        self.lib = _load_c_lib()
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        self.core = payload["core"]
        self.adapters = payload["adapters"]
        self.H = self.core["H"]
        self.O = self.core["O"]
        self.h_states = [np.zeros(self.H, dtype=np.int8) for _ in range(batch_size)]
        self.scratch = np.zeros(6 * self.H, dtype=np.int32)
        self.active = [_make_runtime_struct(self.core, self.adapters["default"]) for _ in range(batch_size)]

    def _match_adapter(self, tag_str: str):
        for day, ad in self.adapters.items():
            if day != "default" and day in tag_str:
                return ad
        return self.adapters["default"]

    def reset(self, dataset_tags: List[Path] = [""]):
        self.h_states = [np.zeros(self.H, dtype=np.int8) for _ in range(max(len(dataset_tags), self.batch_size))]
        self.active = [_make_runtime_struct(self.core, self._match_adapter(str(t))) for t in dataset_tags]

    def on_done(self, dones: np.ndarray):
        for i, done in enumerate(dones):
            if done:
                self.h_states[i].fill(0)

    def _step_one(self, idx: int, obs: np.ndarray) -> np.ndarray:
        rt = self.active[idx]
        z = ((obs.astype(np.float32) - rt["x_mean"]) / self.core["x_std"]) * rt["ch_scale"] + rt["ch_bias"]
        x_q8 = np.ascontiguousarray(np.clip(np.round(z / self.core["x_scale"]), -127, 127).astype(np.int8))
        out_q12 = np.zeros(self.O, dtype=np.int32)
        self.lib.gru_int4_step(
            ctypes.byref(rt["c"]),
            x_q8.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.h_states[idx].ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            out_q12.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return (out_q12.astype(np.float32) / 4096.0) * self.core["y_std"]

    def observe(self, neural_observations: np.ndarray):
        for i in range(neural_observations.shape[0]):
            self._step_one(i, neural_observations[i])

    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        return np.stack([self._step_one(i, neural_observations[i]) for i in range(neural_observations.shape[0])], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="eval")
    p.add_argument("--data-dir", default="data/falcon_h1")
    p.add_argument("--out", default="falcon_submit/h1_int4_bundle.pkl")
    p.add_argument("--evaluation", choices=["local", "remote"], default="remote")
    p.add_argument("--model-path", default="falcon_submit/h1_int4_bundle.pkl")
    p.add_argument("--split", choices=["h1", "h2", "m1", "m2"], default="h1")
    p.add_argument("--phase", choices=["minival", "test"], default="minival")
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    if args.mode == "train":
        train_bundle(args.data_dir, args.out)
        return

    task = getattr(FalconTask, args.split)
    config = FalconConfig(task=task)
    decoder = Int4AnchoredGRUDecoder(task_config=config, model_path=args.model_path, batch_size=args.batch_size)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split)
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()
