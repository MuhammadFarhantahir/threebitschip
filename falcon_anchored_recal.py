"""Upgrade to Track C (FALCON H1 human implant): test whether unsupervised channel mean-alignment + L2 weight-anchored
48-second recalibration eliminates noisy-day overfitting across all precisions (32, 8, 4, 3, 2 bits).

In falcon_precision_drift.py, unconstrained 48 s fine-tuning overfit a noisy calibration day (1925-01-27: -0.280 R²).
Here we add two chip-practical safeguards (with fixed principled settings, never tuned on the 7 held-out days):
  1. Unsupervised firing-rate alignment: recenter each new day's channels using ONLY unlabeled session-1 mean firing rates
     (keeping training std), testing how much cross-day drift is pure baseline-rate shift.
  2. Weight-anchored QAT recalibration: add L2 anchoring lam * ||theta - theta_0||^2 back to the trained day 1-6 weights
     during 48 s recalibration (preventing 3 noisy trials from pulling the decoder away from its trained manifold).

Run (~4 minutes on 3 seeds):
    .venv/bin/python falcon_anchored_recal.py data/falcon_h1
"""
import copy
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

from falcon_precision_drift import BITS, SEEDS, CausalGRU, _load_all, eval_days, fit_model, masked_mse, pad_trials, split_trials
from precision_sweep import Rounded

OUT = Path(__file__).with_name("falcon_anchored_recal.jsonl")
ANCHOR_LAM = 0.05   # principled weight-anchoring strength (not tuned on held-out days)
RECAL_STEPS = 30
RECAL_LR = 8e-4


def recalibrate_anchored(model, calib_sess, x_mean, x_std, y_std, mode="full"):
    """48-second recalibration with L2 anchoring to the pre-trained weights + unsupervised channel mean alignment."""
    m = copy.deepcopy(model)
    # Unsupervised channel mean shift from the calibration session (no movement labels needed!)
    day_mean = calib_sess["spikes"].mean(0).astype(np.float32)
    trials = split_trials(calib_sess)
    Xc, Yc, Mc = pad_trials(trials, day_mean, x_std, y_std)

    base = m.decoder if isinstance(m, Rounded) else m
    if mode == "adapter":
        for p in m.parameters():
            p.requires_grad_(False)
        for p in (base.ch_scale, base.ch_bias, base.readout.weight, base.readout.bias):
            p.requires_grad_(True)
        params = [p for p in m.parameters() if p.requires_grad]
        lr = RECAL_LR * 2.5
    else:
        params = [p for p in m.parameters() if p.requires_grad]
        lr = RECAL_LR

    init = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train()
    for _ in range(RECAL_STEPS):
        opt.zero_grad()
        loss = masked_mse(m(Xc), Yc, Mc)
        anchor = sum((p - p0).square().mean() for p, p0 in zip(params, init))
        (loss + ANCHOR_LAM * anchor).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.5)
        opt.step()
    m.eval()
    return m, day_mean


def run_seed(args):
    folder, seed = args
    torch.set_num_threads(2)
    calib_in, minival_in, ho_days, x_mean, x_std, y_std = _load_all(folder)
    mv_by_day = defaultdict(list)
    for s in minival_in:
        mv_by_day[s["day"]].append(s)

    all_trials = [tr for s in calib_in for tr in split_trials(s)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_trials))
    n_tune = len(all_trials) * 15 // 100
    tune_trials = [all_trials[i] for i in order[:n_tune]]
    fit_trials = [all_trials[i] for i in order[n_tune:]]
    Xf, Yf, Mf = pad_trials(fit_trials, x_mean, x_std, y_std)
    Xt, Yt, Mt = pad_trials(tune_trials, x_mean, x_std, y_std)

    torch.manual_seed(seed)
    base, _ = fit_model(CausalGRU(Xf.shape[-1]), Xf, Yf, Mf, Xt, Yt, Mt, lr=2e-3)
    matrices = sum(p.numel() for p in base.parameters() if p.dim() == 2)

    out_rows = []
    for bits in BITS:
        torch.manual_seed(1000 * seed + bits)
        start = copy.deepcopy(base)
        if bits == 32:
            model, _ = fit_model(start, Xf, Yf, Mf, Xt, Yt, Mt, lr=1e-3, max_epochs=100, patience=12)
        else:
            model, _ = fit_model(Rounded(start, bits), Xf, Yf, Mf, Xt, Yt, Mt, lr=1e-3, max_epochs=100, patience=12)

        mv_scores = eval_days(model, mv_by_day, x_mean, x_std, y_std)
        ho_zero = eval_days(model, ho_days, x_mean, x_std, y_std)
        ho_ctrl = eval_days(model, ho_days, x_mean, x_std, y_std, reverse_control=True)

        ho_unsup, ho_full, ho_adapt = {}, {}, {}
        for day, (s0, s1) in sorted(ho_days.items()):
            u_sc, f_sc, a_sc = [], [], []
            for cal, tst in ((s0, s1), (s1, s0)):
                day_mean = cal["spikes"].mean(0).astype(np.float32)
                u_sc.append(eval_days(model, {day: [tst]}, day_mean, x_std, y_std)[day])
                mf, dm = recalibrate_anchored(model, cal, x_mean, x_std, y_std, mode="full")
                f_sc.append(eval_days(mf, {day: [tst]}, dm, x_std, y_std)[day])
                ma, dm = recalibrate_anchored(model, cal, x_mean, x_std, y_std, mode="adapter")
                a_sc.append(eval_days(ma, {day: [tst]}, dm, x_std, y_std)[day])
            ho_unsup[day] = float(np.mean(u_sc))
            ho_full[day] = float(np.mean(f_sc))
            ho_adapt[day] = float(np.mean(a_sc))

        row = {
            "bits": bits, "seed": seed, "kB": float(matrices * bits / 8000),
            "minival_mean": float(np.mean(list(mv_scores.values()))),
            "zeroshot_mean": float(np.mean(list(ho_zero.values()))),
            "unsup_align_mean": float(np.mean(list(ho_unsup.values()))),
            "anchored_full_mean": float(np.mean(list(ho_full.values()))),
            "anchored_adapter_mean": float(np.mean(list(ho_adapt.values()))),
            "control_mean": float(np.mean(list(ho_ctrl.values()))),
            "per_day_unsup": ho_unsup, "per_day_full": ho_full, "per_day_adapter": ho_adapt,
        }
        out_rows.append(row)
        print(f"seed {seed} | {bits:2d}-bit ({row['kB']:5.1f} kB) | minival {row['minival_mean']:.3f} | "
              f"zero {row['zeroshot_mean']:.3f} | unsup-mean {row['unsup_align_mean']:.3f} | "
              f"anch-full {row['anchored_full_mean']:.3f} | anch-adapt {row['anchored_adapter_mean']:.3f} | ctrl {row['control_mean']:.3f}", flush=True)
    return out_rows


def summary():
    rows = [json.loads(l) for l in open(OUT)]
    print(f"\n=== FALCON H1 Human Implant: Weight-Anchored 48s Recalibration ({len(rows)} runs) ===")
    print(f"{'bits':>4} {'kB':>6} | {'minival':>7} | {'zero-shot':>9} | {'unsup-mean':>10} | {'anch-full':>9} | {'anch-adapt':>10} | {'ctrl':>6}")
    for bits in BITS:
        sub = [r for r in rows if r["bits"] == bits]
        m = lambda k: np.mean([r[k] for r in sub])
        print(f"{bits:>4} {sub[0]['kB']:6.1f} | {m('minival_mean'):7.3f} | {m('zeroshot_mean'):9.3f} | "
              f"{m('unsup_align_mean'):10.3f} | {m('anchored_full_mean'):9.3f} | {m('anchored_adapter_mean'):10.3f} | "
              f"{max(r['control_mean'] for r in sub):6.3f}")


def main(folder):
    with Pool(SEEDS) as pool:
        all_rows = pool.map(run_seed, [(folder, s) for s in range(SEEDS)])
    with open(OUT, "w") as f:
        for srows in all_rows:
            for r in srows:
                f.write(json.dumps(r) + "\n")
                assert r["control_mean"] < 0.05 < r["minival_mean"]
    summary()


if __name__ == "__main__":
    main(sys.argv[1]) if len(sys.argv) > 1 else summary()
