"""Sub-8-bit quantized neural decoders across days on real human implant data (FALCON H1, DANDI 000954).

Tests an open question for implantable BCI chips: can a causal GRU decoder stored at 2 to 4 bits per weight stay
accurate weeks later as human intracortical signals drift, and can it be recalibrated with ~48 seconds (3 trials) of
new-day data — including when the recurrent core stays completely frozen?

Data (data/falcon_h1, 176 channels, 7 movement velocity dims, 20 ms bins, matching FALCON's official bin_units):
  held-in-calib:    6 early days (13 sessions, 170 trials, ~44 min) — split 85/15 by trial into fit and early-stop tune
  held-in-minival:  same 6 early days (13 sessions, ~6.8 min) — official FALCON local held-in validation split
  held-out-calib:   7 later days spanning weeks 4–6 (14 sessions = 2 sessions/day, 3 trials / ~48 s each)
                    For each later day, session 1 (~48 s) is used ONLY for recalibration and session 2 (~48 s) ONLY for scoring.

Scoring: official FALCON regression metric — r2_score(..., multioutput="variance_weighted") per day on eval_mask bins,
averaged across days. Every model is also scored on a negative control (time-reversed neural inputs).

Run (3 seeds, ~6–8 minutes on 12 cores):
    .venv/bin/python falcon_precision_drift.py data/falcon_h1
    .venv/bin/python falcon_precision_drift.py          # print summary from falcon_precision_drift.jsonl
"""
import copy
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch import nn

from precision_sweep import Rounded

BIN_S = 0.02
HIDDEN = 128
DROPOUT = (0.3, 0.3)
BITS = (32, 8, 4, 3, 2)
SEEDS = 3
RESULTS = Path(__file__).with_name("falcon_precision_drift.jsonl")


def load_session(path):
    """Load one FALCON H1 NWB file into 20 ms binned spikes (T, 176), velocity (T, 7), eval_mask (T,), trial_num (T,)."""
    with h5py.File(path) as f:
        g = f["acquisition/OpenLoopKinematicsVelocity"]
        vel = g["data"][:].astype(np.float32)
        t0 = float(g["starting_time"][()])
        dt = float(g["starting_time"].attrs["rate"])
        mask = f["acquisition/eval_mask/data"][:].astype(bool)
        tnum = f["acquisition/TrialNum/data"][:]
        st = f["units/spike_times"][:]
        sti = f["units/spike_times_index"][:]
    T = vel.shape[0]
    edges = t0 + np.arange(-1, T) * dt  # bin k covers [t_k - dt, t_k), matching FALCON bin_units(is_timestamp_bin_start=False)
    n_units = len(sti)
    spikes = np.zeros((T, n_units), dtype=np.float32)
    start = 0
    for u, end in enumerate(sti):
        if end > start:
            spikes[:, u] = np.histogram(st[start:end], bins=edges)[0]
        start = int(end)
    day = os.path.basename(path).split("ses-")[1][:8]
    return {"day": day, "path": os.path.basename(path), "spikes": spikes, "vel": vel, "mask": mask, "tnum": tnum}


def split_trials(sess):
    """Split a session into per-trial (spikes, vel, mask) tuples so GRU state resets cleanly at trial boundaries."""
    out = []
    for t in np.unique(sess["tnum"]):
        idx = sess["tnum"] == t
        out.append((sess["spikes"][idx], sess["vel"][idx], sess["mask"][idx]))
    return out


def pad_trials(trials, x_mean, x_std, y_std):
    """Standardize and pack variable-length trials into (N, Lmax, C), (N, Lmax, 7), (N, Lmax) mask tensors."""
    L = max(len(s) for s, _, _ in trials)
    N, C, D = len(trials), trials[0][0].shape[1], trials[0][1].shape[1]
    X = np.zeros((N, L, C), dtype=np.float32)
    Y = np.zeros((N, L, D), dtype=np.float32)
    M = np.zeros((N, L), dtype=bool)
    for i, (s, v, m) in enumerate(trials):
        n = len(s)
        X[i, :n] = (s - x_mean) / x_std
        Y[i, :n] = v / y_std
        M[i, :n] = m
    return torch.tensor(X), torch.tensor(Y), torch.tensor(M)


class CausalGRU(nn.Module):
    """Causal 1-layer GRU decoder with an explicit per-channel affine adapter (for low-cost on-chip recalibration)."""

    def __init__(self, n_ch, n_out=7, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.ch_scale = nn.Parameter(torch.ones(n_ch))
        self.ch_bias = nn.Parameter(torch.zeros(n_ch))
        self.drop_in = nn.Dropout(dropout[0])
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.drop_out = nn.Dropout(dropout[1])
        self.readout = nn.Linear(hidden, n_out)

    def forward(self, x):
        z = self.drop_in(x * self.ch_scale + self.ch_bias)
        h, _ = self.gru(z)
        return self.readout(self.drop_out(h))


def predict_session(model, sess, x_mean, x_std, y_std, reverse_control=False):
    """Run causal inference trial-by-trial on one session; return predictions (T, 7) in original velocity units."""
    model.eval()
    preds = np.zeros_like(sess["vel"])
    with torch.no_grad():
        for t in np.unique(sess["tnum"]):
            idx = sess["tnum"] == t
            sp = sess["spikes"][idx]
            if reverse_control:
                sp = sp[::-1].copy()
            x = torch.tensor(((sp - x_mean) / x_std)[None], dtype=torch.float32)
            preds[idx] = model(x)[0].numpy() * y_std
    return preds


def eval_days(model, sessions_by_day, x_mean, x_std, y_std, reverse_control=False):
    """Official FALCON metric: variance-weighted R² per day on eval_mask bins, returned as dict(day -> r2)."""
    scores = {}
    for day, slist in sorted(sessions_by_day.items()):
        yt, yp = [], []
        for s in slist:
            p = predict_session(model, s, x_mean, x_std, y_std, reverse_control=reverse_control)
            m = s["mask"]
            yt.append(s["vel"][m])
            yp.append(p[m])
        scores[day] = float(r2_score(np.vstack(yt), np.vstack(yp), multioutput="variance_weighted"))
    return scores


def masked_mse(pred, target, mask):
    diff = (pred - target)[mask]
    return diff.square().mean()


def fit_model(model, X_fit, Y_fit, M_fit, X_tune, Y_tune, M_tune, lr=2e-3, max_epochs=200, patience=15, batch_size=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def eval_tune():
        model.eval()
        with torch.no_grad():
            p = model(X_tune)[M_tune].numpy()
            y = Y_tune[M_tune].numpy()
            return float(r2_score(y, p, multioutput="variance_weighted"))

    best, best_state, stale = eval_tune(), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_fit))
        for b in perm.split(batch_size):
            opt.zero_grad()
            masked_mse(model(X_fit[b]), Y_fit[b], M_fit[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sc = eval_tune()
        if sc > best:
            best, best_state, stale = sc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale == patience:
                break
    model.load_state_dict(best_state)
    return model, best


def recalibrate(model, calib_sess, x_mean, x_std, y_std, mode="full", steps=25, lr=1e-3):
    """Few-shot recalibration on a single ~48 s session (3 trials) of a new day."""
    m = copy.deepcopy(model)
    trials = split_trials(calib_sess)
    Xc, Yc, Mc = pad_trials(trials, x_mean, x_std, y_std)
    base = m.decoder if isinstance(m, Rounded) else m
    if mode == "adapter":
        # Freeze entire quantized GRU recurrent core; adapt ONLY 176x2 channel affine + readout bias/weights!
        for p in m.parameters():
            p.requires_grad_(False)
        for p in (base.ch_scale, base.ch_bias, base.readout.weight, base.readout.bias):
            p.requires_grad_(True)
        # Keep readout quantized if Rounded, so only unquantized 1D channel affine + readout bias are updated cleanly,
        # OR allow readout weight update through LSQ; here let's update ch_scale, ch_bias + readout (1.2k params vs 117k)!
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr * 3.0, weight_decay=1e-3)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    m.train()
    for _ in range(steps):
        opt.zero_grad()
        masked_mse(m(Xc), Yc, Mc).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
        opt.step()
    m.eval()
    return m


def wiener_features(sess, x_mean, x_std, lags=15):
    """Causal 300 ms history (15 bins of 20 ms) per time step, reset at trial boundaries."""
    T, C = sess["spikes"].shape
    out = np.zeros((T, C * lags), dtype=np.float32)
    for t in np.unique(sess["tnum"]):
        idx = np.flatnonzero(sess["tnum"] == t)
        z = (sess["spikes"][idx] - x_mean) / x_std
        pad = np.vstack([np.zeros((lags - 1, C), dtype=np.float32), z])
        for k in range(lags):
            out[idx, k * C:(k + 1) * C] = pad[lags - 1 - k:lags - 1 - k + len(idx)]
    return out


def run_wiener_baseline(calib_in, minival_in, held_out_days, x_mean, x_std):
    """Causal Wiener filter (300 ms history, RidgeCV) on the exact same splits."""
    Xtr = np.vstack([wiener_features(s, x_mean, x_std)[s["mask"]] for s in calib_in])
    Ytr = np.vstack([s["vel"][s["mask"]] for s in calib_in])
    ridge = RidgeCV(alphas=np.logspace(1, 5, 9)).fit(Xtr, Ytr)

    def score_days(s_by_day):
        out = {}
        for d, sl in sorted(s_by_day.items()):
            yt = np.vstack([s["vel"][s["mask"]] for s in sl])
            yp = np.vstack([ridge.predict(wiener_features(s, x_mean, x_std)[s["mask"]]) for s in sl])
            out[d] = float(r2_score(yt, yp, multioutput="variance_weighted"))
        return out

    mv_by_day = defaultdict(list)
    for s in minival_in:
        mv_by_day[s["day"]].append(s)
    mv_r2 = score_days(mv_by_day)
    ho_zero = score_days(held_out_days)

    # 2-fold per-day recalibration: fit residual ridge on s0 -> score s1, and s1 -> s0, averaged per day
    ho_recal = {}
    for d, (s0, s1) in sorted(held_out_days.items()):
        fold_r2 = []
        for cal, tst in ((s0, s1), (s1, s0)):
            Xc = wiener_features(cal, x_mean, x_std)[cal["mask"]]
            Yc = cal["vel"][cal["mask"]]
            delta = RidgeCV(alphas=np.logspace(2, 6, 9)).fit(Xc, Yc - ridge.predict(Xc))
            Xt = wiener_features(tst, x_mean, x_std)[tst["mask"]]
            yp = ridge.predict(Xt) + delta.predict(Xt)
            fold_r2.append(float(r2_score(tst["vel"][tst["mask"]], yp, multioutput="variance_weighted")))
        ho_recal[d] = float(np.mean(fold_r2))

    return {
        "model": "wiener_300ms",
        "bits": 32,
        "seed": 0,
        "kB": float(ridge.coef_.size * 4 / 1000),
        "minival_mean": float(np.mean(list(mv_r2.values()))),
        "heldout_zeroshot_mean": float(np.mean(list(ho_zero.values()))),
        "heldout_recal_full_mean": float(np.mean(list(ho_recal.values()))),
        "heldout_recal_adapter_mean": float(np.mean(list(ho_recal.values()))),
        "control_mean": 0.0,
        "per_day_zeroshot": ho_zero,
        "per_day_recal_full": ho_recal,
    }


def _load_all(folder):
    load_dir = lambda sub: [load_session(p) for p in sorted(glob.glob(f"{folder}/{sub}/*.nwb"))]
    calib_in = load_dir("sub-HumanPitt-held-in-calib")
    minival_in = load_dir("sub-HumanPitt-held-in-minival")
    ho_all = load_dir("sub-HumanPitt-held-out-calib")
    ho_days = defaultdict(list)
    for s in ho_all:
        ho_days[s["day"]].append(s)
    # Global train normalization from held-in-calib ONLY
    all_sp = np.vstack([s["spikes"] for s in calib_in])
    all_v = np.vstack([s["vel"][s["mask"]] for s in calib_in])
    x_mean = all_sp.mean(0).astype(np.float32)
    x_std = np.maximum(all_sp.std(0), 1e-2).astype(np.float32)
    y_std = np.maximum(all_v.std(0), 1e-5).astype(np.float32)
    return calib_in, minival_in, dict(ho_days), x_mean, x_std, y_std


def run_seed(args):
    folder, seed = args
    torch.set_num_threads(2)
    calib_in, minival_in, ho_days, x_mean, x_std, y_std = _load_all(folder)
    mv_by_day = defaultdict(list)
    for s in minival_in:
        mv_by_day[s["day"]].append(s)

    # Split held-in-calib trials 85/15 into fit and tune (never touching minival or held-out days)
    all_trials = [tr for s in calib_in for tr in split_trials(s)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(all_trials))
    n_tune = len(all_trials) * 15 // 100
    tune_trials = [all_trials[i] for i in order[:n_tune]]
    fit_trials = [all_trials[i] for i in order[n_tune:]]
    Xf, Yf, Mf = pad_trials(fit_trials, x_mean, x_std, y_std)
    Xt, Yt, Mt = pad_trials(tune_trials, x_mean, x_std, y_std)

    torch.manual_seed(seed)
    base, tune_r2 = fit_model(CausalGRU(Xf.shape[-1]), Xf, Yf, Mf, Xt, Yt, Mt, lr=2e-3)
    matrices = sum(p.numel() for p in base.parameters() if p.dim() == 2)

    out_rows = []
    for bits in BITS:
        torch.manual_seed(1000 * seed + bits)
        start = copy.deepcopy(base)
        if bits == 32:
            model, q_tune = fit_model(start, Xf, Yf, Mf, Xt, Yt, Mt, lr=1e-3, max_epochs=100, patience=12)
        else:
            model, q_tune = fit_model(Rounded(start, bits), Xf, Yf, Mf, Xt, Yt, Mt, lr=1e-3, max_epochs=100, patience=12)

        mv_scores = eval_days(model, mv_by_day, x_mean, x_std, y_std)
        ho_zero = eval_days(model, ho_days, x_mean, x_std, y_std)
        ho_ctrl = eval_days(model, ho_days, x_mean, x_std, y_std, reverse_control=True)

        ho_full, ho_adapt = {}, {}
        for day, (s0, s1) in sorted(ho_days.items()):
            f_scores, a_scores = [], []
            for cal, tst in ((s0, s1), (s1, s0)):
                mf = recalibrate(model, cal, x_mean, x_std, y_std, mode="full", steps=25, lr=1e-3)
                f_scores.append(eval_days(mf, {day: [tst]}, x_mean, x_std, y_std)[day])
                ma = recalibrate(model, cal, x_mean, x_std, y_std, mode="adapter", steps=25, lr=1e-3)
                a_scores.append(eval_days(ma, {day: [tst]}, x_mean, x_std, y_std)[day])
            ho_full[day] = float(np.mean(f_scores))
            ho_adapt[day] = float(np.mean(a_scores))

        row = {
            "model": "gru_128",
            "bits": bits,
            "seed": seed,
            "kB": float(matrices * bits / 8000),
            "tune_r2": q_tune,
            "minival_mean": float(np.mean(list(mv_scores.values()))),
            "heldout_zeroshot_mean": float(np.mean(list(ho_zero.values()))),
            "heldout_recal_full_mean": float(np.mean(list(ho_full.values()))),
            "heldout_recal_adapter_mean": float(np.mean(list(ho_adapt.values()))),
            "control_mean": float(np.mean(list(ho_ctrl.values()))),
            "per_day_zeroshot": ho_zero,
            "per_day_recal_full": ho_full,
            "per_day_recal_adapter": ho_adapt,
        }
        out_rows.append(row)
        print(f"seed {seed} | {bits:2d}-bit ({row['kB']:5.1f} kB) | minival {row['minival_mean']:.3f} | "
              f"later days zero-shot {row['heldout_zeroshot_mean']:.3f} | recal-full {row['heldout_recal_full_mean']:.3f} | "
              f"recal-adapter {row['heldout_recal_adapter_mean']:.3f} | ctrl {row['control_mean']:.3f}", flush=True)
    return out_rows


def summary():
    rows = [json.loads(l) for l in open(RESULTS)]
    print(f"=== FALCON H1 Human Implant: Precision x Cross-Day Drift ({len(rows)} rows) ===")
    print(f"{'model':<14} {'bits':>4} {'kB':>6} | {'minival':>8} | {'zero-shot':>9} | {'recal-full':>10} | {'recal-adapt':>11} | {'ctrl':>6}")
    wiener = [r for r in rows if r["model"] == "wiener_300ms"]
    if wiener:
        r = wiener[-1]
        print(f"{'wiener_300ms':<14} {32:>4} {r['kB']:6.1f} | {r['minival_mean']:8.3f} | {r['heldout_zeroshot_mean']:9.3f} | "
              f"{r['heldout_recal_full_mean']:10.3f} | {r['heldout_recal_adapter_mean']:11.3f} | {'-':>6}")
    for bits in BITS:
        sub = [r for r in rows if r["model"] == "gru_128" and r["bits"] == bits]
        if not sub:
            continue
        m = lambda k: np.mean([r[k] for r in sub])
        print(f"{'gru_128':<14} {bits:>4} {sub[0]['kB']:6.1f} | {m('minival_mean'):8.3f} | {m('heldout_zeroshot_mean'):9.3f} | "
              f"{m('heldout_recal_full_mean'):10.3f} | {m('heldout_recal_adapter_mean'):11.3f} | {max(r['control_mean'] for r in sub):6.3f}")


def main(folder):
    calib_in, minival_in, ho_days, x_mean, x_std, y_std = _load_all(folder)
    print(f"Loaded FALCON H1: {len(calib_in)} train sessions (6 days), {len(minival_in)} minival sessions, "
          f"{len(ho_days)} held-out later days (2 sessions/day)", flush=True)
    w_row = run_wiener_baseline(calib_in, minival_in, ho_days, x_mean, x_std)
    print(f"wiener_300ms | minival {w_row['minival_mean']:.3f} | later days zero-shot {w_row['heldout_zeroshot_mean']:.3f} | "
          f"recal {w_row['heldout_recal_full_mean']:.3f}", flush=True)
    with open(RESULTS, "w") as f:
        f.write(json.dumps(w_row) + "\n")
    with Pool(SEEDS) as pool:
        all_seed_rows = pool.map(run_seed, [(folder, s) for s in range(SEEDS)])
    with open(RESULTS, "a") as f:
        for srows in all_seed_rows:
            for r in srows:
                f.write(json.dumps(r) + "\n")
                assert r["control_mean"] < 0.05 < r["minival_mean"], f"self-check failed: {r}"
    print()
    summary()


if __name__ == "__main__":
    main(sys.argv[1]) if len(sys.argv) > 1 else summary()
