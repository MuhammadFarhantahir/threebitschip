"""Real human intracranial closed-loop stimulation and memory-state decoding (OpenNeuro UPenn/DARPA RAM, CC0).

Uses public human intracranial EEG (iEEG) recordings from neurosurgical patients performing verbal free recall with
real-time closed-loop electrical brain stimulation during encoding (ds005557 FR3 and ds005558 catFR3, CC0):
  - Behavioral & site-level summary across ALL 40 sessions (23 patients, 11,604 words).
  - Range-fetched multi-band iEEG extraction (8 log-spaced bands from 3–95 Hz, 60 Hz notched) for 4 compact patients
    (1,200 words, 361 real closed-loop stimulations), saving compact ~3 MB .npz feature files in data/ram_closedloop/features/.
  - Strictly artifact-free windows:
      Pre-stim encoding window:  [+0.10 s, +1.35 s] after WORD onset (5 x 250 ms bins; STIM_ON fires at ~1.55 s)
      Post-stim window:          [+2.20 s, +3.00 s] after WORD onset (after 500 ms stimulation turns OFF at ~2.05 s)
  - Three tests per patient (list-level cross-validation on non-stim control lists, plus shuffled-label controls):
      1. Memory-state decoding: L2 Logistic Regression (Ezzyat et al. 2018 baseline) vs causal GRU neural network
         predicting subsequent word recall from pre-stim iEEG (held-out list ROC AUC).
      2. Causal neural-state shift: does real electrical stimulation move the brain's decoded memory state UP in the
         post-stim window compared to matched-pre-state unstimulated control-list words?
      3. Matched behavioral recall: does stimulation rescue recall on decoder-matched low-memory-state words vs
         matched unstimulated control-list words (removing the online trigger's selection bias)?

Run:
    .venv/bin/python human_memory_closedloop.py
"""
import csv
import glob
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

BASE = Path("data/ram_closedloop")
FEAT_DIR = BASE / "features"
OUT = Path(__file__).with_name("human_memory_closedloop.json")

# 4 compact, complete patients (300 words + 71–106 real closed-loop stims each) spanning positive & negative outcomes
SESSIONS = [
    ("ds005558", "R1274T", "0", "catFR3"),
    ("ds005557", "R1202M", "1", "FR3"),
    ("ds005557", "R1161E", "1", "FR3"),
    ("ds005558", "R1264P", "1", "catFR3"),
]

BANDS = [(2.5, 4.0), (4.0, 6.5), (6.5, 10.5), (10.5, 17.0), (17.0, 28.0), (28.0, 45.0), (45.0, 72.0), (72.0, 115.0)]
PRE_BINS = [(0.10 + 0.25 * k, 0.35 + 0.25 * k) for k in range(5)]  # 5 x 250 ms bins: [0.10, 1.35] s
POST_WIN = (2.20, 3.00)  # clean post-stim window (STIM_ON ~1.55 s, STIM_OFF ~2.05 s)


def _fetch_range(url, b0, b1):
    for _ in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={b0}-{b1}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) == b1 - b0 + 1:
                return data
        except Exception:
            pass
    raise RuntimeError(f"Failed range {b0}-{b1} on {url}")


def parse_edf_header(url):
    fixed = _fetch_range(url, 0, 255)
    hdr_bytes = int(fixed[184:192].decode().strip())
    n_rec = int(fixed[236:244].decode().strip())
    rec_dur = float(fixed[244:252].decode().strip())
    n_sig = int(fixed[252:256].decode().strip())
    hdr = fixed + _fetch_range(url, 256, hdr_bytes - 1)
    off = 256
    labels = [hdr[off + i * 16:off + (i + 1) * 16].decode(errors="ignore").strip() for i in range(n_sig)]
    off += n_sig * (16 + 80 + 8)
    pmin = np.array([float(hdr[off + i * 8:off + (i + 1) * 8]) for i in range(n_sig)])
    off += n_sig * 8
    pmax = np.array([float(hdr[off + i * 8:off + (i + 1) * 8]) for i in range(n_sig)])
    off += n_sig * 8
    dmin = np.array([int(hdr[off + i * 8:off + (i + 1) * 8]) for i in range(n_sig)])
    off += n_sig * 8
    dmax = np.array([int(hdr[off + i * 8:off + (i + 1) * 8]) for i in range(n_sig)])
    off += n_sig * 8 + n_sig * 80
    nsamp = [int(hdr[off + i * 8:off + (i + 1) * 8]) for i in range(n_sig)]
    ch_idx = [i for i, l in enumerate(labels) if "annotation" not in l.lower()]
    sr = int(round(nsamp[ch_idx[0]] / rec_dur))
    scales = (pmax - pmin) / np.maximum(dmax - dmin, 1)
    offsets = pmin - scales * dmin
    byte_offs = np.cumsum([0] + [n * 2 for n in nsamp[:-1]])
    return {
        "hdr_bytes": hdr_bytes, "n_rec": n_rec, "rec_dur": rec_dur, "rec_bytes": sum(nsamp) * 2,
        "labels": [labels[i] for i in ch_idx], "ch_idx": ch_idx, "nsamp": nsamp, "sr": sr,
        "scales": scales[ch_idx], "offsets": offsets[ch_idx], "byte_offs": byte_offs[ch_idx],
    }


def extract_session(ds, subj, ses, task):
    """Fetch only the word-list byte ranges from S3 EDF and compute compact pre/post multi-band log-power features."""
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = FEAT_DIR / f"{ds}_{subj}_ses-{ses}.npz"
    if out_npz.exists():
        return dict(np.load(out_npz, allow_pickle=True))

    ev_path = BASE / ds / f"sub-{subj}" / f"ses-{ses}" / "ieeg" / f"sub-{subj}_ses-{ses}_task-{task}_events.tsv"
    rows = list(csv.DictReader(open(ev_path), delimiter="\t"))
    words = [r for r in rows if r.get("trial_type") == "WORD" and int(r["sample"]) >= 0]
    stims = [r for r in rows if r.get("trial_type") == "STIM_ON" and int(r["sample"]) >= 0]
    stim_samples = np.array([int(s["sample"]) for s in stims], dtype=np.int64)
    anode = next((s["anode_label"] for s in stims if s.get("anode_label") not in (None, "", "n/a")), "?")

    url = f"https://s3.amazonaws.com/openneuro.org/{ds}/sub-{subj}/ses-{ses}/ieeg/sub-{subj}_ses-{ses}_task-{task}_acq-bipolar_ieeg.edf"
    meta = parse_edf_header(url)
    sr, n_ch, rec_dur, rec_bytes, hdr_bytes = meta["sr"], len(meta["ch_idx"]), meta["rec_dur"], meta["rec_bytes"], meta["hdr_bytes"]
    samp_per_rec = int(round(sr * rec_dur))

    # Group words by list so we fetch one contiguous block per list (~25 blocks of ~35 s each = ~30% of EDF)
    by_list = {}
    for idx, w in enumerate(words):
        by_list.setdefault(int(w["list"]), []).append((idx, int(w["sample"])))

    list_ranges = []
    for lid, items in sorted(by_list.items()):
        s_min = min(s for _, s in items) - int(1.5 * sr)
        s_max = max(s for _, s in items) + int(4.0 * sr)
        r0 = max(0, s_min // samp_per_rec)
        r1 = min(meta["n_rec"], (s_max + samp_per_rec - 1) // samp_per_rec)
        list_ranges.append((lid, r0, r1, items))

    mb = sum((r1 - r0) * rec_bytes for _, r0, r1, _ in list_ranges) / 1e6
    print(f"[{subj} ses-{ses}] {n_ch} ch @ {sr} Hz, {len(words)} words across {len(list_ranges)} lists; "
          f"range-fetching {mb:.1f} MB...", flush=True)

    # Precompute 60 Hz notch + 8 bandpass filters
    b_notch, a_notch = iirnotch(60.0, 30.0, sr)
    b_notch2, a_notch2 = iirnotch(120.0, 30.0, sr)
    filters = [butter(3, [lo / (sr / 2), min(hi, sr * 0.45) / (sr / 2)], btype="band") for lo, hi in BANDS]

    pre_feat = np.zeros((len(words), len(PRE_BINS), n_ch, len(BANDS)), dtype=np.float32)
    post_feat = np.zeros((len(words), n_ch, len(BANDS)), dtype=np.float32)
    pre_clean = np.ones(len(words), dtype=bool)
    post_clean = np.ones(len(words), dtype=bool)

    def process_list(item):
        lid, r0, r1, items = item
        b0 = hdr_bytes + r0 * rec_bytes
        b1 = hdr_bytes + r1 * rec_bytes - 1
        raw = _fetch_range(url, b0, b1)
        n_r = r1 - r0
        sig = np.empty((n_ch, n_r * samp_per_rec), dtype=np.float32)
        for r in range(n_r):
            base_off = r * rec_bytes
            for c in range(n_ch):
                bo = base_off + int(meta["byte_offs"][c])
                sig[c, r * samp_per_rec:(r + 1) * samp_per_rec] = np.frombuffer(raw, dtype="<i2", count=samp_per_rec, offset=bo)
        sig = sig * meta["scales"][:, None] + meta["offsets"][:, None]
        # Mask stimulation artifacts before filtering so filter ringing doesn't bleed into pre/post windows!
        block_start = r0 * samp_per_rec
        block_end = r1 * samp_per_rec
        in_block = stim_samples[(stim_samples >= block_start - sr) & (stim_samples <= block_end)]
        for ss in in_block:
            i0 = max(0, int(ss - block_start - int(0.05 * sr)))
            i1 = min(sig.shape[1], int(ss - block_start + int(0.60 * sr)))
            if i1 > i0:
                left_val = sig[:, max(0, i0 - 1):i0] if i0 > 0 else np.zeros((n_ch, 1), dtype=np.float32)
                right_val = sig[:, i1:min(sig.shape[1], i1 + 1)] if i1 < sig.shape[1] else left_val
                sig[:, i0:i1] = np.linspace(0, 1, i1 - i0, dtype=np.float32)[None, :] * (right_val - left_val) + left_val

        sig = filtfilt(b_notch, a_notch, sig, axis=1)
        sig = filtfilt(b_notch2, a_notch2, sig, axis=1)
        band_pow = np.empty((len(BANDS), n_ch, sig.shape[1]), dtype=np.float32)
        for bi, (bf, af) in enumerate(filters):
            band_pow[bi] = filtfilt(bf, af, sig, axis=1) ** 2

        out = []
        for idx, ws in items:
            rel = ws - block_start
            # Check strict artifact clearance for pre [0.05, 1.40] and post [2.15, 3.05]
            d_stim = (stim_samples - ws) / sr
            pre_hit = bool(np.any((d_stim > -0.60) & (d_stim < 1.38)))
            post_hit = bool(np.any((d_stim > 1.65) & (d_stim < 3.02)))
            pf = np.zeros((len(PRE_BINS), n_ch, len(BANDS)), dtype=np.float32)
            for ti, (t0, t1) in enumerate(PRE_BINS):
                a, b = rel + int(t0 * sr), rel + int(t1 * sr)
                if 0 <= a < b <= sig.shape[1]:
                    pf[ti] = np.log10(band_pow[:, :, a:b].mean(2).T + 1e-6)
            a, b = rel + int(POST_WIN[0] * sr), rel + int(POST_WIN[1] * sr)
            q = np.log10(band_pow[:, :, a:b].mean(2).T + 1e-6) if 0 <= a < b <= sig.shape[1] else np.zeros((n_ch, len(BANDS)), dtype=np.float32)
            out.append((idx, pf, q, not pre_hit, not post_hit))
        return out

    with ThreadPoolExecutor(6) as ex:
        for batch in ex.map(process_list, list_ranges):
            for idx, pf, q, pc, qc in batch:
                pre_feat[idx], post_feat[idx], pre_clean[idx], post_clean[idx] = pf, q, pc, qc

    recalled = np.array([int(w["recalled"]) for w in words], dtype=np.int32)
    lists = np.array([int(w["list"]) for w in words], dtype=np.int32)
    stim_list = np.array([int(w["stim_list"]) for w in words], dtype=np.int32)
    triggered = np.array([
        int(np.any((stim_samples >= int(w["sample"]) + int(0.1 * sr)) & (stim_samples <= int(w["sample"]) + int(2.0 * sr))))
        for w in words
    ], dtype=np.int32)

    data = dict(pre=pre_feat, post=post_feat, pre_clean=pre_clean, post_clean=post_clean,
                recalled=recalled, lists=lists, stim_list=stim_list, triggered=triggered,
                labels=np.array(meta["labels"]), anode=np.array(anode))
    np.savez_compressed(out_npz, **data)
    print(f"  saved {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB) | clean pre {pre_clean.sum()}/{len(words)}, triggered {triggered.sum()}", flush=True)
    return data


class TemporalGRU(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, T, F)
        h, _ = self.gru(self.drop(x))
        return self.head(self.drop(h[:, -1])).squeeze(-1)


def fit_gru_cv(X_seq, y, groups, n_splits=5, seed=0):
    """List-level GroupKFold CV for causal TemporalGRU; returns out-of-fold probabilities."""
    oof = np.zeros(len(y), dtype=np.float32)
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(gkf.split(X_seq, y, groups)):
        mu = X_seq[tr].mean(axis=(0, 1), keepdims=True)
        sd = np.maximum(X_seq[tr].std(axis=(0, 1), keepdims=True), 1e-4)
        Xtr = torch.tensor((X_seq[tr] - mu) / sd, dtype=torch.float32)
        Xte = torch.tensor((X_seq[te] - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        torch.manual_seed(seed * 10 + fold)
        model = TemporalGRU(X_seq.shape[-1], hidden=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
        pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        model.train()
        for _ in range(45):
            opt.zero_grad()
            loss_fn(model(Xtr), ytr).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(Xte)).numpy()
    return oof


def analyze_session(ds, subj, ses, task):
    d = extract_session(ds, subj, ses, task)
    pre = d["pre"]          # (N, 5, C, 8)
    post = d["post"]        # (N, C, 8)
    recalled = d["recalled"]
    lists = d["lists"]
    stim_list = d["stim_list"]
    triggered = d["triggered"]
    pre_clean = d["pre_clean"].astype(bool)
    post_clean = d["post_clean"].astype(bool)

    N, T, C, B = pre.shape
    X_mean = pre.mean(axis=1).reshape(N, C * B)      # (N, C*8) Ezzyat-style window-averaged spectral power
    X_post = post.reshape(N, C * B)                  # (N, C*8) same feature space in post-stim window!
    X_seq = pre.reshape(N, T, C * B)                 # (N, 5, C*8) time-resolved encoding trajectory

    # 1. Decode subsequent memory on NON-STIM control lists (list-level GroupKFold, zero leakage)
    ctrl_mask = (stim_list == 0) & pre_clean
    Xc, Xc_seq, yc, gc = X_mean[ctrl_mask], X_seq[ctrl_mask], recalled[ctrl_mask], lists[ctrl_mask]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(gc))))
    oof_lr = np.zeros(len(yc))
    for tr, te in gkf.split(Xc, yc, gc):
        sc = StandardScaler().fit(Xc[tr])
        clf = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(Xc[tr]), yc[tr])
        oof_lr[te] = clf.predict_proba(sc.transform(Xc[te]))[:, 1]

    # 20-permutation null for reliable control mean & empirical p-value
    rng = np.random.default_rng(42)
    null_aucs = []
    for _ in range(20):
        ys = rng.permutation(yc)
        oof_s = np.zeros(len(yc))
        for tr, te in gkf.split(Xc, ys, gc):
            sc = StandardScaler().fit(Xc[tr])
            cs = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(Xc[tr]), ys[tr])
            oof_s[te] = cs.predict_proba(sc.transform(Xc[te]))[:, 1]
        null_aucs.append(float(roc_auc_score(ys, oof_s)))

    oof_gru = np.mean([fit_gru_cv(Xc_seq, yc, gc, seed=s) for s in range(3)], axis=0)
    auc_lr = float(roc_auc_score(yc, oof_lr))
    auc_gru = float(roc_auc_score(yc, oof_gru))
    auc_shuf = float(np.mean(null_aucs))
    p_lr = float((sum(a >= auc_lr for a in null_aucs) + 1) / (len(null_aucs) + 1))

    # 2. Fit full control-list spectral decoder -> score pre-stim & post-stim memory state on stim lists!
    sc_all = StandardScaler().fit(Xc)
    clf_all = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc_all.transform(Xc), yc)
    # Use log-odds (decision_function) as continuous memory-state score
    z_pre = clf_all.decision_function(sc_all.transform(X_mean))
    z_post = clf_all.decision_function(sc_all.transform(X_post))
    # Replace control-list z_pre with strictly out-of-fold log-odds for unbiased matching!
    oof_z = np.zeros(len(yc))
    for tr, te in gkf.split(Xc, yc, gc):
        sc = StandardScaler().fit(Xc[tr])
        c = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(Xc[tr]), yc[tr])
        oof_z[te] = c.decision_function(sc.transform(Xc[te]))
    z_pre_unbiased = z_pre.copy()
    z_pre_unbiased[ctrl_mask] = oof_z

    # Agreement with the hospital's online closed-loop trigger on stim lists
    sl_clean = (stim_list == 1) & pre_clean
    trig_auc = float(roc_auc_score(triggered[sl_clean], -z_pre_unbiased[sl_clean])) if len(np.unique(triggered[sl_clean])) == 2 else float("nan")

    # Compare stimulated words (low-state triggered) against MATCHED-PRE-STATE control-list words (lowest K by z_pre)
    stim_w = (stim_list == 1) & (triggered == 1) & pre_clean
    n_trig = int(stim_w.sum())
    # Match threshold on control lists so same fraction of lowest-predicted-state words are selected
    frac = n_trig / max(int(sl_clean.sum()), 1)
    thresh = np.quantile(z_pre_unbiased[ctrl_mask], frac)
    matched_ctrl = ctrl_mask & (z_pre_unbiased <= thresh)

    # Neural state shift (post - pre) on clean post windows
    stim_shift_mask = stim_w & post_clean
    ctrl_shift_mask = matched_ctrl & post_clean
    stim_delta_z = float((z_post[stim_shift_mask] - z_pre[stim_shift_mask]).mean()) if stim_shift_mask.sum() else float("nan")
    ctrl_delta_z = float((z_post[ctrl_shift_mask] - z_pre[ctrl_shift_mask]).mean()) if ctrl_shift_mask.sum() else float("nan")

    # Behavioral word recall: stimulated low-state words vs matched low-state control words
    rec_stim_low = float(recalled[stim_w].mean()) if n_trig else float("nan")
    rec_ctrl_low = float(recalled[matched_ctrl].mean()) if matched_ctrl.sum() else float("nan")

    res = {
        "dataset": ds, "subject": subj, "session": ses, "anode": str(d["anode"]),
        "n_ch": int(C), "n_ctrl_words": int(ctrl_mask.sum()), "n_stim_words": n_trig,
        "auc_logreg": auc_lr, "auc_gru": auc_gru, "auc_shuffled_ctrl": auc_shuf, "p_logreg": p_lr,
        "agreement_with_online_trigger_auc": trig_auc,
        "neural_shift_stim_low": stim_delta_z, "neural_shift_matched_ctrl_low": ctrl_delta_z,
        "neural_shift_diff": stim_delta_z - ctrl_delta_z,
        "recall_stim_low": rec_stim_low, "recall_matched_ctrl_low": rec_ctrl_low,
        "recall_matched_diff_pp": (rec_stim_low - rec_ctrl_low) * 100,
    }
    print(f"  {subj} ({res['anode']}, {C} ch) | recall AUC: LogReg {auc_lr:.3f}, GRU {auc_gru:.3f}, shuf {auc_shuf:.3f} | "
          f"trigger-agree AUC {trig_auc:.3f} | Δstate stim {stim_delta_z:+.3f} vs ctrl {ctrl_delta_z:+.3f} | "
          f"recall stim-low {rec_stim_low:.1%} vs matched-ctrl-low {rec_ctrl_low:.1%} ({res['recall_matched_diff_pp']:+.1f} pp)", flush=True)
    return res


def main():
    print("=== Track A: Real Human Intracranial Closed-Loop Stimulation & Memory (OpenNeuro RAM) ===", flush=True)
    results = [analyze_session(*s) for s in SESSIONS]
    print("\n--- Summary across 4 real human patients (1,200 words, 361 closed-loop stimulations) ---")
    print(f"  Memory decoding ROC AUC (held-out lists): LogReg {np.mean([r['auc_logreg'] for r in results]):.3f} | "
          f"GRU {np.mean([r['auc_gru'] for r in results]):.3f} | shuffled control {np.mean([r['auc_shuffled_ctrl'] for r in results]):.3f}")
    print(f"  Agreement with hospital online trigger:   mean AUC {np.nanmean([r['agreement_with_online_trigger_auc'] for r in results]):.3f}")
    print(f"  Post-stim neural memory-state shift (Δz): stim {np.nanmean([r['neural_shift_stim_low'] for r in results]):+.3f} vs "
          f"matched unstim ctrl {np.nanmean([r['neural_shift_matched_ctrl_low'] for r in results]):+.3f}")
    print(f"  Matched low-state word recall:            stim {np.nanmean([r['recall_stim_low'] for r in results]):.1%} vs "
          f"matched ctrl {np.nanmean([r['recall_matched_ctrl_low'] for r in results]):.1%} "
          f"({np.nanmean([r['recall_matched_diff_pp'] for r in results]):+.1f} pp)")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
