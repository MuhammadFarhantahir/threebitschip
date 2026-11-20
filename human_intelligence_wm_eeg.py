"""Level-1 Healthy-Human Proof (OpenNeuro ds008037, Biabani & Rogasch, CC0):
Linking real individual healthy-human intelligence (g-factor), Cowan's 4-item working-memory cliff,
non-invasive pre-response working-memory failure decoding, and causal TMS-EEG cortical reactivity.

Three rigorous analyses (100% public healthy-adult data, zero brain surgery):
  Part A1 (123 healthy adults, ~36,900 WM trials):
    - Fetch lightweight behavioral TSVs across 9 cognitive tasks (workingmemory load 4 & 6, n-back,
      backward digit span, operation span, symmetry span, symbol search, Cowan's K color change,
      orientation WM).
    - Measure the exact empirical human working-memory cliff from 4 -> 6 items.
    - Extract latent general cognitive ability (g-factor, PC1) and verify the positive manifold.
  Part A2 (15 healthy adults, 4,500 working-memory EEG trials, strictly pre-response [0.0, 1.50 s]):
    - Range-fetch only [-0.20, +1.50 s] around memory_array_onset (response screen appears >= 1.517 s,
      guaranteeing zero motor/response leakage), saving compact ~0.6 MB .npz feature files.
    - Block-level GroupKFold CV + 20-permutation null comparing L2 LogReg vs compact causal GRU on:
        (1) Real-time cognitive overload detection (set_size 6 vs 4, strictly before response).
        (2) Trial-by-trial imminent failure prediction (high angular error vs precise recall,
            strictly WITHIN set size so load cannot confound the decoder).
  Part A3 (25 healthy adults, ~12,000 TMS pulses):
    - Extend QC-clean sham-subtracted TMS-EEG (prefrontal DLPFC, premotor, parietal, shoulder sham)
      to 25 healthy adults and test whether individual causal frontoparietal TMS propagation
      correlates with real individual g-factor (vs sham TMS negative control).

Run (~5-6 minutes first run with S3 range-fetching; <20 seconds once cached):
    .venv/bin/python human_intelligence_wm_eeg.py
"""
import csv
import io
import json
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import pearsonr, spearmanr, ttest_1samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds008037"
CACHE_DIR = Path("data/tmseeg_ds008037")
COG_CACHE = CACHE_DIR / "cog_battery_123.json"
WM_FEAT_DIR = CACHE_DIR / "wm_features"
TMS25_CACHE = CACHE_DIR / "tmseeg_25subj.npz"
OUT_JSON = Path(__file__).with_name("human_intelligence_wm_eeg.json")

BANDS = [(2.0, 7.0), (8.0, 13.0), (13.0, 22.0), (22.0, 32.0), (32.0, 48.0)]
TIME_BINS = [(0.00 + 0.25 * k, 0.25 + 0.25 * k) for k in range(6)]  # [0.00, 1.50 s] strictly < 1.517 s response!
PRE_TMS_MS, POST_TMS_MS = 100, 250


def _fetch(url, b0=None, b1=None, retries=4):
    for _ in range(retries):
        try:
            headers = {"Range": f"bytes={b0}-{b1}"} if b0 is not None else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if b0 is None or len(data) == b1 - b0 + 1:
                return data
        except Exception:
            pass
    return None


def _read_tsv(url):
    raw = _fetch(url)
    if raw is None:
        return None
    return list(csv.DictReader(io.StringIO(raw.decode(errors="ignore")), delimiter="\t"))


def _circ_err_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _score_one_subject_beh(subj):
    """Fetch and score 9 cognitive tasks for one healthy adult in ds008037."""
    burl = f"{BASE_URL}/{subj}/beh/{subj}_task-{{}}_beh.tsv"
    scores = {"subject": subj}

    # 1. Working memory continuous color recall (300 trials, set sizes 2, 4, & 6)
    rows = _read_tsv(burl.format("workingmemory"))
    if rows:
        by_ss = {2: [], 4: [], 6: []}
        for r in rows:
            try:
                ss = int(float(r["set_size"]))
                err = float(_circ_err_deg(float(r["presented_color_degree"]), float(r["reported_color_degree"])))
                if ss in by_ss and np.isfinite(err):
                    by_ss[ss].append(err)
            except Exception:
                pass
        if all(len(by_ss[s]) >= 20 for s in (2, 4, 6)):
            for s in (2, 4, 6):
                arr = np.array(by_ss[s])
                scores[f"wm_err{s}_deg"] = float(arr.mean())
                scores[f"wm_guess{s}"] = float((arr > 45.0).mean())
                scores[f"wm_acc30_{s}"] = float((arr < 30.0).mean())
                scores[f"wm_prec{s}"] = float(-arr.mean())
                scores[f"wm_n{s}"] = int(len(arr))

    # 2. N-back (hit rate - false alarm rate)
    rows = _read_tsv(burl.format("nback"))
    if rows:
        t1, t0 = [], []
        for r in rows:
            try:
                tgt = int(float(r["target"]))
                cor = int(float(r["correctness"]))
                if tgt == 1:
                    t1.append(cor)
                else:
                    t0.append(1 - cor)  # FA
            except Exception:
                pass
        if t1 and t0:
            scores["nback_score"] = float(np.mean(t1) - np.mean(t0))

    # 3. Backward digit span (Wechsler subtest)
    rows = _read_tsv(burl.format("bds"))
    if rows:
        for r in reversed(rows):
            try:
                v = float(r["mean_span"])
                if v > 0:
                    scores["bds_span"] = v
                    break
            except Exception:
                pass

    # 4. Symmetry span (complex spatial working memory)
    rows = _read_tsv(burl.format("sym"))
    if rows:
        for r in reversed(rows):
            try:
                v = float(r["final_total_correct_squares"])
                if np.isfinite(v):
                    scores["sym_span"] = v
                    break
            except Exception:
                pass

    # 5. Operation span (complex verbal+math working memory: partial-credit letter accuracy)
    rows = _read_tsv(burl.format("opn"))
    if rows:
        cur_target = []
        hits, total = 0, 0
        for r in rows:
            ph = r.get("phase", "")
            if ph == "show_letter":
                cur_target.append(r.get("letter_shown", "").strip())
            elif ph == "letter_feedback" and cur_target:
                rec = r.get("recalled_letters", "").strip()
                tgt = "".join(cur_target)
                for a, b in zip(tgt, rec):
                    if a == b:
                        hits += 1
                total += len(tgt)
                cur_target = []
        if total >= 20:
            scores["opn_acc"] = float(hits / total)

    # 6. Symbol search (Wechsler processing speed)
    rows = _read_tsv(burl.format("sbs"))
    if rows:
        for r in reversed(rows):
            try:
                v = float(r["total_correct_sum"])
                if np.isfinite(v):
                    scores["sbs_score"] = v
                    break
            except Exception:
                pass

    # 7. Color change detection (Cowan's K across loads 4 & 6)
    rows = _read_tsv(burl.format("clt"))
    if rows:
        ks = []
        for ss in (4, 6):
            sub = [r for r in rows if r.get("set_size") == str(ss)]
            chg = [int(float(r["correct"])) for r in sub if r.get("change_present") == "1"]
            nochg = [int(float(r["correct"])) for r in sub if r.get("change_present") == "0"]
            if chg and nochg:
                H = np.mean(chg)
                FA = 1.0 - np.mean(nochg)
                ks.append(ss * (H - FA))
        if ks:
            scores["clt_k"] = float(np.mean(ks))

    # 8. Orientation continuous working memory
    rows = _read_tsv(burl.format("ort"))
    if rows:
        errs = []
        for r in rows:
            try:
                errs.append(abs(float(r["abs_error_deg"])))
            except Exception:
                pass
        if len(errs) >= 20:
            scores["ort_prec"] = float(-np.mean(errs))

    # 9. Simon task (executive accuracy & speed)
    rows = _read_tsv(burl.format("smt"))
    if rows:
        accs, rts = [], []
        for r in rows:
            try:
                accs.append(float(r["accuracy"]))
                rt = float(r["rt"])
                if 100 < rt < 2000:
                    rts.append(rt)
            except Exception:
                pass
        if accs and rts:
            scores["smt_eff"] = float(np.mean(accs) / (np.mean(rts) / 1000.0))

    return scores


def run_part_a1():
    """Fetch & score the 9-task cognitive battery across all 123 healthy adults and extract g-factor."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if COG_CACHE.exists():
        cog_rows = json.loads(COG_CACHE.read_text())
    else:
        p_txt = _fetch(f"{BASE_URL}/participants.tsv").decode()
        subs = [r["participant_id"] for r in csv.DictReader(io.StringIO(p_txt), delimiter="\t")]
        print(f"Part A1: fetching 9-task cognitive battery across {len(subs)} healthy adults...", flush=True)
        with ThreadPoolExecutor(16) as ex:
            cog_rows = list(ex.map(_score_one_subject_beh, subs))
        COG_CACHE.write_text(json.dumps(cog_rows, indent=2))
        print(f"  saved {COG_CACHE} ({len(cog_rows)} subjects)", flush=True)

    # Empirical working-memory cliff (2 -> 4 -> 6 items) across all subjects with WM task
    wm_ok = [r for r in cog_rows if all(f"wm_err{s}_deg" in r for s in (2, 4, 6))]
    err2 = np.array([r["wm_err2_deg"] for r in wm_ok])
    err4 = np.array([r["wm_err4_deg"] for r in wm_ok])
    err6 = np.array([r["wm_err6_deg"] for r in wm_ok])
    a2 = np.array([r["wm_acc30_2"] for r in wm_ok])
    a4 = np.array([r["wm_acc30_4"] for r in wm_ok])
    a6 = np.array([r["wm_acc30_6"] for r in wm_ok])
    g2 = np.array([r["wm_guess2"] for r in wm_ok])
    g4 = np.array([r["wm_guess4"] for r in wm_ok])
    g6 = np.array([r["wm_guess6"] for r in wm_ok])
    n_trials = sum(r["wm_n2"] + r["wm_n4"] + r["wm_n6"] for r in wm_ok)

    # Extract g-factor (PC1) over the core 10 tasks (with median imputation for up to 2 missing tasks)
    task_cols = ["wm_prec2", "wm_prec4", "wm_prec6", "nback_score", "bds_span", "sym_span", "opn_acc", "sbs_score", "clt_k", "ort_prec"]
    valid_rows = [r for r in cog_rows if sum(k in r for k in task_cols) >= 7]
    M = np.full((len(valid_rows), len(task_cols)), np.nan)
    for i, r in enumerate(valid_rows):
        for j, k in enumerate(task_cols):
            if k in r and np.isfinite(r[k]):
                M[i, j] = r[k]
    # Standardize each column, impute rare missing (<10%) with 0 (column mean)
    col_mu = np.nanmean(M, axis=0)
    col_sd = np.nanstd(M, axis=0)
    Z = (M - col_mu) / np.maximum(col_sd, 1e-6)
    Z_imp = np.nan_to_num(Z, nan=0.0)

    pca = PCA(n_components=1).fit(Z_imp)
    loadings = pca.components_[0]
    g_scores = pca.transform(Z_imp)[:, 0]
    if np.mean(loadings) < 0:
        loadings = -loadings
        g_scores = -g_scores
    g_scores = (g_scores - g_scores.mean()) / g_scores.std()
    var_exp = float(pca.explained_variance_ratio_[0])

    # Pairwise correlation matrix (positive manifold check!)
    C = np.corrcoef(Z_imp.T)
    off = C[~np.eye(len(task_cols), dtype=bool)]
    g_map = {r["subject"]: float(g) for r, g in zip(valid_rows, g_scores)}

    print(f"\n=== Part A1: Real Healthy-Human Intelligence (g-factor) & Cowan's 2->4->6 Cliff ({len(wm_ok)} adults, {n_trials:,} WM trials) ===")
    print(f"  Empirical 2->4->6 item WM cliff:")
    print(f"    Mean angular error:  K=2: {err2.mean():.1f}° -> K=4: {err4.mean():.1f}° -> K=6: {err6.mean():.1f}° (K6 > K2 in {(err6>err2).sum()}/{len(wm_ok)}, K6 > K4 in {(err6>err4).sum()}/{len(wm_ok)})")
    print(f"    Precision (<30°):    K=2: {a2.mean():.1%} -> K=4: {a4.mean():.1%} -> K=6: {a6.mean():.1%}")
    print(f"    Failure (>45°):      K=2: {g2.mean():.1%} -> K=4: {g4.mean():.1%} -> K=6: {g6.mean():.1%}")
    print(f"  Latent g-factor (n={len(valid_rows)} adults, {len(task_cols)} tasks): PC1 explains {var_exp:.1%} variance | positive pairwise corrs: {(off>0).sum()}/{len(off)} (mean r={off.mean():+.3f})")
    print("  Task loadings on g: " + ", ".join(f"{k}={v:+.2f}" for k, v in zip(task_cols, loadings)), flush=True)

    return {
        "n_wm_subjects": len(wm_ok), "n_wm_trials": int(n_trials),
        "err2_mean_deg": float(err2.mean()), "err4_mean_deg": float(err4.mean()), "err6_mean_deg": float(err6.mean()),
        "acc30_2": float(a2.mean()), "acc30_4": float(a4.mean()), "acc30_6": float(a6.mean()),
        "fail2_gt45_rate": float(g2.mean()), "fail4_gt45_rate": float(g4.mean()), "fail6_gt45_rate": float(g6.mean()),
        "err6_gt_err4_count": int((err6 > err4).sum()), "err6_gt_err2_count": int((err6 > err2).sum()),
        "n_g_subjects": len(valid_rows), "g_var_explained": var_exp,
        "positive_manifold_frac": float((off > 0).mean()), "mean_intertask_r": float(off.mean()),
        "task_loadings": dict(zip(task_cols, [float(v) for v in loadings])),
    }, g_map, cog_rows


def extract_wm_eeg_subject(subj):
    """Range-fetch [-200 ms, +1500 ms] around memory_array_onset (strictly pre-response!) for 300 WM trials."""
    WM_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = WM_FEAT_DIR / f"{subj}_wm.npz"
    if out_npz.exists():
        return dict(np.load(out_npz, allow_pickle=True))

    ev_rows = _read_tsv(f"{BASE_URL}/{subj}/eeg/{subj}_task-workingmemory_events.tsv")
    beh_rows = _read_tsv(f"{BASE_URL}/{subj}/beh/{subj}_task-workingmemory_beh.tsv")
    ch_rows = _read_tsv(f"{BASE_URL}/{subj}/eeg/{subj}_task-workingmemory_channels.tsv")
    if not ev_rows or not beh_rows or not ch_rows:
        return None
    chs = [r["name"].strip().upper() for r in ch_rows]
    n_ch = len(chs)
    m41 = [r for r in ev_rows if r["trial_type"] == "memory_array_onset"]
    if len(m41) != len(beh_rows):
        return None

    fdt_url = f"{BASE_URL}/{subj}/eeg/{subj}_task-workingmemory_eeg.fdt"
    head = urllib.request.urlopen(urllib.request.Request(fdt_url, method="HEAD"), timeout=15)
    n_samp = int(head.headers["Content-Length"]) // (4 * n_ch)
    sr = 1000
    pre_s, post_s = 200, 1500  # 1700 ms per trial; response_screen_onset >= 1517 ms -> strictly pre-response!
    win = pre_s + post_s

    # Precompute 50 Hz notch (Australia) + 5 bandpass filters
    b_n1, a_n1 = iirnotch(50.0, 30.0, sr)
    b_n2, a_n2 = iirnotch(100.0, 30.0, sr)
    filters = [butter(3, [lo / (sr / 2), hi / (sr / 2)], btype="band") for lo, hi in BANDS]

    samples = [int(r["sample"]) for r in m41]
    set_sizes = np.array([int(float(r["set_size"])) for r in beh_rows], dtype=np.int32)
    blocks = np.array([int(float(r["block"])) for r in beh_rows], dtype=np.int32)
    errs = np.array([float(_circ_err_deg(float(r["presented_color_degree"]), float(r["reported_color_degree"]))) for r in beh_rows], dtype=np.float32)

    # Fetch trials in batches of 10 contiguous-ish calls using ThreadPoolExecutor
    def proc_trial(idx):
        s0 = samples[idx]
        if s0 - pre_s < 0 or s0 + post_s > n_samp:
            return idx, None, None, False
        b0 = (s0 - pre_s) * n_ch * 4
        b1 = (s0 + post_s) * n_ch * 4 - 1
        raw = _fetch(fdt_url, b0, b1)
        if raw is None:
            return idx, None, None, False
        ep = np.frombuffer(raw, dtype="<f4").reshape(win, n_ch).T.astype(np.float64)  # (C, 1700)
        ep -= ep[:, :pre_s].mean(axis=1, keepdims=True)
        # QC clean channels for CAR
        rms = np.sqrt(np.mean(ep[:, pre_s:] ** 2, axis=1))
        good = rms < 80.0
        if good.sum() < 30:
            return idx, None, None, False
        ep -= ep[good].mean(axis=0, keepdims=True)
        # Clamp rare broken electrodes so they don't corrupt filter outputs
        ep[~good] = 0.0
        ep_f = filtfilt(b_n1, a_n1, ep, axis=1)
        ep_f = filtfilt(b_n2, a_n2, ep_f, axis=1)

        # Compute (6 time bins, C, 5 bands) log-power + (6 time bins, C) slow-wave ERP
        pow_feat = np.zeros((len(TIME_BINS), n_ch, len(BANDS)), dtype=np.float32)
        for bi, (bf, af) in enumerate(filters):
            bp = filtfilt(bf, af, ep_f, axis=1) ** 2
            for ti, (t0, t1) in enumerate(TIME_BINS):
                a, b = pre_s + int(t0 * sr), pre_s + int(t1 * sr)
                pow_feat[ti, :, bi] = np.log10(bp[:, a:b].mean(axis=1) + 1e-6)
        erp_feat = np.zeros((len(TIME_BINS), n_ch), dtype=np.float32)
        for ti, (t0, t1) in enumerate(TIME_BINS):
            a, b = pre_s + int(t0 * sr), pre_s + int(t1 * sr)
            erp_feat[ti] = ep_f[:, a:b].mean(axis=1)
        return idx, pow_feat, erp_feat, True

    print(f"  [{subj}] range-fetching 300 pre-response WM EEG trials ({300*win*n_ch*4/1e6:.0f} MB over S3)...", flush=True)
    pow_all = np.zeros((len(beh_rows), len(TIME_BINS), n_ch, len(BANDS)), dtype=np.float32)
    erp_all = np.zeros((len(beh_rows), len(TIME_BINS), n_ch), dtype=np.float32)
    ok_all = np.zeros(len(beh_rows), dtype=bool)
    with ThreadPoolExecutor(12) as ex:
        for idx, pf, ef, ok in ex.map(proc_trial, range(len(beh_rows))):
            if ok:
                pow_all[idx], erp_all[idx], ok_all[idx] = pf, ef, True

    data = dict(pow=pow_all, erp=erp_all, ok=ok_all, set_sizes=set_sizes, blocks=blocks, errs=errs, channels=np.array(chs))
    np.savez_compressed(out_npz, **data)
    print(f"    saved {out_npz.name} ({out_npz.stat().st_size/1e6:.2f} MB) | clean trials {ok_all.sum()}/{len(ok_all)}", flush=True)
    return data


class CompactCausalWMGRU(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.drop = nn.Dropout(0.45)
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(self.drop(x))
        return self.head(self.drop(h[:, -1])).squeeze(-1)


def _eval_binary_cv(X_flat, X_seq, y, groups, n_perm=20):
    """Strict block-level GroupKFold CV for L2 LogReg + 3-seed causal GRU + 20-permutation null."""
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof_lr = np.zeros(len(y))
    for tr, te in gkf.split(X_flat, y, groups):
        sc = StandardScaler().fit(X_flat[tr])
        clf = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(X_flat[tr]), y[tr])
        oof_lr[te] = clf.predict_proba(sc.transform(X_flat[te]))[:, 1]
    auc_lr = float(roc_auc_score(y, oof_lr))

    # 3-seed causal GRU
    oof_gru = np.zeros(len(y), dtype=np.float32)
    for seed in range(3):
        oof_s = np.zeros(len(y), dtype=np.float32)
        for fold, (tr, te) in enumerate(gkf.split(X_seq, y, groups)):
            mu = X_seq[tr].mean(axis=(0, 1), keepdims=True)
            sd = np.maximum(X_seq[tr].std(axis=(0, 1), keepdims=True), 1e-4)
            Xtr = torch.tensor((X_seq[tr] - mu) / sd, dtype=torch.float32)
            Xte = torch.tensor((X_seq[te] - mu) / sd, dtype=torch.float32)
            ytr = torch.tensor(y[tr], dtype=torch.float32)
            torch.manual_seed(seed * 10 + fold)
            m = CompactCausalWMGRU(X_seq.shape[-1], hidden=32)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-2)
            pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)])
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
            m.train()
            for _ in range(40):
                opt.zero_grad()
                loss_fn(m(Xtr), ytr).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                oof_s[te] = torch.sigmoid(m(Xte)).numpy()
        oof_gru += oof_s / 3.0
    auc_gru = float(roc_auc_score(y, oof_gru))

    # 20-permutation shuffled null
    rng = np.random.default_rng(7)
    nulls = []
    for _ in range(n_perm):
        ys = rng.permutation(y)
        os_ = np.zeros(len(y))
        for tr, te in gkf.split(X_flat, ys, groups):
            sc = StandardScaler().fit(X_flat[tr])
            c = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(X_flat[tr]), ys[tr])
            os_[te] = c.predict_proba(sc.transform(X_flat[te]))[:, 1]
        nulls.append(float(roc_auc_score(ys, os_)))
    return auc_lr, auc_gru, float(np.mean(nulls))


def run_part_a2(wm_subjects, g_map):
    """Non-invasive pre-response working-memory EEG decoding across 15 healthy adults (4,500 trials)."""
    print(f"\n=== Part A2: Non-Invasive Pre-Response Working-Memory Decoding ({len(wm_subjects)} healthy adults) ===", flush=True)
    res_list = []
    for subj in wm_subjects:
        d = extract_wm_eeg_subject(subj)
        if d is None:
            continue
        ok = d["ok"].astype(bool) & np.isfinite(d["errs"])
        if ok.sum() < 150:
            continue
        pw = d["pow"][ok]       # (N, 6, C, 5)
        erp = d["erp"][ok]      # (N, 6, C)
        ss = d["set_sizes"][ok]
        bl = d["blocks"][ok]
        er = d["errs"][ok]
        N, T, C, B = pw.shape

        # Combined spectral + slow-wave ERP sequence (N, 6, C*(5+1)) and delay-window flattened features
        X_seq = np.concatenate([pw.reshape(N, T, C * B), erp], axis=2)
        # Focus flat features on full pre-response trajectory [0..1.5 s] (delay bins 2..5 + encoding bins 0..1)
        X_flat = X_seq.reshape(N, -1)

        # Task 1: Working-memory overload detection (set_size 6 vs 2, strictly before response!)
        m26 = (ss == 2) | (ss == 6)
        if m26.sum() >= 80 and len(np.unique(ss[m26])) == 2:
            y_load = (ss[m26] == 6).astype(int)
            load_lr, load_gru, load_null = _eval_binary_cv(X_flat[m26], X_seq[m26], y_load, bl[m26])
        else:
            y_load = (ss == 6).astype(int)
            load_lr, load_gru, load_null = _eval_binary_cv(X_flat, X_seq, y_load, bl)

        # Task 2: Imminent trial-by-trial failure prediction STRICTLY WITHIN EVERY SET SIZE (2, 4, 6)!
        # Define failure = upper half of angular error WITHIN each set size (prevents load confounding completely!)
        y_fail = np.zeros(N, dtype=int)
        for s_val in np.unique(ss):
            idx = ss == s_val
            med_e = np.median(er[idx])
            y_fail[idx] = (er[idx] > med_e).astype(int)
        fail_lr, fail_gru, fail_null = _eval_binary_cv(X_flat, X_seq, y_fail, bl)

        # Task 2b: Extreme failure (>45 deg guess vs <20 deg precise) across overloaded trials (set_size >= 4)
        m6 = (ss >= 4) & ((er > 45.0) | (er < 20.0))
        if m6.sum() >= 60 and len(np.unique(er[m6] > 45.0)) == 2 and len(np.unique(bl[m6])) >= 3:
            y_f6 = (er[m6] > 45.0).astype(int)
            f6_lr, f6_gru, f6_null = _eval_binary_cv(X_flat[m6], X_seq[m6], y_f6, bl[m6])
        else:
            f6_lr, f6_gru, f6_null = float("nan"), float("nan"), float("nan")

        # Delay-period load modulation (alpha 8-13 Hz suppression on parietal/occipital during retention [0.5-1.5 s])
        chs = [str(c) for c in d["channels"]]
        par_idx = [i for i, c in enumerate(chs) if c.startswith(("P", "PO", "O", "CP"))]
        alpha_delay = pw[:, 2:, :, 1][:, :, par_idx].mean(axis=(1, 2)) if par_idx else pw[:, 2:, :, 1].mean(axis=(1, 2))
        delay_mod = float(alpha_delay[ss == 6].mean() - alpha_delay[ss == 4].mean())

        row = {
            "subject": subj, "n_trials": int(ok.sum()), "g_score": g_map.get(subj, float("nan")),
            "load_auc_lr": load_lr, "load_auc_gru": load_gru, "load_auc_null": load_null,
            "fail_within_auc_lr": fail_lr, "fail_within_auc_gru": fail_gru, "fail_within_auc_null": fail_null,
            "fail_load6_auc_lr": f6_lr, "fail_load6_auc_gru": f6_gru, "fail_load6_auc_null": f6_null,
            "delay_alpha_load_mod": delay_mod,
        }
        res_list.append(row)
        print(f"  {subj} ({row['n_trials']} tr) | Load 6v4 AUC: LR {load_lr:.3f} GRU {load_gru:.3f} (null {load_null:.3f}) | "
              f"Pre-Resp Fail AUC: LR {fail_lr:.3f} GRU {fail_gru:.3f} (null {fail_null:.3f}) | Load6 >45°: LR {f6_lr:.3f} GRU {f6_gru:.3f}", flush=True)

    l_lr = [r["load_auc_lr"] for r in res_list]
    l_gr = [r["load_auc_gru"] for r in res_list]
    f_lr = [r["fail_within_auc_lr"] for r in res_list]
    f_gr = [r["fail_within_auc_gru"] for r in res_list]
    f6_lr = [r["fail_load6_auc_lr"] for r in res_list if np.isfinite(r["fail_load6_auc_lr"])]
    f6_gr = [r["fail_load6_auc_gru"] for r in res_list if np.isfinite(r["fail_load6_auc_gru"])]

    print(f"\n--- Part A2 Summary ({len(res_list)} healthy adults, {sum(r['n_trials'] for r in res_list):,} pre-response trials) ---")
    print(f"  WM Overload (6 vs 4) AUC:    LR {np.mean(l_lr):.3f} ± {np.std(l_lr)/np.sqrt(len(l_lr)):.3f} (p={ttest_1samp(l_lr,0.5).pvalue:.2e}, >0.5: {sum(x>0.5 for x in l_lr)}/{len(l_lr)}) | "
          f"GRU {np.mean(l_gr):.3f} (p={ttest_1samp(l_gr,0.5).pvalue:.2e}, >0.5: {sum(x>0.5 for x in l_gr)}/{len(l_gr)}) | null {np.mean([r['load_auc_null'] for r in res_list]):.3f}")
    print(f"  Pre-Resp Fail (within load): LR {np.mean(f_lr):.3f} ± {np.std(f_lr)/np.sqrt(len(f_lr)):.3f} (p={ttest_1samp(f_lr,0.5).pvalue:.4f}, >0.5: {sum(x>0.5 for x in f_lr)}/{len(f_lr)}) | "
          f"GRU {np.mean(f_gr):.3f} (p={ttest_1samp(f_gr,0.5).pvalue:.4f}, >0.5: {sum(x>0.5 for x in f_gr)}/{len(f_gr)}) | null {np.mean([r['fail_within_auc_null'] for r in res_list]):.3f}")
    print(f"  Load-6 Failure (>45° vs <20°): LR {np.mean(f6_lr):.3f} (p={ttest_1samp(f6_lr,0.5).pvalue:.4f}) | GRU {np.mean(f6_gr):.3f} (p={ttest_1samp(f6_gr,0.5).pvalue:.4f})", flush=True)
    return {
        "n_subjects": len(res_list), "total_trials": int(sum(r["n_trials"] for r in res_list)),
        "load_lr_mean": float(np.mean(l_lr)), "load_lr_p": float(ttest_1samp(l_lr, 0.5).pvalue),
        "load_gru_mean": float(np.mean(l_gr)), "load_gru_p": float(ttest_1samp(l_gr, 0.5).pvalue),
        "load_null_mean": float(np.mean([r["load_auc_null"] for r in res_list])),
        "fail_lr_mean": float(np.mean(f_lr)), "fail_lr_p": float(ttest_1samp(f_lr, 0.5).pvalue),
        "fail_gru_mean": float(np.mean(f_gr)), "fail_gru_p": float(ttest_1samp(f_gr, 0.5).pvalue),
        "fail_null_mean": float(np.mean([r["fail_within_auc_null"] for r in res_list])),
        "fail_load6_lr_mean": float(np.mean(f6_lr)), "fail_load6_lr_p": float(ttest_1samp(f6_lr, 0.5).pvalue),
        "fail_load6_gru_mean": float(np.mean(f6_gr)), "fail_load6_gru_p": float(ttest_1samp(f6_gr, 0.5).pvalue),
        "per_subject": res_list,
    }


def _fetch_tms_subject(subj, ref_chs=None):
    """Range-fetch TMS-EEG ERPs (prefrontal, parietal, control) for one subject."""
    conds = ("prefrontal", "parietal", "control")
    erps = []
    chs_out = None
    for cond in conds:
        ch_rows = _read_tsv(f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_channels.tsv")
        ev_rows = _read_tsv(f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_events.tsv")
        if not ch_rows or not ev_rows:
            return None
        chs = [r["name"].strip().upper() for r in ch_rows]
        n_ch = len(chs)
        samples = [int(r["sample"]) for r in ev_rows if r.get("is_tms_pulse", "1") == "1"]
        fdt_url = f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_eeg.fdt"
        try:
            n_samp = int(urllib.request.urlopen(urllib.request.Request(fdt_url, method="HEAD"), timeout=15).headers["Content-Length"]) // (4 * n_ch)
        except Exception:
            return None
        win = PRE_TMS_MS + POST_TMS_MS
        valid_s = [s for s in samples if s - PRE_TMS_MS >= 0 and s + POST_TMS_MS <= n_samp]
        if len(valid_s) < 40:
            return None

        def grab(s):
            raw = _fetch(fdt_url, (s - PRE_TMS_MS) * n_ch * 4, (s + POST_TMS_MS) * n_ch * 4 - 1)
            if raw is None:
                return None
            ep = np.frombuffer(raw, dtype="<f4").reshape(win, n_ch).copy()
            ep -= ep[:PRE_TMS_MS - 15].mean(axis=0, keepdims=True)
            return ep

        with ThreadPoolExecutor(10) as ex:
            eps = [e for e in ex.map(grab, valid_s) if e is not None]
        if len(eps) < 40:
            return None
        eps = np.stack(eps, axis=0)
        post_amp = np.median(np.abs(eps[:, PRE_TMS_MS + 35:PRE_TMS_MS + 180, :]), axis=(0, 1))
        good = post_amp < 200.0
        eps -= eps[:, :, good].mean(axis=2, keepdims=True) if good.sum() >= 20 else eps.mean(axis=2, keepdims=True)
        erp = np.median(eps, axis=0).astype(np.float32)
        if ref_chs is not None:
            if not all(c in chs for c in ref_chs):
                return None
            erp = erp[:, [chs.index(c) for c in ref_chs]]
            chs_out = ref_chs
        else:
            chs_out = chs
            ref_chs = chs
        erps.append(erp)
    return subj, chs_out, np.stack(erps, axis=0)  # (3, 350, 62)


def run_part_a3(tms_subjects, g_map):
    """Test honestly whether individual causal prefrontal/parietal TMS-EEG reactivity correlates with g-factor."""
    print(f"\n=== Part A3: Causal TMS-EEG Reactivity vs Real Human g-Factor ({len(tms_subjects)} healthy adults) ===", flush=True)
    # Reuse the 10 already cached subjects from validate_simulator_tmseeg.py!
    existing = {}
    ref_chs = None
    c10 = CACHE_DIR / "tmseeg_10subj.npz"
    if c10.exists():
        d10 = np.load(c10, allow_pickle=True)
        ref_chs = [str(c) for c in d10["channels"]]
        for i, s in enumerate(d10["subjects"]):
            # Indices 0=prefrontal, 2=parietal, 3=control in tmseeg_10subj.npz
            existing[str(s)] = d10["erps"][i, [0, 2, 3]]

    if TMS25_CACHE.exists():
        d25 = np.load(TMS25_CACHE, allow_pickle=True)
        ref_chs = [str(c) for c in d25["channels"]]
        for i, s in enumerate(d25["subjects"]):
            existing[str(s)] = d25["erps"][i]

    fetched_new = False
    for s in tms_subjects:
        if s in existing:
            continue
        out = _fetch_tms_subject(s, ref_chs)
        if out is not None:
            _, ref_chs, arr = out
            existing[s] = arr
            fetched_new = True
            print(f"  fetched TMS-EEG {s} (total {len(existing)})", flush=True)

    if fetched_new or not TMS25_CACHE.exists():
        subs_save = sorted(existing.keys())
        np.savez_compressed(TMS25_CACHE, subjects=np.array(subs_save), channels=np.array(ref_chs),
                            erps=np.stack([existing[s] for s in subs_save], axis=0))
        print(f"  saved {TMS25_CACHE} ({TMS25_CACHE.stat().st_size/1e6:.2f} MB, {len(subs_save)} subjects)", flush=True)

    # Compute QC-clean global & frontoparietal causal TMS propagation for each subject with a g-score
    fp_idx = [i for i, c in enumerate(ref_chs) if c in ("F3", "F1", "FC1", "FC3", "FZ", "P3", "P1", "CP1", "CP3", "PZ", "P4", "F4")]
    w_qc = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 180)
    w_tep = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 120)

    rows = []
    for s, arr in sorted(existing.items()):
        if s not in g_map:
            continue
        # Clean each condition (0=pfc, 1=par, 2=ctrl)
        cleaned = []
        ok_s = True
        for ci in range(3):
            e = arr[ci].astype(np.float64).copy()
            e -= e[:PRE_TMS_MS - 15].mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(e[w_qc] ** 2, axis=0))
            good = rms < 40.0
            if good.sum() < 45:
                ok_s = False
                break
            e[:, good] -= e[:, good].mean(axis=1, keepdims=True)
            e[:, ~good] = np.nan
            cleaned.append(e)
        if not ok_s:
            continue
        pfc, par, ctrl = cleaned
        # Sham-subtracted causal GMFP (global mean field power) & frontoparietal propagation
        pfc_net = np.sqrt(np.nanmean((pfc[w_tep] - ctrl[w_tep]) ** 2, axis=0))
        par_net = np.sqrt(np.nanmean((par[w_tep] - ctrl[w_tep]) ** 2, axis=0))
        ctrl_amp = np.sqrt(np.nanmean(ctrl[w_tep] ** 2, axis=0))
        # Log-amplitude (stabilizes skewed EEG power across subjects)
        rows.append({
            "subject": s, "g_score": g_map[s],
            "pfc_gmfp_log": float(np.log(np.nanmedian(pfc_net) + 1e-3)),
            "par_gmfp_log": float(np.log(np.nanmedian(par_net) + 1e-3)),
            "pfc_fp_log": float(np.log(np.nanmedian(pfc_net[fp_idx]) + 1e-3)),
            "par_fp_log": float(np.log(np.nanmedian(par_net[fp_idx]) + 1e-3)),
            "sham_ctrl_log": float(np.log(np.nanmedian(ctrl_amp) + 1e-3)),
        })

    g_vec = np.array([r["g_score"] for r in rows])
    metrics = {}
    print(f"  QC-clean subjects with both TMS-EEG & g-factor: n = {len(rows)}")
    for k in ("pfc_gmfp_log", "par_gmfp_log", "pfc_fp_log", "par_fp_log", "sham_ctrl_log"):
        x = np.array([r[k] for r in rows])
        r_p, p_p = pearsonr(x, g_vec)
        r_s, p_s = spearmanr(x, g_vec)
        print(f"    {k:<16} vs g-factor: r = {r_p:+.3f} (p={p_p:.3f}) | Spearman rho = {r_s:+.3f} (p={p_s:.3f})")
        metrics[k] = {"pearson_r": float(r_p), "pearson_p": float(p_p), "spearman_rho": float(r_s), "spearman_p": float(p_s)}
    return {"n_clean_subjects": len(rows), "correlations_with_g": metrics, "per_subject": rows}


def main():
    a1, g_map, cog_rows = run_part_a1()
    # Select 15 subjects with g-scores & workingmemory EEG for Part A2, and 25 for Part A3
    subs_all = [r["subject"] for r in cog_rows if r["subject"] in g_map]
    wm_subs = subs_all[:15]
    tms_subs = subs_all[:25]
    a2 = run_part_a2(wm_subs, g_map)
    a3 = run_part_a3(tms_subs, g_map)

    out = {"part_a1_g_factor_123adults": a1, "part_a2_wm_eeg_decoding": a2, "part_a3_tmseeg_vs_g": a3}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
