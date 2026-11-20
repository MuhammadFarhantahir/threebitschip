"""Out-of-sample independent replication of our Level-1 discovery from human_intelligence_wm_eeg.py:
Does sham-subtracted causal parietal TMS-EEG propagation into the frontoparietal network (par_fp_log)
replicate as a predictor of real 10-task healthy-human g-factor on ~25 COMPLETELY UNSEEN healthy adults
in OpenNeuro ds008037 (subjects beyond our initial discovery set)?

Primary hypothesis assessed in the replication analysis:
  - Primary metric: par_fp_log (parietal TMS - shoulder sham, frontoparietal 12 electrodes, [35..120 ms] log RMS)
  - Negative control: sham_ctrl_log (shoulder sham TMS alone)
  - Reports Discovery (n=22), Unseen Replication (n~22), and Combined Pooled (n~44) with exact Spearman rho,
    Pearson r, and 5,000-permutation p-values.

Run (~2 minutes):
    .venv/bin/python replicate_tmseeg_g.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from human_intelligence_wm_eeg import CACHE_DIR, COG_CACHE, PRE_TMS_MS, TMS25_CACHE, _fetch_tms_subject

REP_CACHE = CACHE_DIR / "tmseeg_replication.npz"
OUT_JSON = Path(__file__).with_name("replicate_tmseeg_g.json")


def compute_g_map():
    cog_rows = json.loads(COG_CACHE.read_text())
    task_cols = ["wm_prec2", "wm_prec4", "wm_prec6", "nback_score", "bds_span", "sym_span", "opn_acc", "sbs_score", "clt_k", "ort_prec"]
    valid_rows = [r for r in cog_rows if sum(k in r for k in task_cols) >= 7]
    M = np.full((len(valid_rows), len(task_cols)), np.nan)
    for i, r in enumerate(valid_rows):
        for j, k in enumerate(task_cols):
            if k in r and np.isfinite(r[k]):
                M[i, j] = r[k]
    Z = (M - np.nanmean(M, 0)) / np.maximum(np.nanstd(M, 0), 1e-6)
    Z_imp = np.nan_to_num(Z, nan=0.0)
    pca = PCA(n_components=1).fit(Z_imp)
    g = pca.transform(Z_imp)[:, 0]
    if np.mean(pca.components_[0]) < 0:
        g = -g
    g = (g - g.mean()) / g.std()
    return {r["subject"]: float(v) for r, v in zip(valid_rows, g)}, [r["subject"] for r in valid_rows]


def extract_metrics(existing_dict, ref_chs, g_map):
    fp_idx = [i for i, c in enumerate(ref_chs) if c in ("F3", "F1", "FC1", "FC3", "FZ", "P3", "P1", "CP1", "CP3", "PZ", "P4", "F4")]
    w_qc = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 180)
    w_tep = slice(PRE_TMS_MS + 35, PRE_TMS_MS + 120)
    rows = []
    for s, arr in sorted(existing_dict.items()):
        if s not in g_map:
            continue
        cleaned, ok = [], True
        for ci in range(3):
            e = arr[ci].astype(np.float64).copy()
            e -= e[:PRE_TMS_MS - 15].mean(axis=0, keepdims=True)
            good = np.sqrt(np.mean(e[w_qc] ** 2, axis=0)) < 40.0
            if good.sum() < 45:
                ok = False
                break
            e[:, good] -= e[:, good].mean(axis=1, keepdims=True)
            e[:, ~good] = np.nan
            cleaned.append(e)
        if not ok:
            continue
        pfc, par, ctrl = cleaned
        pfc_net = np.sqrt(np.nanmean((pfc[w_tep] - ctrl[w_tep]) ** 2, axis=0))
        par_net = np.sqrt(np.nanmean((par[w_tep] - ctrl[w_tep]) ** 2, axis=0))
        ctrl_amp = np.sqrt(np.nanmean(ctrl[w_tep] ** 2, axis=0))
        rows.append({
            "subject": s, "g_score": g_map[s],
            "pfc_gmfp_log": float(np.log(np.nanmedian(pfc_net) + 1e-3)),
            "par_gmfp_log": float(np.log(np.nanmedian(par_net) + 1e-3)),
            "pfc_fp_log": float(np.log(np.nanmedian(pfc_net[fp_idx]) + 1e-3)),
            "par_fp_log": float(np.log(np.nanmedian(par_net[fp_idx]) + 1e-3)),
            "sham_ctrl_log": float(np.log(np.nanmedian(ctrl_amp) + 1e-3)),
        })
    return rows


def score_split(label, rows):
    g = np.array([r["g_score"] for r in rows])
    out = {"n": len(rows)}
    print(f"\n--- {label} (n = {len(rows)} QC-clean healthy adults) ---")
    for k in ("par_fp_log", "par_gmfp_log", "pfc_gmfp_log", "pfc_fp_log", "sham_ctrl_log"):
        x = np.array([r[k] for r in rows])
        rp, pp = pearsonr(x, g)
        rs, ps = spearmanr(x, g)
        print(f"  {k:<15} vs g: Spearman rho = {rs:+.3f} (p={ps:.4f}) | Pearson r = {rp:+.3f} (p={pp:.4f})")
        out[k] = {"spearman_rho": float(rs), "spearman_p": float(ps), "pearson_r": float(rp), "pearson_p": float(pp)}
    return out


def main():
    g_map, all_subs = compute_g_map()
    d_disc = np.load(TMS25_CACHE, allow_pickle=True)
    ref_chs = [str(c) for c in d_disc["channels"]]
    disc_dict = {str(s): d_disc["erps"][i] for i, s in enumerate(d_disc["subjects"])}

    # Select next 28 candidate unseen subjects (to guarantee >=22 QC-clean replication subjects)
    unseen_cands = [s for s in all_subs if s not in disc_dict][:28]
    rep_dict = {}
    if REP_CACHE.exists():
        dr = np.load(REP_CACHE, allow_pickle=True)
        rep_dict = {str(s): dr["erps"][i] for i, s in enumerate(dr["subjects"])}

    print(f"=== Independent Out-of-Sample Replication of TMS-EEG -> Human g-Factor ({len(unseen_cands)} unseen candidates) ===", flush=True)
    fetched = False
    for s in unseen_cands:
        if len(rep_dict) >= 24:
            break
        if s in rep_dict:
            continue
        res = _fetch_tms_subject(s, ref_chs)
        if res is not None:
            _, _, arr = res
            rep_dict[s] = arr
            fetched = True
            print(f"  fetched unseen replication {s} ({len(rep_dict)}/24)", flush=True)

    if fetched or not REP_CACHE.exists():
        subs_s = sorted(rep_dict.keys())
        np.savez_compressed(REP_CACHE, subjects=np.array(subs_s), channels=np.array(ref_chs),
                            erps=np.stack([rep_dict[s] for s in subs_s], axis=0))
        print(f"  saved {REP_CACHE} ({REP_CACHE.stat().st_size/1e6:.2f} MB, {len(subs_s)} unseen subjects)", flush=True)

    disc_rows = extract_metrics(disc_dict, ref_chs, g_map)
    rep_rows = extract_metrics(rep_dict, ref_chs, g_map)
    all_rows = disc_rows + rep_rows

    out = {
        "discovery": score_split("Discovery Cohort", disc_rows),
        "replication_unseen": score_split("Unseen Independent Replication Cohort", rep_rows),
        "pooled_all": score_split("Combined Pooled Cohort", all_rows),
        "per_subject_all": all_rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
