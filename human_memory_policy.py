"""Scale real human closed-loop stimulation & memory decoding to 15 patients (4,500 words, 1,154 real stimulations)
from OpenNeuro UPenn/DARPA RAM (ds005557 FR3 + ds005558 catFR3, CC0), and evaluate a leave-one-patient-out
offline causal stimulation policy.

Reuses cached 3-MB .npz feature files from data/ram_closedloop/features/ and range-fetches only the word-list
byte ranges for the 11 additional crystal-clean sessions (all 15 patients share ~1.43-1.60 s STIM_ON latency).

Key scientific tests across 15 patients (10 Lateral Temporal Cortex [LTC], 2 Hippocampus [CA1/DG], 2 Frontal, 1 Insula):
  1. Per-patient memory-state decoding on held-out non-stim lists: L2 Logistic Regression vs causal Temporal GRU
     vs 20-permutation shuffled null, plus agreement with the hospital's online trigger.
  2. Unbiased causal estimation via out-of-fold pre-state matching:
       - Stimulated low-state words vs matched low-state control-list words (post-stim neural shift Δz & recall Δ).
       - Unstimulated high-state words on stim lists vs matched high-state control-list words (matching sanity control).
  3. Anatomical site specificity: LTC vs Hippocampus vs Frontal/Insula (both neural shift Δz and matched recall Δ).
  4. Leave-one-patient-out offline causal policy evaluation: compare No-Stim, Hospital Trigger, Permuted-Site Control,
     and Learned Causal Policy (stimulate only when predicted site+state benefit > 0).

Run (~4-5 minutes first run for S3 range-fetching 11 new sessions; <30 seconds once cached):
    .venv/bin/python human_memory_policy.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_1samp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from human_memory_closedloop import BASE, extract_session, fit_gru_cv


def load_schaefer_centroids():
    labels, coords = [], []
    with open("data/hcp_connectome/schaefer100_centroids_mni.csv") as f:
        for r in csv.DictReader(f):
            labels.append(r["ROI Name"].strip())
            coords.append([float(r["R"]), float(r["A"]), float(r["S"])])
    return labels, np.array(coords, dtype=float)

OUT = Path(__file__).with_name("human_memory_policy.json")

# 15 unique patients with clean >=1.42s post-WORD STIM_ON latency, >=50 stims, >=100 ctrl words, >=12% recall
COHORT = [
    # 4 already-cached patients
    ("ds005558", "R1274T", "0", "catFR3"),
    ("ds005557", "R1202M", "1", "FR3"),
    ("ds005557", "R1161E", "1", "FR3"),
    ("ds005558", "R1264P", "1", "catFR3"),
    # 11 new patients spanning LTC, Hippocampus (DG), Frontal, Insula
    ("ds005557", "R1166D", "0", "FR3"),
    ("ds005557", "R1170J", "1", "FR3"),
    ("ds005557", "R1154D", "0", "FR3"),
    ("ds005557", "R1200T", "0", "FR3"),
    ("ds005557", "R1217T", "0", "FR3"),
    ("ds005558", "R1247P", "0", "catFR3"),
    ("ds005557", "R1204T", "0", "FR3"),
    ("ds005558", "R1286J", "0", "catFR3"),
    ("ds005557", "R1195E", "0", "FR3"),
    ("ds005557", "R1201P", "0", "FR3"),
    ("ds005557", "R1145J", "0", "FR3"),
]

LTC_REGIONS = {"middletemporal", "inferiortemporal", "superiortemporal", "bankssts"}


def get_anatomy(ds, subj, ses, task, anode):
    """Read exact anatomical region and MNI coordinates of the stimulation anode from BIDS electrodes.tsv."""
    ieeg_dir = BASE / ds / f"sub-{subj}" / f"ses-{ses}" / "ieeg"
    el_files = list(ieeg_dir.glob("*_electrodes.tsv"))
    info = {"region": "unknown", "site_group": "Other", "xyz": [np.nan, np.nan, np.nan], "hemi": "?"}
    if not el_files:
        return info
    with open(el_files[0]) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("name") == anode:
                ind = (r.get("ind.region") or "").strip()
                das = (r.get("das.region") or "").strip()
                wb = (r.get("wb.region") or "").strip()
                reg = ind if ind not in ("", "n/a") else (das if das not in ("", "n/a") else wb)
                try:
                    xyz = [float(r["x"]), float(r["y"]), float(r["z"])]
                except Exception:
                    xyz = [np.nan, np.nan, np.nan]
                if reg in LTC_REGIONS:
                    grp = "LTC"
                elif reg in ("CA1", "DG", "CA3", "sub", "ERC") or "hippocamp" in wb.lower():
                    grp = "Hippocampus"
                elif "frontal" in reg.lower():
                    grp = "Frontal"
                elif "insula" in reg.lower():
                    grp = "Insula"
                else:
                    grp = "Other"
                info = {"region": reg, "site_group": grp, "xyz": xyz, "hemi": r.get("hemisphere", "?")}
                break
    return info


def analyze_patient(ds, subj, ses, task, schaefer_labels, schaefer_coords):
    d = extract_session(ds, subj, ses, task)
    pre = d["pre"]
    post = d["post"]
    recalled = d["recalled"].astype(int)
    lists = d["lists"].astype(int)
    stim_list = d["stim_list"].astype(int)
    triggered = d["triggered"].astype(int)
    pre_clean = d["pre_clean"].astype(bool)
    post_clean = d["post_clean"].astype(bool)
    anode = str(d["anode"])

    anat = get_anatomy(ds, subj, ses, task, anode)
    xyz = np.array(anat["xyz"], dtype=float)
    if np.all(np.isfinite(xyz)):
        dists = np.linalg.norm(schaefer_coords - xyz[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_parcel = schaefer_labels[nearest_idx]
        nearest_dist_mm = float(dists[nearest_idx])
    else:
        nearest_parcel, nearest_dist_mm = "?", float("nan")

    N, T, C, B = pre.shape
    X_mean = pre.mean(axis=1).reshape(N, C * B)
    X_post = post.reshape(N, C * B)
    X_seq = pre.reshape(N, T, C * B)

    # 1. Held-out list-level memory decoding on non-stim control lists
    ctrl_mask = (stim_list == 0) & pre_clean
    Xc, Xc_seq, yc, gc = X_mean[ctrl_mask], X_seq[ctrl_mask], recalled[ctrl_mask], lists[ctrl_mask]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(gc))))

    oof_lr = np.zeros(len(yc))
    oof_z = np.zeros(len(yc))
    for tr, te in gkf.split(Xc, yc, gc):
        sc = StandardScaler().fit(Xc[tr])
        clf = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc.transform(Xc[tr]), yc[tr])
        oof_lr[te] = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
        oof_z[te] = clf.decision_function(sc.transform(Xc[te]))

    # 20-permutation null (list-stratified / full permutation)
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

    # 2. Full control-list decoder -> standardize z-scores using control-list OOF mean & std for cross-patient comparability!
    sc_all = StandardScaler().fit(Xc)
    clf_all = LogisticRegression(C=1e-3, max_iter=500, class_weight="balanced").fit(sc_all.transform(Xc), yc)
    z_pre_raw = clf_all.decision_function(sc_all.transform(X_mean))
    z_post_raw = clf_all.decision_function(sc_all.transform(X_post))
    z_pre_unbiased_raw = z_pre_raw.copy()
    z_pre_unbiased_raw[ctrl_mask] = oof_z

    # Normalize z-scores into units of control-list OOF SD so Δz is dimensionless across patients
    z_mu, z_sd = float(oof_z.mean()), max(float(oof_z.std()), 1e-6)
    z_pre = (z_pre_raw - z_mu) / z_sd
    z_post = (z_post_raw - z_mu) / z_sd
    z_pre_u = (z_pre_unbiased_raw - z_mu) / z_sd

    sl_clean = (stim_list == 1) & pre_clean
    trig_auc = float(roc_auc_score(triggered[sl_clean], -z_pre_u[sl_clean])) if len(np.unique(triggered[sl_clean])) == 2 else float("nan")

    # 3. Matched low-state (stimulated vs ctrl) AND matched high-state placebo control (unstimulated on stim list vs ctrl)
    stim_low = sl_clean & (triggered == 1)
    stim_high = sl_clean & (triggered == 0)
    n_trig = int(stim_low.sum())
    frac = n_trig / max(int(sl_clean.sum()), 1)
    thresh = float(np.quantile(z_pre_u[ctrl_mask], frac))
    ctrl_low = ctrl_mask & (z_pre_u <= thresh)
    ctrl_high = ctrl_mask & (z_pre_u > thresh)

    # Neural state shift (post - pre) in SD units on clean post windows
    s_low_post = stim_low & post_clean
    c_low_post = ctrl_low & post_clean
    s_high_post = stim_high & post_clean
    c_high_post = ctrl_high & post_clean

    dz_stim_low = float((z_post[s_low_post] - z_pre[s_low_post]).mean()) if s_low_post.sum() else float("nan")
    dz_ctrl_low = float((z_post[c_low_post] - z_pre[c_low_post]).mean()) if c_low_post.sum() else float("nan")
    dz_stim_high = float((z_post[s_high_post] - z_pre[s_high_post]).mean()) if s_high_post.sum() else float("nan")
    dz_ctrl_high = float((z_post[c_high_post] - z_pre[c_high_post]).mean()) if c_high_post.sum() else float("nan")

    rec_stim_low = float(recalled[stim_low].mean()) if stim_low.sum() else float("nan")
    rec_ctrl_low = float(recalled[ctrl_low].mean()) if ctrl_low.sum() else float("nan")
    rec_stim_high = float(recalled[stim_high].mean()) if stim_high.sum() else float("nan")
    rec_ctrl_high = float(recalled[ctrl_high].mean()) if ctrl_high.sum() else float("nan")

    # Split triggered words by our decoder's median pre-stim state (very-low vs moderate-state) for state-dependence test
    med_trig = float(np.median(z_pre_u[stim_low])) if stim_low.sum() else 0.0
    very_low_s = stim_low & (z_pre_u <= med_trig)
    mod_low_s = stim_low & (z_pre_u > med_trig)
    very_low_c = ctrl_low & (z_pre_u <= med_trig)
    mod_low_c = ctrl_low & (z_pre_u > med_trig)
    diff_very_low_pp = (float(recalled[very_low_s].mean()) - float(recalled[very_low_c].mean())) * 100 if (very_low_s.sum() >= 5 and very_low_c.sum() >= 5) else float("nan")
    diff_mod_low_pp = (float(recalled[mod_low_s].mean()) - float(recalled[mod_low_c].mean())) * 100 if (mod_low_s.sum() >= 5 and mod_low_c.sum() >= 5) else float("nan")

    res = {
        "dataset": ds, "subject": subj, "session": ses, "task": task,
        "anode": anode, "region": anat["region"], "site_group": anat["site_group"],
        "hemi": anat["hemi"], "xyz": anat["xyz"],
        "nearest_schaefer": nearest_parcel, "nearest_schaefer_mm": nearest_dist_mm,
        "n_ch": int(C), "n_words": int(N), "n_ctrl_words": int(ctrl_mask.sum()),
        "n_stim_list_words": int(sl_clean.sum()), "n_stim_words": n_trig, "stim_frac": float(frac),
        "auc_logreg": auc_lr, "auc_gru": auc_gru, "auc_shuffled_ctrl": auc_shuf, "p_logreg": p_lr,
        "agreement_with_online_trigger_auc": trig_auc,
        "dz_stim_low": dz_stim_low, "dz_ctrl_low": dz_ctrl_low, "dz_diff_low": dz_stim_low - dz_ctrl_low,
        "dz_stim_high_placebo": dz_stim_high, "dz_ctrl_high_placebo": dz_ctrl_high,
        "dz_diff_high_placebo": dz_stim_high - dz_ctrl_high,
        "recall_ctrl_all": float(recalled[ctrl_mask].mean()),
        "recall_stim_all": float(recalled[sl_clean].mean()),
        "recall_stim_low": rec_stim_low, "recall_ctrl_low": rec_ctrl_low,
        "recall_diff_low_pp": (rec_stim_low - rec_ctrl_low) * 100,
        "recall_stim_high_placebo": rec_stim_high, "recall_ctrl_high_placebo": rec_ctrl_high,
        "recall_diff_high_placebo_pp": (rec_stim_high - rec_ctrl_high) * 100,
        "recall_diff_very_low_pp": diff_very_low_pp,
        "recall_diff_mod_low_pp": diff_mod_low_pp,
    }
    print(f"  {subj:7s} [{anat['site_group']:11s} {anat['region']:19s}] | AUC LR {auc_lr:.3f} GRU {auc_gru:.3f} (null {auc_shuf:.3f}) | "
          f"trig {trig_auc:.3f} | Δz {res['dz_diff_low']:+.3f} (placebo {res['dz_diff_high_placebo']:+.3f}) | "
          f"low-rec {rec_stim_low:.1%} vs {rec_ctrl_low:.1%} ({res['recall_diff_low_pp']:+.1f} pp)", flush=True)
    return res


def evaluate_lopo_policy(results):
    """Leave-one-patient-out offline causal policy evaluation: predict patient-level stimulation benefit
    from anatomical target class + baseline features, and decide whether to stimulate low-state words or withhold."""
    n = len(results)
    is_ltc = np.array([1.0 if r["site_group"] == "LTC" else 0.0 for r in results])
    is_hipp = np.array([1.0 if r["site_group"] == "Hippocampus" else 0.0 for r in results])
    is_front = np.array([1.0 if r["site_group"] == "Frontal" else 0.0 for r in results])
    base_rec = np.array([r["recall_ctrl_all"] for r in results])
    X = np.column_stack([is_ltc, is_hipp, is_front])
    y_eff = np.array([r["recall_diff_low_pp"] for r in results])  # counterfactual low-state gain in pp

    # 1. Learned LOPO site policy (Ridge on site-group indicators only -> strictly out-of-sample decision!)
    lopo_pred = np.zeros(n)
    lopo_act = np.zeros(n, dtype=bool)
    for i in range(n):
        tr = np.arange(n) != i
        reg = Ridge(alpha=1.0).fit(X[tr], y_eff[tr])
        lopo_pred[i] = float(reg.predict(X[i:i + 1])[0])
        lopo_act[i] = lopo_pred[i] > 0.0

    # Policy expected outcome on low-state candidate words (pp vs no-stim baseline):
    #   if action == 1 (stimulate): realize y_eff[i]; if action == 0 (withhold): realize 0.0
    hosp_gain = y_eff
    lopo_gain = np.where(lopo_act, y_eff, 0.0)

    # 2. Permuted-site negative control (200 permutations of site labels across patients)
    rng = np.random.default_rng(7)
    perm_means = []
    for _ in range(200):
        perm = rng.permutation(n)
        Xp = X[perm]
        pa = np.zeros(n, dtype=bool)
        for i in range(n):
            tr = np.arange(n) != i
            pa[i] = float(Ridge(alpha=1.0).fit(Xp[tr], y_eff[tr]).predict(Xp[i:i + 1])[0]) > 0.0
        perm_means.append(float(np.where(pa, y_eff, 0.0).mean()))

    # Harm-avoidance check: among sessions with strongly negative observed outcome (< -3 pp), how many did LOPO withhold?
    harmful = y_eff < -3.0
    harm_avoided = int((harmful & (~lopo_act)).sum())
    n_harmful = int(harmful.sum())

    # Correlation between predicted LOPO benefit and actual held-out patient benefit
    lopo_r = float(np.corrcoef(lopo_pred, y_eff)[0, 1]) if np.std(lopo_pred) > 1e-9 else 0.0
    lopo_rho = float(spearmanr(lopo_pred, y_eff).statistic)

    return {
        "n_patients": n,
        "hospital_trigger_mean_pp": float(hosp_gain.mean()),
        "lopo_policy_mean_pp": float(lopo_gain.mean()),
        "lopo_vs_hospital_gain_pp": float((lopo_gain - hosp_gain).mean()),
        "permuted_site_ctrl_mean_pp": float(np.mean(perm_means)),
        "lopo_beats_perm_p": float((sum(p >= lopo_gain.mean() for p in perm_means) + 1) / (len(perm_means) + 1)),
        "lopo_stimulated_count": int(lopo_act.sum()),
        "lopo_withheld_count": int((~lopo_act).sum()),
        "harmful_sessions_lt_minus3pp": n_harmful,
        "harmful_sessions_avoided": harm_avoided,
        "lopo_pred_vs_actual_r": lopo_r,
        "lopo_pred_vs_actual_rho": lopo_rho,
        "per_patient_decisions": [
            {
                "subject": r["subject"], "site_group": r["site_group"], "region": r["region"],
                "lopo_pred_pp": float(lopo_pred[i]), "stimulate": bool(lopo_act[i]),
                "actual_low_diff_pp": float(y_eff[i]), "policy_realized_pp": float(lopo_gain[i]),
            }
            for i, r in enumerate(results)
        ],
    }


def main():
    schaefer_labels, schaefer_coords = load_schaefer_centroids()
    print(f"=== Step 2: 15-Patient Real Human Closed-Loop Memory Cohort & Causal Policy ({len(COHORT)} patients) ===", flush=True)
    results = [analyze_patient(*s, schaefer_labels, schaefer_coords) for s in COHORT]

    total_words = sum(r["n_words"] for r in results)
    total_stims = sum(r["n_stim_words"] for r in results)
    print(f"\n--- Cohort Summary ({len(results)} patients, {total_words:,} words, {total_stims:,} real closed-loop stims) ---")
    aucs_lr = [r["auc_logreg"] for r in results]
    aucs_gru = [r["auc_gru"] for r in results]
    aucs_null = [r["auc_shuffled_ctrl"] for r in results]
    trigs = [r["agreement_with_online_trigger_auc"] for r in results]
    t_lr = ttest_1samp(aucs_lr, 0.5)
    t_gru = ttest_1samp(aucs_gru, 0.5)
    print(f"  Held-out list decoding AUC: LogReg {np.mean(aucs_lr):.3f} ± {np.std(aucs_lr)/np.sqrt(len(results)):.3f} (p={t_lr.pvalue:.4f}, >0.5 in {sum(a>0.5 for a in aucs_lr)}/{len(results)})")
    print(f"                              GRU    {np.mean(aucs_gru):.3f} ± {np.std(aucs_gru)/np.sqrt(len(results)):.3f} (p={t_gru.pvalue:.4f}, >0.5 in {sum(a>0.5 for a in aucs_gru)}/{len(results)})")
    print(f"                              Null   {np.mean(aucs_null):.3f} | Online-trigger agreement AUC: {np.nanmean(trigs):.3f}")

    # Sanity check: high-state placebo control (unstimulated words on stim lists vs matched high-state ctrl words)
    plac_rec = [r["recall_diff_high_placebo_pp"] for r in results]
    plac_dz = [r["dz_diff_high_placebo"] for r in results]
    print(f"  Unstimulated high-state matching control: recall Δ = {np.nanmean(plac_rec):+.2f} pp, neural Δz = {np.nanmean(plac_dz):+.3f} SD (should be ~0)")

    # Site-group breakdown
    print("\n--- Anatomical Site Specificity (Matched Low-State Stimulated vs Control Words) ---")
    by_group = {}
    for grp in ("LTC", "Hippocampus", "Frontal", "Insula"):
        sub = [r for r in results if r["site_group"] == grp]
        if not sub:
            continue
        rec_d = [r["recall_diff_low_pp"] for r in sub]
        dz_d = [r["dz_diff_low"] for r in sub]
        vlow = [r["recall_diff_very_low_pp"] for r in sub if np.isfinite(r["recall_diff_very_low_pp"])]
        mlow = [r["recall_diff_mod_low_pp"] for r in sub if np.isfinite(r["recall_diff_mod_low_pp"])]
        by_group[grp] = {
            "n_patients": len(sub),
            "n_stims": int(sum(r["n_stim_words"] for r in sub)),
            "mean_recall_diff_pp": float(np.mean(rec_d)),
            "se_recall_diff_pp": float(np.std(rec_d) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0,
            "mean_dz_diff": float(np.nanmean(dz_d)),
            "mean_very_low_pp": float(np.mean(vlow)) if vlow else float("nan"),
            "mean_mod_low_pp": float(np.mean(mlow)) if mlow else float("nan"),
        }
        print(f"  {grp:12s} (n={len(sub):2d}, {by_group[grp]['n_stims']:3d} stims) | recall Δ = {np.mean(rec_d):+6.2f} ± {by_group[grp]['se_recall_diff_pp']:.2f} pp | "
              f"neural Δz = {np.nanmean(dz_d):+.3f} SD | very-low {by_group[grp]['mean_very_low_pp']:+.1f} vs mod-low {by_group[grp]['mean_mod_low_pp']:+.1f} pp")

    # Compare LTC vs Non-LTC
    ltc_vals = [r["recall_diff_low_pp"] for r in results if r["site_group"] == "LTC"]
    non_vals = [r["recall_diff_low_pp"] for r in results if r["site_group"] != "LTC"]
    ltc_dz = [r["dz_diff_low"] for r in results if r["site_group"] == "LTC"]
    non_dz = [r["dz_diff_low"] for r in results if r["site_group"] != "LTC"]
    print(f"  LTC (n={len(ltc_vals)}) vs Non-LTC (n={len(non_vals)}): recall Δ {np.mean(ltc_vals):+.2f} vs {np.mean(non_vals):+.2f} pp (gap {np.mean(ltc_vals)-np.mean(non_vals):+.2f} pp) | "
          f"neural Δz {np.nanmean(ltc_dz):+.3f} vs {np.nanmean(non_dz):+.3f} SD")

    # Neural shift <-> behavioral recall link across patients
    dz_all = np.array([r["dz_diff_low"] for r in results])
    rc_all = np.array([r["recall_diff_low_pp"] for r in results])
    ok = np.isfinite(dz_all) & np.isfinite(rc_all)
    r_dz_rec = float(np.corrcoef(dz_all[ok], rc_all[ok])[0, 1])
    rho_dz_rec = float(spearmanr(dz_all[ok], rc_all[ok]).statistic)
    print(f"  Across-patient link (post-stim neural Δz vs recall Δ): r = {r_dz_rec:+.3f}, Spearman rho = {rho_dz_rec:+.3f}")

    # 4. Leave-one-patient-out causal policy evaluation
    policy = evaluate_lopo_policy(results)
    print("\n--- Leave-One-Patient-Out Offline Causal Policy Evaluation (Held-Out Patients) ---")
    print(f"  No stimulation baseline:            +0.00 pp")
    print(f"  Hospital online trigger (all sites): {policy['hospital_trigger_mean_pp']:+.2f} pp")
    print(f"  Permuted-site negative control:      {policy['permuted_site_ctrl_mean_pp']:+.2f} pp")
    print(f"  Learned LOPO causal policy:          {policy['lopo_policy_mean_pp']:+.2f} pp "
          f"(+{policy['lopo_vs_hospital_gain_pp']:.2f} pp over hospital trigger, p_vs_perm={policy['lopo_beats_perm_p']:.3f})")
    print(f"  Policy decisions: stimulate {policy['lopo_stimulated_count']}/{policy['n_patients']}, "
          f"withhold {policy['lopo_withheld_count']}/{policy['n_patients']} | "
          f"avoided {policy['harmful_sessions_avoided']}/{policy['harmful_sessions_lt_minus3pp']} strongly harmful (< -3 pp) sessions")

    summary = {
        "n_patients": len(results), "total_words": total_words, "total_stims": total_stims,
        "mean_auc_logreg": float(np.mean(aucs_lr)), "p_auc_logreg": float(t_lr.pvalue),
        "mean_auc_gru": float(np.mean(aucs_gru)), "p_auc_gru": float(t_gru.pvalue),
        "mean_auc_null": float(np.mean(aucs_null)),
        "mean_trigger_agreement_auc": float(np.nanmean(trigs)),
        "placebo_high_state_recall_diff_pp": float(np.nanmean(plac_rec)),
        "placebo_high_state_dz_diff": float(np.nanmean(plac_dz)),
        "by_site_group": by_group,
        "ltc_vs_nonltc_gap_pp": float(np.mean(ltc_vals) - np.mean(non_vals)),
        "neural_shift_vs_recall_r": r_dz_rec,
        "neural_shift_vs_recall_rho": rho_dz_rec,
        "lopo_policy": policy,
        "patients": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
