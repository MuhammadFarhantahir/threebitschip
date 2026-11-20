"""Stage-2 Non-Invasive Healthy-Human Bridge: Zero-Shot Validation of our Invasive CCEP-Calibrated
Whole-Brain Simulator on Real Healthy-Human TMS-EEG (OpenNeuro ds008037, Biabani & Rogasch, CC0).

Range-fetches [-100 ms, +250 ms] TMS-evoked EEG epochs for 10 healthy adults across 4 conditions:
  - prefrontal (left DLPFC, MNI [-46, 33, 34])
  - premotor   (left premotor, MNI [-23, 4, 63])
  - parietal   (left parietal, MNI [-26, -69, 63])
  - control    (left shoulder sham control — removes auditory click & somatosensory peripheral evoked potentials)

Zero free parameters are fit to TMS-EEG. We take cortical propagation G[src, :] from:
  1. CCEP-calibrated Hopf (trained ONLY on invasive epilepsy CCEP in OpenNeuro ds004080)
  2. Untuned anatomy-Hopf (1 param g on HCP diffusion MRI)
  3. Resting-fMRI-tuned Hopf (free-tuned on HCP resting fMRI)
  4. Rewired white-matter control
  5. Wrong-site negative control (permuted stimulation sites)
  6. Distance-from-coil-only null (no connectome)
and project the 100 Schaefer cortical parcels to the 62 standard 10-10 scalp electrodes (MNE fsaverage_1005.tsv)
via a parameter-free physical inverse-square leadfield L_{e,j} = 1 / (||r_e - r_j||^2 + 20^2).

Run (~2-3 minutes first run to range-fetch 10 subjects; <5 seconds once cached in data/tmseeg_ds008037/):
    .venv/bin/python validate_simulator_tmseeg.py

Use --expanded for the 47-candidate, two-site extension. The script range-fetches public
recordings if the derived waveform caches are unavailable.
"""
import argparse
import csv
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from causal_brain_simulator import fit_causal
from human_brain_model import load as load_hcp, rewire, tune, untuned
from validate_simulator_ccep import build_ccep_matrix, greens_function, load_centroids, partial_spearman

torch.set_default_dtype(torch.float64)

BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds008037"
CACHE_DIR = Path("data/tmseeg_ds008037")
CACHE_NPZ = CACHE_DIR / "tmseeg_10subj.npz"
OUT_JSON = Path(__file__).with_name("validate_simulator_tmseeg.json")

SUBJECTS = ["sub-001", "sub-005", "sub-007", "sub-008", "sub-010", "sub-012", "sub-013", "sub-014", "sub-016", "sub-018"]
CONDS = ["prefrontal", "premotor", "parietal", "control"]
ACTIVE_CONDS = ["prefrontal", "premotor", "parietal"]
EXPANDED_ACTIVE_CONDS = ["prefrontal", "parietal"]
EXPANDED_OUT = Path(__file__).with_name("validate_simulator_tmseeg_expanded.json")
EXPANDED_CANDIDATE_NUMBERS = (
    1, 5, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 21, 22, 24, 25,
    26, 27, 28, 29, 30, 32, 33, 34, 37, 38, 39, 40, 41, 43, 44, 45,
    46, 47, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63,
)

# Exact published MNI targets (Biabani et al., Imaging Neuroscience / Monash working-memory TMS-EEG battery)
TARGET_MNI = {
    "prefrontal": np.array([-46.0, 33.0, 34.0]),
    "premotor": np.array([-23.0, 4.0, 63.0]),
    "parietal": np.array([-26.0, -69.0, 63.0]),
}
PRE_MS, POST_MS = 100, 250  # 350 samples @ 1000 Hz


def _fetch(url, b0=None, b1=None):
    for _ in range(5):
        try:
            headers = {"Range": f"bytes={b0}-{b1}"} if b0 is not None else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = r.read()
            if b0 is None or len(data) == b1 - b0 + 1:
                return data
        except Exception:
            pass
    raise RuntimeError(f"Failed fetch {url} [{b0}-{b1}]")


def load_eeg_1005_mni():
    """Load MNE fsaverage_1005.tsv (meters -> mm, case-insensitive lookup)."""
    pos = {}
    with open(CACHE_DIR / "fsaverage_1005.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pos[r["label"].strip().upper()] = np.array([float(r["x"]), float(r["y"]), float(r["z"])]) * 1000.0
    return pos


def extract_subject_cond(subj, cond):
    """Range-fetch [-100, +250 ms] epochs around TMS pulses for one (subject, condition); return robust ERP (350, 62)."""
    ch_txt = _fetch(f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_channels.tsv").decode()
    chs = [r["name"].strip().upper() for r in csv.DictReader(io.StringIO(ch_txt), delimiter="\t")]
    ev_txt = _fetch(f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_events.tsv").decode()
    evs = [r for r in csv.DictReader(io.StringIO(ev_txt), delimiter="\t") if r.get("is_tms_pulse", "1") == "1"]
    samples = [int(r["sample"]) for r in evs]
    n_ch = len(chs)
    fdt_url = f"{BASE_URL}/{subj}/eeg/{subj}_task-tmseeg_acq-{cond}_eeg.fdt"
    fdt_sz = int(urllib.request.urlopen(urllib.request.Request(fdt_url, method="HEAD"), timeout=15).headers["Content-Length"])
    n_samp = fdt_sz // (4 * n_ch)
    win = PRE_MS + POST_MS

    valid_s = [s for s in samples if s - PRE_MS >= 0 and s + POST_MS <= n_samp]

    def grab_epoch(s):
        b0 = (s - PRE_MS) * n_ch * 4
        b1 = (s + POST_MS) * n_ch * 4 - 1
        raw = _fetch(fdt_url, b0, b1)
        ep = np.frombuffer(raw, dtype="<f4").reshape(win, n_ch).copy()
        ep -= ep[:PRE_MS - 15].mean(axis=0, keepdims=True)
        return ep

    with ThreadPoolExecutor(10) as ex:
        eps = np.stack(list(ex.map(grab_epoch, valid_s)), axis=0)  # (N_pulses, 350, 62)

    # Common-average reference using clean channels (ignoring extreme outlier channels > 200 uV in [35..180 ms])
    post_amp = np.median(np.abs(eps[:, PRE_MS + 35:PRE_MS + 180, :]), axis=(0, 1))
    good_ch = post_amp < 200.0
    if good_ch.sum() >= 20:
        eps = eps - eps[:, :, good_ch].mean(axis=2, keepdims=True)
    else:
        eps = eps - eps.mean(axis=2, keepdims=True)

    # Robust trial-median ERP (eliminates rare blink/muscle-twitch trials without hand-picking)
    erp = np.median(eps, axis=0).astype(np.float32)  # (350, 62)
    return chs, erp, len(valid_s)


def load_or_fetch_tmseeg():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_NPZ.exists():
        d = np.load(CACHE_NPZ, allow_pickle=True)
        return list(d["channels"]), d["erps"], d["n_pulses"]

    erps_all = []
    n_pulses_all = []
    ref_chs = None
    for si, subj in enumerate(SUBJECTS):
        sub_erps, sub_np = [], []
        for cond in CONDS:
            chs, erp, n_p = extract_subject_cond(subj, cond)
            if ref_chs is None:
                ref_chs = chs
            else:
                # Reorder channels to match ref_chs if order ever differs
                idx = [chs.index(c) for c in ref_chs]
                erp = erp[:, idx]
            sub_erps.append(erp)
            sub_np.append(n_p)
        erps_all.append(sub_erps)
        n_pulses_all.append(sub_np)
        print(f"  [{si + 1:2d}/{len(SUBJECTS)}] {subj}: fetched 4 conditions ({sum(sub_np)} TMS pulses)", flush=True)
    erps_arr = np.array(erps_all, dtype=np.float32)      # (10, 4, 350, 62)
    np_arr = np.array(n_pulses_all, dtype=np.int32)
    np.savez_compressed(CACHE_NPZ, channels=np.array(ref_chs), erps=erps_arr, n_pulses=np_arr, subjects=np.array(SUBJECTS))
    print(f"  saved {CACHE_NPZ} ({CACHE_NPZ.stat().st_size / 1e6:.2f} MB)", flush=True)
    return ref_chs, erps_arr, np_arr


def load_expanded_cohort():
    """Read derived median waveforms or reconstruct them from public ds008037."""
    from human_intelligence_wm_eeg import _fetch_tms_subject

    subjects = [f"sub-{number:03d}" for number in EXPANDED_CANDIDATE_NUMBERS]
    cache = CACHE_DIR / "tmseeg_expanded.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if data["subjects"].tolist() != subjects:
                raise ValueError(f"Unexpected candidate roster in {cache}")
            return data["channels"].tolist(), data["erps"]

    # Reuse the pilot's shared conditions and any available extension waveforms.
    # Visit archived BIDS sources in the fixed roster order for missing waveforms.
    pilot_channels, pilot_erps, _ = load_or_fetch_tmseeg()
    channels = list(pilot_channels)
    pilot_indices = [CONDS.index(cond) for cond in (*EXPANDED_ACTIVE_CONDS, "control")]
    waveforms = {subject: pilot_erps[si, pilot_indices] for si, subject in enumerate(SUBJECTS)}
    for name in ("tmseeg_25subj.npz", "tmseeg_replication.npz"):
        source = CACHE_DIR / name
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as data:
            source_channels = data["channels"].tolist()
            positions = [source_channels.index(channel) for channel in channels]
            for subject, erps in zip(data["subjects"].tolist(), data["erps"]):
                waveforms[subject] = erps[:, :, positions]

    for index, subject in enumerate(subjects, start=1):
        if subject in waveforms:
            continue
        result = _fetch_tms_subject(subject, channels)
        if result is None:
            raise RuntimeError(f"Unable to reconstruct waveforms for {subject}; retry the archive fetch")
        _, _, waveforms[subject] = result
        print(f"  Fetched candidate {index}/{len(subjects)}", flush=True)

    erps = np.stack([waveforms[subject] for subject in subjects])
    np.savez_compressed(cache, subjects=np.array(subjects), channels=np.array(channels), erps=erps)
    return channels, erps


def evaluate_tmseeg(chs, erps, active_conds, out_json, n_pulses=None):
    """Score fixed model predictions against participant and group scalp topographies."""
    mni_map = load_eeg_1005_mni()
    n_subjects, n_conditions = erps.shape[:2]
    n_active = len(active_conds)
    control_idx = n_active
    # Keep channels that exist in standard fsaverage_1005.tsv
    keep = [i for i, c in enumerate(chs) if c in mni_map]
    ch_names = [chs[i] for i in keep]
    ch_coords = np.array([mni_map[c] for c in ch_names], dtype=np.float64)  # (C, 3) in mm
    erps = erps[:, :, :, keep].astype(np.float64)  # (subjects, conditions, samples, channels)

    # Strict EEG QC: mask unplugged/saturated electrodes (post-artifact [35..180 ms] RMS > 40 uV),
    # re-reference to remaining clean channels, and drop corrupted subject-conditions (<45 clean ch)
    w_qc = slice(PRE_MS + 35, PRE_MS + 180)
    clean_erps = np.full_like(erps, np.nan)
    n_good_sc = 0
    for si in range(erps.shape[0]):
        for ci in range(n_conditions):
            e = erps[si, ci].copy()
            e -= e[:PRE_MS - 15].mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(e[w_qc] ** 2, axis=0))
            good = rms < 40.0
            if good.sum() >= 45:
                e[:, good] -= e[:, good].mean(axis=1, keepdims=True)
                clean_erps[si, ci][:, good] = e[:, good]
                n_good_sc += 1
    print(f"  Cohort: {n_subjects} adults, {len(ch_names)} mapped scalp electrodes, "
          f"{n_good_sc}/{n_subjects * n_conditions} conditions pass QC", flush=True)

    # Aggregate condition waveforms before evaluating group-median topographies.
    grand = np.nanmedian(clean_erps, axis=0)
    ctrl_erp = grand[control_idx]

    # Check both standard post-artifact cortical propagation windows: [35..100 ms] (primary) and [45..150 ms] (robustness)
    w_primary = slice(PRE_MS + 35, PRE_MS + 100)
    w_late = slice(PRE_MS + 45, PRE_MS + 150)

    # Shoulder-control-subtracted waveform RMS across [35, 100) ms.
    amp_net = np.zeros((n_active, len(ch_names)), dtype=np.float64)
    amp_late = np.zeros((n_active, len(ch_names)), dtype=np.float64)
    for si in range(n_active):
        diff_wave = grand[si] - ctrl_erp
        amp_net[si] = np.sqrt(np.mean(diff_wave[w_primary] ** 2, axis=0))
        amp_late[si] = np.sqrt(np.mean(diff_wave[w_late] ** 2, axis=0))
        top5 = [f"{ch_names[i]}({amp_net[si, i]:.2f}uV)" for i in np.argsort(amp_net[si])[::-1][:5]]
        print(f"  Empirical sham-subtracted {active_conds[si]:10s} TEP [35-100ms] top-5 scalp: {', '.join(top5)}")

    # Load Schaefer-100 centroids & build our 4 cortical propagation models (ZERO TMS-EEG fitting!)
    labels, sc_coords, hemi = load_centroids()
    trials, _, prob, _, _, _, _ = build_ccep_matrix(sc_coords, hemi, exclude_soz=True)
    wiring, real_fc, _ = load_hcp("data/hcp_connectome")
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    valid = (trials >= 10) & (~np.eye(100, dtype=bool))

    # 1. CCEP-calibrated Hopf (invasive epilepsy ds004080 only)
    m_ccep = fit_causal(wiring, g0, prob, trials, valid)
    with torch.no_grad():
        G_ccep = m_ccep.green().numpy()  # [src, dst]
    # 2. Untuned anatomy-Hopf (1 param g)
    G_untuned, _ = greens_function(g0 * wiring, np.ones(100))
    # 3. Free resting-fMRI-tuned Hopf (observational model that failed invasive CCEP!)
    tuned_fmri = tune(wiring, real_fc, upper, np.random.default_rng(0))
    with torch.no_grad():
        w_fmri = tuned_fmri.weights().numpy()
        d_fmri = tuned_fmri.log_distance.exp().numpy()
    G_fmri, _ = greens_function(w_fmri, d_fmri)
    # 4. Rewired control
    w_rew = rewire(wiring, np.random.default_rng(1))
    g0_rew = untuned(w_rew, real_fc, upper)
    G_rew, _ = greens_function(g0_rew * w_rew, np.ones(100))

    # Parameter-free physical volume-conduction leadfield L[e, j] = 1 / (||r_e - r_j||^2 + 20^2)
    dist_ec = np.linalg.norm(ch_coords[:, None, :] - sc_coords[None, :, :], axis=-1)  # (C, 100)
    L = 1.0 / (dist_ec ** 2 + 20.0 ** 2)
    L = L / L.sum(axis=1, keepdims=True)  # normalize per electrode so scalp geometry alone doesn't bias electrodes

    # Map each TMS target MNI coordinate to a localized Gaussian cortical activation over Schaefer-100 (FWHM ~ 15mm, sigma=12mm)
    src_weights = {}
    coil_dists = np.zeros((n_active, len(ch_names)), dtype=np.float64)
    for si, cond in enumerate(active_conds):
        pt = TARGET_MNI[cond]
        d_sc = np.linalg.norm(sc_coords - pt[None, :], axis=1)
        w_src = np.exp(-0.5 * (d_sc / 12.0) ** 2)
        w_src /= w_src.sum()
        src_weights[cond] = w_src
        top_p = labels[int(np.argmax(w_src))]
        print(f"  TMS target {cond:10s} MNI {pt.astype(int).tolist()} -> peak parcel {top_p} ({d_sc.min():.1f} mm)")
        coil_dists[si] = np.linalg.norm(ch_coords - pt[None, :], axis=1)

    def predict_scalp(G_mat, perm_sites=False):
        pred = np.zeros((n_active, len(ch_names)), dtype=np.float64)
        order = np.roll(np.arange(n_active), -1) if perm_sites else range(n_active)
        for si, src_i in enumerate(order):
            cond = active_conds[src_i]
            # Cortical steady/impulse propagation from stimulated patch -> 100 parcels -> project to 62 scalp channels
            cort_resp = src_weights[cond] @ G_mat  # (100,)
            pred[si] = L @ cort_resp
        return pred

    # Distance-only baseline (zero connectome: purely 1 / (dist_to_coil^2 + 20^2))
    pred_dist_only = 1.0 / (coil_dists ** 2 + 20.0 ** 2)

    models = [
        ("CCEP-calibrated Hopf (invasive -> non-invasive zero-shot)", predict_scalp(G_ccep)),
        ("Untuned anatomy-Hopf (1 param g on dMRI)", predict_scalp(G_untuned)),
        ("Resting-fMRI-tuned Hopf (observational)", predict_scalp(G_fmri)),
        ("Rewired white-matter control", predict_scalp(G_rew)),
        ("Wrong-site negative control (permuted targets)", predict_scalp(G_ccep, perm_sites=True)),
        ("Coil-distance-only null (no connectome)", pred_dist_only),
    ]

    # Site-demeaned contrast & z-scored per-site evaluation (prevents between-site intensity scaling from confounding pooled scores)
    y_all = amp_net.ravel()
    y_z = ((amp_net - amp_net.mean(axis=1, keepdims=True)) / amp_net.std(axis=1, keepdims=True)).ravel()
    y_diff = (amp_net - amp_net.mean(axis=0, keepdims=True)).ravel()
    y_late_diff = (amp_late - amp_late.mean(axis=0, keepdims=True)).ravel()
    d_all = coil_dists.ravel()

    print(f"\n--- Zero-Shot TMS–EEG Prediction ({n_active} sites × {len(ch_names)} electrodes) ---")
    summary = {}
    for name, P in models:
        P_z = ((P - P.mean(axis=1, keepdims=True)) / P.std(axis=1, keepdims=True)).ravel()
        P_diff = P - P.mean(axis=0, keepdims=True)
        per_site_part = [partial_spearman(P[i], amp_net[i], coil_dists[i]) for i in range(n_active)]
        per_site_raw = [float(spearmanr(P[i], amp_net[i]).statistic) for i in range(n_active)]
        mean_part = float(np.mean(per_site_part))
        zpool_part = partial_spearman(P_z, y_z, d_all)
        rho_diff = float(spearmanr(P_diff.ravel(), y_diff).statistic)
        rho_late = float(spearmanr(P_diff.ravel(), y_late_diff).statistic)
        # Per-subject consistency of mean per-site partial_spearman|dist across clean healthy humans
        sub_parts = []
        for s_idx in range(n_subjects):
            sp_i = []
            for k in range(n_active):
                s_net_k = np.sqrt(np.mean((clean_erps[s_idx, k] - clean_erps[s_idx, control_idx])[w_primary] ** 2, axis=0))
                ok_k = np.isfinite(s_net_k)
                if ok_k.sum() >= 45:
                    sp_i.append(partial_spearman(P[k, ok_k], s_net_k[ok_k], coil_dists[k, ok_k]))
            if len(sp_i) == n_active:
                sub_parts.append(float(np.mean(sp_i)))
        site_scores = ", ".join(f"{site} {score:+.3f}" for site, score in zip(active_conds, per_site_part))
        print(f"  {name}: {site_scores}; group mean {mean_part:+.3f}; "
              f"individual mean {np.mean(sub_parts):+.3f}, {sum(r > 0 for r in sub_parts)}/{len(sub_parts)} positive")
        summary[name] = {
            "per_site_partial_given_dist": dict(zip(active_conds, per_site_part)),
            "per_site_raw_spearman": dict(zip(active_conds, per_site_raw)),
            "mean_per_site_partial_given_dist": mean_part,
            "zpooled_partial_given_dist": zpool_part,
            "spearman_site_contrast_35_100ms": rho_diff,
            "spearman_site_contrast_45_150ms": rho_late,
            "indiv_mean_partial_given_dist": float(np.mean(sub_parts)),
            "indiv_positive_count": int(sum(r > 0 for r in sub_parts)),
            "indiv_n": len(sub_parts),
        }

    # Also test purely remote electrodes (> 45 mm from TMS coil) to rule out local TMS artifact completely!
    remote = d_all > 45.0
    print(f"\n--- Strictly Remote Electrodes Only (> 45 mm from TMS coil, n={int(remote.sum())}/{len(d_all)} pairs) ---")
    remote_summary = {}
    for name, P in models:
        P_diff = P - P.mean(axis=0, keepdims=True)
        r_rem = float(spearmanr(P.ravel()[remote], y_all[remote]).statistic)
        rd_rem = float(spearmanr(P_diff.ravel()[remote], y_diff[remote]).statistic)
        print(f"  {name:<58} remote rho = {r_rem:+.3f} | remote site-contrast rho = {rd_rem:+.3f}")
        remote_summary[name] = {"remote_rho": r_rem, "remote_site_contrast_rho": rd_rem}

    result = {
        "dataset": "OpenNeuro ds008037 (CC0, Biabani & Rogasch)",
        "dataset_doi": "10.18112/openneuro.ds008037.v1.0.1",
        "n_subjects": n_subjects,
        "n_conditions_passing_qc": n_good_sc,
        "n_conditions_evaluated": n_subjects * n_conditions,
        "active_sites": active_conds,
        "n_electrodes": len(ch_names),
        "results": summary,
        "remote_gt45mm": remote_summary,
    }
    if n_pulses is not None:
        result["total_pulses"] = int(n_pulses.sum())
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nsaved {out_json}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", action="store_true", help="Run the expanded two-site cohort")
    args = parser.parse_args()
    if args.expanded:
        channels, erps = load_expanded_cohort()
        evaluate_tmseeg(channels, erps, EXPANDED_ACTIVE_CONDS, EXPANDED_OUT)
    else:
        channels, erps, pulses = load_or_fetch_tmseeg()
        evaluate_tmseeg(channels, erps, ACTIVE_CONDS, OUT_JSON, pulses)


if __name__ == "__main__":
    main()
