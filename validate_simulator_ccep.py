"""Validate the 100-region human whole-brain simulator against REAL human intracranial electrical stimulation.

Empirical causal ground truth: OpenNeuro ds004080 (CC0; van Blooijs et al., Nature Neuroscience 2023), 74 human patients
(ages 4–51) undergoing single-pulse electrical stimulation (SPES) with cortico-cortical evoked potential (CCEP) recording.
Each patient's stimulated electrode pairs and good recording electrodes are mapped via MNI152 coordinates (matching
hemisphere, <= 25 mm from centroid) to the exact 100-region Schaefer atlas of human_brain_model.py.

Model tested (ZERO electrical stimulation data used during fitting):
  - Our tuned whole-brain Hopf simulator (human_brain_model.py, tuned ONLY on HCP resting-state fMRI + diffusion MRI),
    AND our 50-person MICA-MICs personal models averaged (personal_models.npz).
  - Predicted causal response from stimulating region i to region j: exact linear-regime impulse response (Green's
    function G = K^{-1} where K = diag(W*1) - W + diag(d)), plus correlation-normalized response.
  - Compared against: Euclidean distance alone, diffusion MRI wiring alone, resting-state fMRI alone, untuned 1-parameter
    Hopf model, and a randomly rewired negative control — both raw (Spearman rho / Pearson r) AND partial correlation
    strictly controlling for Euclidean distance (the standard neuroscience test so spatial proximity cannot fool the score).

Run (~20 seconds):
    .venv/bin/python validate_simulator_ccep.py
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import spearmanr, rankdata

from human_brain_model import TARGET, Brain, correlations, load as load_hcp, pearson, rewire, tune, untuned

OUT = Path(__file__).with_name("validate_simulator_ccep.json")
MAX_DIST_MM = 25.0


def load_centroids():
    labels = open("data/hcp_connectome/strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    by_name = {}
    for row in csv.DictReader(open("data/hcp_connectome/schaefer100_centroids_mni.csv")):
        by_name[row["ROI Name"].strip()] = np.array([float(row["R"]), float(row["A"]), float(row["S"])])
    coords = np.stack([by_name[l] for l in labels])
    hemi = np.array(["L" if "_LH_" in l else "R" for l in labels])
    return labels, coords, hemi


def build_ccep_matrix(coords, hemi, exclude_soz=False):
    """Aggregate all 74 patients' SPES trials into 100x100 Schaefer empirical stimulation-response probability & latency."""
    mat = sio.loadmat("data/ccep_ds004080/ccepData_V2.mat", simplify_cells=True)["ccepData"]
    # Index electrode TSVs by subject id
    elec_files = {}
    for p in glob.glob("data/ccep_ds004080/electrodes/*.tsv"):
        sub = Path(p).name.split("_")[0]
        elec_files.setdefault(sub, []).append(p)

    trials = np.zeros((100, 100), dtype=np.int64)
    hits = np.zeros((100, 100), dtype=np.int64)
    lat_lists = [[[] for _ in range(100)] for _ in range(100)]
    n_patients_used, n_elec_mapped, n_raw_trials = 0, 0, 0

    for pat in mat:
        sub = pat["id"]
        efiles = elec_files.get(sub, [])
        if not efiles:
            continue
        # Match session if multiple electrode files exist
        ses = pat.get("ses", "")
        ef = next((p for p in efiles if f"_{ses}_" in p), efiles[0])
        elec_xyz, elec_roi, elec_soz = {}, {}, {}
        for row in csv.DictReader(open(ef), delimiter="\t"):
            name = row["name"].strip()
            try:
                xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            except ValueError:
                continue
            if not np.all(np.isfinite(xyz)):
                continue
            h = row.get("hemisphere", "").strip().upper()
            if h not in ("L", "R"):
                h = "L" if xyz[0] < 0 else "R"
            cand = np.flatnonzero(hemi == h)
            dists = np.linalg.norm(coords[cand] - xyz[None, :], axis=1)
            best_i = int(np.argmin(dists))
            elec_xyz[name] = (xyz, h)
            elec_soz[name] = (row.get("soz", "no").strip().lower() == "yes")
            if dists[best_i] <= MAX_DIST_MM:
                elec_roi[name] = int(cand[best_i])
                n_elec_mapped += 1

        runs = pat["run"] if isinstance(pat["run"], list) else [pat["run"]]
        pat_had = False
        for run in runs:
            if not isinstance(run, dict) or "n1_peak_sample" not in run:
                continue
            n1 = np.asarray(run["n1_peak_sample"], dtype=float)
            if n1.ndim == 1:
                n1 = n1[:, None]
            ch_names = [str(x).strip() for x in np.atleast_1d(run["channel_names"])]
            sp_names = [str(x).strip() for x in np.atleast_1d(run["stimpair_names"])]
            good_idx = np.atleast_1d(np.asarray(run["good_channels"], dtype=int)) - 1
            tt = np.asarray(run["tt"], dtype=float)

            # Map recording channels
            rec_rois = {}
            for ci in good_idx:
                if 0 <= ci < len(ch_names):
                    cn = ch_names[ci]
                    if cn in elec_roi and (not exclude_soz or not elec_soz.get(cn, False)):
                        rec_rois[ci] = (cn, elec_roi[cn])

            for sj, sp in enumerate(sp_names):
                if sj >= n1.shape[1] or "-" not in sp:
                    continue
                e1, e2 = [x.strip() for x in sp.split("-", 1)]
                if e1 not in elec_xyz or e2 not in elec_xyz:
                    continue
                if exclude_soz and (elec_soz.get(e1, False) or elec_soz.get(e2, False)):
                    continue
                mid = 0.5 * (elec_xyz[e1][0] + elec_xyz[e2][0])
                h = elec_xyz[e1][1]
                cand = np.flatnonzero(hemi == h)
                dists = np.linalg.norm(coords[cand] - mid[None, :], axis=1)
                bi = int(np.argmin(dists))
                if dists[bi] > MAX_DIST_MM:
                    continue
                src = int(cand[bi])
                for ci, (cn, dst) in rec_rois.items():
                    if cn in (e1, e2) or src == dst:
                        continue
                    trials[src, dst] += 1
                    n_raw_trials += 1
                    pat_had = True
                    val = n1[ci, sj]
                    if np.isfinite(val):
                        hits[src, dst] += 1
                        sidx = int(round(val)) - 1
                        if 0 <= sidx < len(tt):
                            lat_lists[src][dst].append(float(tt[sidx] * 1000.0))
        if pat_had:
            n_patients_used += 1

    prob = np.divide(hits, trials, out=np.full((100, 100), np.nan), where=trials > 0)
    lat = np.full((100, 100), np.nan)
    for i in range(100):
        for j in range(100):
            if lat_lists[i][j]:
                lat[i, j] = float(np.median(lat_lists[i][j]))
    return trials, hits, prob, lat, n_patients_used, n_elec_mapped, n_raw_trials


def partial_spearman(x, y, z):
    """Partial Spearman rank correlation between x and y controlling for covariate z (e.g. distance)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    # Residualize rx and ry on [1, rz, rz^2] to remove both linear and monotone-nonlinear distance decay
    A = np.column_stack([np.ones_like(rz), rz, (rz - rz.mean()) ** 2])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def greens_function(W, d):
    """Exact linear-regime Hopf Green's function G = K^{-1} (predicted steady impulse response from region i to j)."""
    K = np.diag(W.sum(1)) - W + np.diag(d)
    G = np.linalg.inv(K)
    sd = np.sqrt(np.maximum(np.diag(G), 1e-12))
    G_corr = G / sd[:, None] / sd[None, :]
    return G, G_corr


def main():
    labels, coords, hemi = load_centroids()
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    print("=== Building real human electrical stimulation (CCEP) matrix from OpenNeuro ds004080 (74 patients) ===", flush=True)
    trials, hits, prob, lat, n_pat, n_elec, n_raw = build_ccep_matrix(coords, hemi, exclude_soz=False)
    trials_ns, hits_ns, prob_ns, _, _, _, n_raw_ns = build_ccep_matrix(coords, hemi, exclude_soz=True)
    print(f"  All electrodes: {n_pat} patients mapped, {n_elec} electrodes within {MAX_DIST_MM:.0f} mm, {n_raw:,} directed stim-record trials")
    print(f"  Directed region pairs with >=10 trials: {int((trials >= 10).sum())} (mean response prob {np.nanmean(prob[trials >= 10]):.3f})")
    print(f"  Non-SOZ electrodes only: {n_raw_ns:,} trials, {int((trials_ns >= 10).sum())} pairs with >=10 trials", flush=True)

    # Fit our exact human_brain_model.py Hopf simulator (ZERO stimulation data seen!)
    wiring, real_fc, hcp_labels = load_hcp("data/hcp_connectome")
    assert hcp_labels == labels
    upper = np.triu_indices(100, 1)
    rng = np.random.default_rng(0)
    g0 = untuned(wiring, real_fc, upper)
    tuned = tune(wiring, real_fc, upper, rng)
    rewired_model = tune(rewire(wiring, rng), real_fc, upper, rng)

    with torch.no_grad():
        W_tuned = tuned.weights().numpy()
        d_tuned = tuned.log_distance.exp().numpy()
        W_rew = rewired_model.weights().numpy()
        d_rew = rewired_model.log_distance.exp().numpy()

    G_untuned, Gc_untuned = greens_function(g0 * wiring, np.ones(100))
    G_tuned, Gc_tuned = greens_function(W_tuned, d_tuned)
    G_rewired, Gc_rewired = greens_function(W_rew, d_rew)

    # Also average Green's function across our 50 personal MICA-MICs models!
    pm = np.load("data/mica_mics/personal_models.npz")
    G_mica = np.mean([greens_function(pm["weights"][i], pm["distance"][i])[0] for i in range(len(pm["people"]))], axis=0)
    Gc_mica = np.mean([greens_function(pm["weights"][i], pm["distance"][i])[1] for i in range(len(pm["people"]))], axis=0)

    predictors = {
        "Euclidean proximity (-distance)": -dist_mat,
        "HCP diffusion MRI wiring alone": wiring,
        "HCP resting-state fMRI alone": real_fc,
        "Untuned Hopf model (1-param G)": G_untuned,
        "Tuned Hopf simulator (G = K^-1)": G_tuned,
        "Tuned Hopf simulator (norm G)": Gc_tuned,
        "50-person MICA-MICs Hopf mean G": G_mica,
        "Randomly rewired Hopf (control)": G_rewired,
    }

    summary_out = {"n_patients": n_pat, "n_raw_trials": n_raw, "results": {}}
    for tag, tr_m, pr_m, min_n in (("all_electrodes_n>=10", trials, prob, 10),
                                   ("all_electrodes_n>=20", trials, prob, 20),
                                   ("non_SOZ_only_n>=10", trials_ns, prob_ns, 10)):
        mask = (tr_m >= min_n) & (~np.eye(100, dtype=bool))
        y = pr_m[mask]
        d_vec = dist_mat[mask]
        print(f"\n--- Predicting real human CCEP response probability ({tag}: {int(mask.sum())} directed pairs) ---")
        print(f"  {'Model / Predictor':<34} {'Spearman rho':>12} {'Pearson r':>10} {'Partial rho | dist':>19}")
        res_block = {}
        for name, mat_pred in predictors.items():
            x = mat_pred[mask]
            rho = float(spearmanr(x, y).statistic)
            pr = float(np.corrcoef(x, y)[0, 1])
            prho = partial_spearman(x, y, d_vec) if "distance" not in name else 0.0
            print(f"  {name:<34} {rho:+12.3f} {pr:+10.3f} {prho:+19.3f}")
            res_block[name] = {"spearman": rho, "pearson": pr, "partial_spearman_given_dist": prho}
        summary_out["results"][tag] = {"n_pairs": int(mask.sum()), "predictors": res_block}

    # Check N1 conduction latency correlation (does stronger model coupling -> faster real human neural transmission?)
    lat_mask = (hits >= 5) & np.isfinite(lat) & (~np.eye(100, dtype=bool))
    if lat_mask.sum() >= 20:
        print(f"\n--- Predicting real human N1 conduction latency ({int(lat_mask.sum())} pairs with >=5 evoked responses) ---")
        for name in ("Euclidean proximity (-distance)", "HCP diffusion MRI wiring alone", "HCP resting-state fMRI alone",
                     "Tuned Hopf simulator (G = K^-1)", "Randomly rewired Hopf (control)"):
            x = predictors[name][lat_mask]
            y_lat = lat[lat_mask]
            rho = float(spearmanr(x, y_lat).statistic)
            prho = partial_spearman(x, y_lat, dist_mat[lat_mask]) if "distance" not in name else 0.0
            print(f"  {name:<34} rho(pred, latency) = {rho:+.3f} | partial rho | dist = {prho:+.3f}")

    # Specifically inspect our intelligence-linked hub 7Networks_LH_Cont_PFCl_1!
    t_idx = labels.index(TARGET)
    # Regions with >=5 trials either stimulating into or recording from LH_Cont_PFCl_1
    sym_tr = trials[t_idx, :] + trials[:, t_idx]
    sym_ht = hits[t_idx, :] + hits[:, t_idx]
    hub_mask = (sym_tr >= 5) & (np.arange(100) != t_idx)
    if hub_mask.sum() >= 5:
        hub_p = sym_ht[hub_mask] / sym_tr[hub_mask]
        rho_hub = float(spearmanr(G_tuned[t_idx, hub_mask], hub_p).statistic)
        prho_hub = partial_spearman(G_tuned[t_idx, hub_mask], hub_p, dist_mat[t_idx, hub_mask])
        rho_hub_w = float(spearmanr(wiring[t_idx, hub_mask], hub_p).statistic)
        rho_hub_rew = float(spearmanr(G_rewired[t_idx, hub_mask], hub_p).statistic)
        print(f"\n--- Target hub {TARGET} specifically ({int(hub_mask.sum())} partner regions with >=5 real human stim trials) ---")
        print(f"  Tuned Hopf G:   Spearman rho = {rho_hub:+.3f} (partial | dist = {prho_hub:+.3f})")
        print(f"  MRI wiring:     Spearman rho = {rho_hub_w:+.3f}")
        print(f"  Rewired ctrl:   Spearman rho = {rho_hub_rew:+.3f}")
        top_emp = np.argsort(hub_p)[::-1][:6]
        hub_ids = np.flatnonzero(hub_mask)
        print("  Strongest empirical human CCEP partners of LH_Cont_PFCl_1:")
        for k in top_emp:
            r = hub_ids[k]
            print(f"    {labels[r]:<32} emp_prob={hub_p[k]:.2f} ({sym_ht[r]}/{sym_tr[r]}) | dist={dist_mat[t_idx,r]:4.1f}mm | Hopf_G={G_tuned[t_idx,r]:.3f}")
        summary_out["hub_validation"] = {
            "n_partners": int(hub_mask.sum()), "hopf_rho": rho_hub, "hopf_partial_rho": prho_hub,
            "wiring_rho": rho_hub_w, "rewired_rho": rho_hub_rew,
        }

    # Regularization sweep on non-SOZ CCEP: quantifies the resting-fMRI vs causal-stimulation trade-off
    print("\n--- Resting-fMRI fit vs real human causal CCEP trade-off (non-SOZ pairs, n>=10) ---")
    mask_ns = (trials_ns >= 10) & (~np.eye(100, dtype=bool))
    y_ns, d_ns = prob_ns[mask_ns], dist_mat[mask_ns]
    tradeoff = []

    def _fit_reg(lam=0.0, edges_frozen=False):
        m = Brain(wiring, g0)
        if edges_frozen:
            m.log_gain.requires_grad_(False)
            g_scale = torch.nn.Parameter(torch.tensor(0.0))
            params = [m.log_distance, g_scale]
        else:
            g_scale, params = None, list(m.parameters())
        opt = torch.optim.Adam(params, lr=0.02)
        target_fc = torch.tensor(real_fc[upper])
        log_g0 = float(np.log(g0))
        for _ in range(1500):
            opt.zero_grad()
            w = m.weights() * (g_scale.exp() if edges_frozen else 1.0)
            z = correlations(w, m.log_distance.exp())
            loss = ((z[upper] - target_fc) ** 2).mean()
            if lam > 0 and not edges_frozen:
                loss = loss + lam * ((m.log_gain - log_g0) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (m.weights() * (g_scale.exp() if edges_frozen else 1.0)).numpy(), m.log_distance.exp().numpy()

    sweep_models = [("Untuned Hopf (1 param g)", g0 * wiring, np.ones(100)),
                    ("Excitability-only Hopf (101p)", *_fit_reg(edges_frozen=True))]
    for lam in (1.0, 0.1, 0.01, 0.001, 0.0):
        sweep_models.append((f"Anchored Hopf (L2 lam={lam})", *_fit_reg(lam=lam)))

    for name, W_m, d_m in sweep_models:
        with torch.no_grad():
            fc_r = pearson(correlations(torch.tensor(W_m), torch.tensor(d_m)).numpy()[upper], real_fc[upper])
        G_m, _ = greens_function(W_m, d_m)
        rho = float(spearmanr(G_m[mask_ns], y_ns).statistic)
        prho = partial_spearman(G_m[mask_ns], y_ns, d_ns)
        hrho = float(spearmanr(G_m[t_idx, hub_mask], hub_p).statistic) if hub_mask.sum() >= 5 else float("nan")
        print(f"  {name:<30} | fMRI r={fc_r:.3f} | CCEP rho={rho:+.3f} (partial|dist {prho:+.3f}) | LH_PFC hub rho={hrho:+.3f}")
        tradeoff.append({"model": name, "fmri_r": fc_r, "ccep_rho": rho, "ccep_partial_rho": prho, "hub_rho": hrho})
    summary_out["fmri_vs_ccep_tradeoff"] = tradeoff

    OUT.write_text(json.dumps(summary_out, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
