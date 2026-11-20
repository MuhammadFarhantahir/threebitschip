"""Causally calibrated human whole-brain simulator: learn effective connectivity directly from real human
intracranial electrical stimulation (OpenNeuro ds004080, 74 patients, 277,638 non-SOZ trials) with strict
cross-validation, then test closed-loop prefrontal stimulation inside the causally validated brain.

Two strict held-out cross-validation tests:
  1. 5-fold pair CV:  tune effective Hopf parameters (directed edge gains anchored to HCP diffusion MRI + regional
                      excitability) on 80% of directed cortical pairs; score on the unseen 20% pairs.
  2. 5-fold source-region CV: hold out 20% of STIMULATED cortical regions completely (zero stimulations of those
                      regions seen during fitting); predict their whole-brain CCEP propagation from anatomy +
                      responses of the other 80% of regions.

Then fit on all non-SOZ CCEP pairs, verify resting-state & causal accuracy, and test closed-loop stimulation
(fixed rhythm, linear, residual MLP) targeting 7Networks_LH_Cont_PFCl_1 under the strict resting-synchrony safety rule.

Run (~2 minutes):
    .venv/bin/python causal_brain_simulator.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from human_brain_model import TARGET, correlations, load as load_hcp, pearson, rewire, untuned
from stimulation_controller import SITES, Brain, Linear, Rhythm, score, train
from stimulation_followups import ResidualNetwork
from validate_simulator_ccep import build_ccep_matrix, greens_function, load_centroids, partial_spearman

torch.set_default_dtype(torch.float64)
OUT = Path(__file__).with_name("causal_brain_simulator.json")


class DirectedHopf(torch.nn.Module):
    """Anatomy-anchored directed Hopf model: directed edge gains over real white-matter tracts + regional damping."""

    def __init__(self, wiring, g0):
        super().__init__()
        self.n = len(wiring)
        # Keep every anatomical edge in BOTH directions (i->j and j->i can differ, as real CCEP is asymmetric!)
        r, c = np.nonzero(wiring > 0)
        keep = r != c
        self.rows = torch.tensor(r[keep], dtype=torch.long)
        self.cols = torch.tensor(c[keep], dtype=torch.long)
        self.base_w = torch.tensor(wiring[r[keep], c[keep]], dtype=torch.float64)
        self.log_g0 = float(np.log(g0))
        self.log_gain = torch.nn.Parameter(torch.full((len(self.rows),), self.log_g0))
        self.log_d = torch.nn.Parameter(torch.zeros(self.n))
        # Monotone link from log(G_ij) -> CCEP response probability: sigmoid(a * log(G_ij + eps) + b)
        self.scale = torch.nn.Parameter(torch.tensor(1.5))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def weights_and_d(self):
        W = torch.zeros(self.n, self.n)
        W[self.rows, self.cols] = self.base_w * self.log_gain.exp()
        return W, self.log_d.exp()

    def green(self):
        W, d = self.weights_and_d()
        # Outgoing/incoming Laplacian: K = diag(W.sum(1)) - W + diag(d); K^-1[j, i] is response at j to pulse at i
        K = torch.diag(W.sum(1)) - W + torch.diag(d)
        G = torch.linalg.inv(K)
        return G.T  # entry [src, dst] = response at dst when stimulating src

    def predict_prob(self):
        G = self.green().clamp(min=1e-9)
        return torch.sigmoid(self.scale * torch.log(G) + self.bias), G


def fit_causal(wiring, g0, prob_mat, trials_mat, train_mask, lam_w=0.05, lam_d=0.05, steps=700):
    """Weighted binary cross-entropy fit to empirical human CCEP probabilities, anchored to anatomical wiring."""
    model = DirectedHopf(wiring, g0)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    m_idx = torch.tensor(train_mask)
    y = torch.tensor(np.nan_to_num(prob_mat[train_mask], nan=0.0))
    # Weight pairs by log(1 + trials) so heavily sampled pairs guide the fit without dominating everything
    w_pair = torch.tensor(np.log1p(trials_mat[train_mask]))
    w_pair = w_pair / w_pair.mean()
    for _ in range(steps):
        opt.zero_grad()
        p, _ = model.predict_prob()
        pm = p[m_idx].clamp(1e-6, 1 - 1e-6)
        bce = -(y * torch.log(pm) + (1 - y) * torch.log(1 - pm))
        loss = (w_pair * bce).mean()
        loss = loss + lam_w * ((model.log_gain - model.log_g0) ** 2).mean() + lam_d * (model.log_d ** 2).mean()
        loss.backward()
        opt.step()
    return model


def main():
    labels, coords, hemi = load_centroids()
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    trials, hits, prob, _, n_pat, _, n_raw = build_ccep_matrix(coords, hemi, exclude_soz=True)
    wiring, real_fc, _ = load_hcp("data/hcp_connectome")
    upper = np.triu_indices(100, 1)
    g0 = untuned(wiring, real_fc, upper)
    G_untuned, _ = greens_function(g0 * wiring, np.ones(100))

    valid = (trials >= 10) & (~np.eye(100, dtype=bool))
    valid_idx = np.argwhere(valid)
    print(f"=== Causal-Calibrated Human Whole-Brain Simulator ({n_pat} patients, {n_raw:,} non-SOZ trials, {len(valid_idx)} pairs) ===", flush=True)

    # 1. 5-fold pair CV (unseen directed pairs)
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(valid_idx)) % 5
    oof_causal = np.full((100, 100), np.nan)
    oof_rewired = np.full((100, 100), np.nan)
    w_rew = rewire(wiring, np.random.default_rng(1))
    g0_rew = untuned(w_rew, real_fc, upper)

    for f in range(5):
        tr_m = np.zeros((100, 100), dtype=bool)
        te_m = np.zeros((100, 100), dtype=bool)
        for k, (r, c) in enumerate(valid_idx):
            (te_m if folds[k] == f else tr_m)[r, c] = True
        m_c = fit_causal(wiring, g0, prob, trials, tr_m)
        m_r = fit_causal(w_rew, g0_rew, prob, trials, tr_m)
        with torch.no_grad():
            oof_causal[te_m] = m_c.green().numpy()[te_m]
            oof_rewired[te_m] = m_r.green().numpy()[te_m]

    # 2. 5-fold HELD-OUT STIMULATED-REGION CV (zero stimulations of held-out source regions seen during training!)
    src_regions = np.unique(valid_idx[:, 0])
    src_folds = np.random.default_rng(2).permutation(len(src_regions)) % 5
    src_fold_map = dict(zip(src_regions, src_folds))
    oof_src = np.full((100, 100), np.nan)
    for f in range(5):
        tr_m = np.zeros((100, 100), dtype=bool)
        te_m = np.zeros((100, 100), dtype=bool)
        for r, c in valid_idx:
            (te_m if src_fold_map[r] == f else tr_m)[r, c] = True
        m_s = fit_causal(wiring, g0, prob, trials, tr_m)
        with torch.no_grad():
            oof_src[te_m] = m_s.green().numpy()[te_m]

    y = prob[valid]
    d_vec = dist_mat[valid]
    rows = [
        ("HCP diffusion MRI wiring alone", wiring[valid]),
        ("Untuned Hopf (1 param g)", G_untuned[valid]),
        ("CCEP-calibrated Hopf (5-fold unseen pairs)", oof_causal[valid]),
        ("CCEP-calibrated Hopf (5-fold unseen stim regions)", oof_src[valid]),
        ("CCEP-calibrated rewired control (unseen pairs)", oof_rewired[valid]),
    ]
    print("\n--- Held-out real human electrical stimulation prediction (3,219 non-SOZ directed pairs) ---")
    print(f"  {'Model':<50} {'Spearman rho':>12} {'Pearson r':>10} {'Partial rho|dist':>17}")
    cv_summary = {}
    for name, pred in rows:
        rho = float(spearmanr(pred, y).statistic)
        pr = float(np.corrcoef(pred, y)[0, 1])
        prho = partial_spearman(pred, y, d_vec)
        print(f"  {name:<50} {rho:+12.3f} {pr:+10.3f} {prho:+17.3f}")
        cv_summary[name] = {"spearman": rho, "pearson": pr, "partial_spearman_given_dist": prho}

    # Check target prefrontal hub LH_Cont_PFCl_1 under strictly held-out pair CV!
    t_idx = labels.index(TARGET)
    hub_out = valid[t_idx, :]
    hub_in = valid[:, t_idx]
    hub_mask = hub_out | hub_in
    # Combine directed OOF predictions on valid directed entries involving LH_Cont_PFCl_1
    hub_dir_mask = np.zeros((100, 100), dtype=bool)
    hub_dir_mask[t_idx, :] = valid[t_idx, :]
    hub_dir_mask[:, t_idx] = valid[:, t_idx]
    yh, dh = prob[hub_dir_mask], dist_mat[hub_dir_mask]
    print(f"\n--- Target prefrontal hub {TARGET} ({int(hub_dir_mask.sum())} held-out directed human stim pairs) ---")
    for name, mat_p in (("Untuned Hopf", G_untuned), ("CCEP-calibrated Hopf (unseen pairs)", oof_causal),
                        ("CCEP-calibrated Hopf (unseen stim regions)", oof_src), ("Rewired control", oof_rewired)):
        rho = float(spearmanr(mat_p[hub_dir_mask], yh).statistic)
        prho = partial_spearman(mat_p[hub_dir_mask], yh, dh)
        print(f"  {name:<44} rho = {rho:+.3f} (partial|dist {prho:+.3f})")

    # 3. Fit full causal Hopf model (symmetrized W for exact closed-loop Brain compatibility, plus verify CCEP score)
    #    and run closed-loop stimulation targeting LH_Cont_PFCl_1 under the strict safety constraint!
    full_m = fit_causal(wiring, g0, prob, trials, valid)
    with torch.no_grad():
        W_dir, d_c = full_m.weights_and_d()
        W_sym = 0.5 * (W_dir.numpy() + W_dir.numpy().T)
        d_np = d_c.numpy()
        G_sym, _ = greens_function(W_sym, d_np)
        sym_rho = float(spearmanr(G_sym[valid], y).statistic)
        sym_prho = partial_spearman(G_sym[valid], y, d_vec)
        fc_r = pearson(correlations(torch.tensor(W_sym), torch.tensor(d_np)).numpy()[upper], real_fc[upper])
    print(f"\n=== Closed-loop control inside causally validated human brain models ===")
    print(f"  CCEP-calibrated symmetrized Hopf: CCEP rho={sym_rho:+.3f} (partial|dist {sym_prho:+.3f}), resting fMRI r={fc_r:.3f}", flush=True)

    sites = torch.tensor([labels.index(s) for s in SITES])
    regions = sorted({i for i, l in enumerate(labels) if "_Cont_" in l} | set(sites.tolist()))
    read = torch.tensor(regions + [r + 100 for r in regions])

    ctrl_out = {}
    for label, W_b, d_b in (("Untuned anatomy-Hopf (CCEP +0.616)", g0 * wiring, np.ones(100)),
                            ("CCEP-calibrated Hopf (causally validated)", W_sym, d_np)):
        brain = Brain(W_b, d_b, t_idx)
        rest = score(brain, None, sites, read, t_idx)
        scale = torch.tensor(np.r_[brain.sd, brain.sd][read])
        print(f"\n  Brain: {label} | rest hub={rest[0].mean():.3f}, rest sync={rest[1].mean():.3f}")
        ctrl_out[label] = {"rest_reach": float(rest[0].mean()), "rest_sync": float(rest[1].mean()), "controllers": {}}
        for name, cls in (("fixed rhythm", Rhythm), ("linear", Linear), ("residual", ResidualNetwork)):
            gains, syncs, ens = [], [], []
            n_seeds = 1 if name == "fixed rhythm" else 3
            for s in range(n_seeds):
                torch.manual_seed(s)
                st = train(brain, cls(len(read), len(sites), brain.energy, scale), sites, read, t_idx,
                           "steadily", float(rest[1].mean()), seed=10 + s)
                r, sy, en = score(brain, st, sites, read, t_idx)
                gains.append(float((r - rest[0]).mean()))
                syncs.append(float((sy - rest[1]).mean()))
                ens.append(en)
            print(f"    {name:<13} hub Δ = {np.mean(gains):+.4f} (seeds {[round(g,3) for g in gains]}) | "
                  f"sync Δ = {np.mean(syncs):+.4f} | energy {np.mean(ens):.0%}", flush=True)
            ctrl_out[label]["controllers"][name] = {
                "mean_reach_change": float(np.mean(gains)), "seeds": gains,
                "mean_sync_change": float(np.mean(syncs)), "energy": float(np.mean(ens)),
            }

    OUT.write_text(json.dumps({"cv_ccep": cv_summary, "closed_loop_causal": ctrl_out}, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
