"""Follow-up checks to make the 50-person closed-loop stimulation study airtight:

1. Cross-person transfer for the linear controller across all 50 people:
   - Retrain Linear (seed 0) for each person in the exact same setup as stimulation_study.py (verified against
     stimulation_study.jsonl) and test it in (a) its own model, (b) the next person's model (same 50 pairs as
     network transfer), (c) a leave-one-out group-average linear controller built from the other 49 people, and
     (d) a channel-permuted control.

2. Nonlinear whole-brain dynamics (full supercritical Hopf / Stuart-Landau oscillator with cubic saturation):
   - Uses exact Van Loan linear-stochastic transition + exact cubic amplitude flow r / sqrt(1 + 2*gamma*dt*r^2)
     over 2 substeps per second, with alpha = 2.0 (supercritical local excitability balanced by cubic saturation
     so resting activity remains realistic while dynamics are genuinely nonlinear limit cycles).
   - Evaluates 6 stratified people (2 weakest, 2 median, 2 strongest resting hubs), 2 random starts each, comparing
     fixed rhythm, linear, MLP network, and residual network (linear + zero-init MLP so nonlinear feedback has no
     optimization handicap).

Run (~6–8 minutes on 12 cores):
    .venv/bin/python stimulation_followups.py data/mica_mics
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm
from torch import nn

from stimulation_controller import RHYTHM, RELAX, SITES, STEP, STEPS, Brain, Linear, Network, Rhythm, score, train

OUT = Path(__file__).with_name("stimulation_followups.json")
STUDY = Path(__file__).with_name("stimulation_study.jsonl")
ALPHA, SUBSTEPS = 2.0, 2


class HopfBrain(Brain):
    """Full nonlinear Hopf (Stuart-Landau) whole-brain model via exact linear + cubic operator splitting."""

    def __init__(self, weights, distance, target, alpha=ALPHA, substeps=SUBSTEPS):
        super().__init__(weights, distance, target)
        self.substeps = substeps
        dt = STEP / substeps
        n = len(distance)
        rate = RELAX / np.median(distance)
        k_nl = (np.diag(weights.sum(1)) - weights + np.diag(distance * (1.0 - alpha))) * rate
        w = 2 * np.pi * RHYTHM * np.eye(n)
        jac = np.block([[-k_nl, -w], [w, -k_nl]])
        e_half = expm(jac * (dt / 2.0))
        step = e_half @ e_half
        # Simpson quadrature for exact noise integral int_0^dt exp(J s) exp(J^T s) ds — guaranteed SPD for any J
        q_cov = (dt / 6.0) * (np.eye(2 * n) + 4.0 * (e_half @ e_half.T) + (step @ step.T))
        aug = np.block([[jac, np.eye(2 * n)], [np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n))]]) * dt
        push_full = expm(aug)[:2 * n, 2 * n:]
        self.sub_step = torch.tensor(step.T)
        self.sub_push = torch.tensor(push_full[:, :n].T)
        self.sub_noise = torch.linalg.cholesky(torch.tensor(q_cov))
        gamma = alpha * (distance * rate) / (2.0 * self.sd[:n] ** 2)
        self.two_gamma_dt = torch.tensor(2.0 * gamma * dt)

    def run(self, stimulator, sites, read, runs, generator):
        state = torch.randn(runs, 2 * self.n, generator=generator) @ self.start.T
        push = self.sub_push[sites]
        activity, currents = [], []
        n = self.n
        for t in range(STEPS):
            current = stimulator(state[:, read], t)
            drive = current @ push
            for _ in range(self.substeps):
                state = state @ self.sub_step + drive + torch.randn(runs, 2 * n, generator=generator) @ self.sub_noise.T
                r2 = state[:, :n].pow(2) + state[:, n:].pow(2)
                sat = torch.rsqrt(1.0 + self.two_gamma_dt * r2)
                state = state * torch.cat([sat, sat], dim=1)
            activity.append(state[:, :n])
            currents.append(current)
        return torch.stack(activity), torch.stack(currents)


class ResidualNetwork(Linear):
    """Linear feedback plus a small MLP initialized at zero at its output, so nonlinear terms only help if needed."""

    def __init__(self, n_read, n_sites, energy, scale):
        super().__init__(n_read, n_sites, energy, scale)
        self.nl = nn.Sequential(nn.Linear(n_read, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_sites))
        nn.init.zeros_(self.nl[-1].weight)
        nn.init.zeros_(self.nl[-1].bias)

    def forward(self, reading, t):
        z = reading / self.scale
        return self.gain * self.energy * (self.map(z) + self.nl(z))


def _load_meta(folder):
    models = np.load(Path(folder) / "personal_models.npz")
    labels = open(Path(folder).parent / "hcp_connectome" / "strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    target, n = int(models["target"]), len(labels)
    sites = torch.tensor([labels.index(s) for s in SITES])
    regions = sorted({i for i, label in enumerate(labels) if "_Cont_" in label} | set(sites.tolist()))
    read = torch.tensor(regions + [r + n for r in regions])
    return models, target, sites, read


def _fit_linear_job(args):
    folder, p = args
    torch.set_num_threads(1)
    models, target, sites, read = _load_meta(folder)

    def setup(q):
        brain = Brain(models["weights"][q], models["distance"][q], target)
        rest = score(brain, None, sites, read, target)
        return brain, rest, torch.tensor(np.r_[brain.sd, brain.sd][read])

    brain, rest, scale = setup(p)
    torch.manual_seed(0)
    stim = train(brain, Linear(len(read), len(sites), brain.energy, scale), sites, read, target,
                 "steadily", float(rest[1].mean()), seed=10)
    reach, sync, used = score(brain, stim, sites, read, target)
    w_eff = (stim.gain * stim.map.weight.detach()).numpy()

    # Transfer to next person
    q = (p + 1) % len(models["people"])
    other, other_rest, other_scale = setup(q)
    tr = Linear(len(read), len(sites), other.energy, other_scale)
    tr.load_state_dict(stim.state_dict())
    tr_reach, tr_sync, tr_used = score(other, tr, sites, read, target)

    # Channel-permuted negative control in own person
    ctrl = Linear(len(read), len(sites), brain.energy, scale)
    perm = torch.tensor(np.random.default_rng(100 + p).permutation(len(read)))
    with torch.no_grad():
        ctrl.map.weight.copy_(stim.map.weight[:, perm])
    c_reach, c_sync, _ = score(brain, ctrl, sites, read, target)

    return {
        "person": str(models["people"][p]),
        "idx": p,
        "rest_reach": float(rest[0].mean()),
        "own_reach_change": float((reach - rest[0]).mean()),
        "own_sync_change": float((sync - rest[1]).mean()),
        "own_energy": used,
        "next_person": str(models["people"][q]),
        "next_reach_change": float((tr_reach - other_rest[0]).mean()),
        "next_sync_change": float((tr_sync - other_rest[1]).mean()),
        "next_energy": tr_used,
        "perm_reach_change": float((c_reach - rest[0]).mean()),
        "weight": w_eff.tolist(),
    }


def _eval_loo_job(args):
    folder, p, mean_w = args
    torch.set_num_threads(1)
    models, target, sites, read = _load_meta(folder)
    brain = Brain(models["weights"][p], models["distance"][p], target)
    rest = score(brain, None, sites, read, target)
    scale = torch.tensor(np.r_[brain.sd, brain.sd][read])
    stim = Linear(len(read), len(sites), brain.energy, scale)
    with torch.no_grad():
        stim.map.weight.copy_(torch.tensor(mean_w))
    reach, sync, used = score(brain, stim, sites, read, target)
    return float((reach - rest[0]).mean()), float((sync - rest[1]).mean()), used


def _nonlinear_job(args):
    folder, p, name, seed = args
    torch.set_num_threads(1)
    models, target, sites, read = _load_meta(folder)
    brain = HopfBrain(models["weights"][p], models["distance"][p], target, alpha=ALPHA, substeps=SUBSTEPS)
    rest = score(brain, None, sites, read, target)
    scale = torch.tensor(np.r_[brain.sd, brain.sd][read])
    cls = {"fixed rhythm": Rhythm, "linear": Linear, "network": Network, "residual": ResidualNetwork}[name]
    torch.manual_seed(seed)
    stim = train(brain, cls(len(read), len(sites), brain.energy, scale), sites, read, target,
                 "steadily", float(rest[1].mean()), seed=10 + seed)
    reach, sync, used = score(brain, stim, sites, read, target)
    change = reach - rest[0]
    return {
        "person": str(models["people"][p]),
        "idx": p,
        "stimulator": name,
        "start": seed,
        "nl_rest_reach": float(rest[0].mean()),
        "nl_rest_sync": float(rest[1].mean()),
        "reach_change": float(change.mean()),
        "reach_change_se": float(change.std() / np.sqrt(len(change))),
        "sync_change": float((sync - rest[1]).mean()),
        "energy_used": used,
    }


def main(folder):
    models, target, sites, read = _load_meta(folder)
    n_people = len(models["people"])
    print(f"=== Part 1: Linear controller cross-person transfer ({n_people} people) ===", flush=True)
    with Pool(8) as pool:
        lin_rows = pool.map(_fit_linear_job, [(folder, p) for p in range(n_people)])

    # Verify exact agreement with stimulation_study.jsonl
    if STUDY.exists():
        ref = {r["person"]: r["reach_change"] for r in map(json.loads, open(STUDY))
               if r["stimulator"] == "linear" and r["start"] == 0}
        max_diff = max(abs(r["own_reach_change"] - ref[r["person"]]) for r in lin_rows)
        print(f"verified against stimulation_study.jsonl (linear seed 0): max |diff| = {max_diff:.2e}", flush=True)
        assert max_diff < 1e-6, f"mismatch with stimulation_study.jsonl: {max_diff}"

    # Leave-one-out group-average linear controller
    weights = np.array([r["weight"] for r in lin_rows])
    normed = weights / np.maximum(np.linalg.norm(weights, axis=(1, 2), keepdims=True), 1e-12)
    loo_jobs = [(folder, p, np.delete(normed, p, axis=0).mean(0)) for p in range(n_people)]
    with Pool(12) as pool:
        loo_res = pool.map(_eval_loo_job, loo_jobs)
    for r, (lr, ls, le) in zip(lin_rows, loo_res):
        r.update(loo_reach_change=lr, loo_sync_change=ls, loo_energy=le)

    own = np.array([r["own_reach_change"] for r in lin_rows])
    nxt = np.array([r["next_reach_change"] for r in lin_rows])
    loo = np.array([r["loo_reach_change"] for r in lin_rows])
    prm = np.array([r["perm_reach_change"] for r in lin_rows])
    print(f"  own person (n=50):              mean {own.mean():+.4f}, median {np.median(own):+.4f}")
    print(f"  next person (50 pairs):         mean {nxt.mean():+.4f}, median {np.median(nxt):+.4f} | own > next: {(own > nxt).sum()}/50")
    print(f"  LOO group-average linear:       mean {loo.mean():+.4f}, median {np.median(loo):+.4f} | own > LOO:  {(own > loo).sum()}/50 | LOO > 0: {(loo > 0).sum()}/50")
    print(f"  channel-permuted control:       mean {prm.mean():+.4f}, median {np.median(prm):+.4f}", flush=True)

    # Part 2: Stratified 6 people on full supercritical nonlinear Hopf brain (alpha=2.0)
    order = np.argsort([r["rest_reach"] for r in lin_rows])
    chosen = [int(order[0]), int(order[1]), int(order[24]), int(order[25]), int(order[-2]), int(order[-1])]
    chosen_names = [f"{models['people'][p]}(rest={lin_rows[p]['rest_reach']:.3f})" for p in chosen]
    print(f"\n=== Part 2: Supercritical nonlinear Hopf (alpha={ALPHA}, substeps={SUBSTEPS}) on 6 stratified people ===", flush=True)
    print("  chosen:", ", ".join(chosen_names), flush=True)
    nl_jobs = (
        [(folder, p, "fixed rhythm", 0) for p in chosen]
        + [(folder, p, kind, seed) for seed in range(2) for kind in ("linear", "network", "residual") for p in chosen]
    )
    print(f"  running {len(nl_jobs)} nonlinear stimulator jobs...", flush=True)
    with Pool(8) as pool:
        nl_rows = pool.map(_nonlinear_job, nl_jobs)

    for kind in ("fixed rhythm", "linear", "network", "residual"):
        sub = [r for r in nl_rows if r["stimulator"] == kind]
        by_p = [np.mean([r["reach_change"] for r in sub if r["idx"] == p]) for p in chosen]
        sync_p = [np.mean([r["sync_change"] for r in sub if r["idx"] == p]) for p in chosen]
        en_p = [np.mean([r["energy_used"] for r in sub if r["idx"] == p]) for p in chosen]
        print(f"  {kind:<13} mean {np.mean(by_p):+.4f} (med {np.median(by_p):+.4f}) | max|sync| {np.max(np.abs(sync_p)):.4f} | energy {np.mean(en_p):.0%}")
        print("    per person: " + " | ".join(f"{models['people'][p]} {g:+.3f} (s{s:+.3f})" for p, g, s in zip(chosen, by_p, sync_p)))

    OUT.write_text(json.dumps({
        "linear_transfer": [{k: v for k, v in r.items() if k != "weight"} for r in lin_rows],
        "nonlinear_hopf": nl_rows,
        "alpha": ALPHA,
        "substeps": SUBSTEPS,
        "chosen_people": [str(models["people"][p]) for p in chosen],
    }, indent=2))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
