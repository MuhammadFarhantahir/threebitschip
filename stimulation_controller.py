"""Train a neural network to control brain stimulation inside a person's brain model, the way robots are trained in
simulators, and compare it with the best fixed stimulation pattern.

Brains: the personal models of personal_brains.py (MICA-MICs), for the first three people. Each model runs here as a moving
system: 100 regions, each a noisy oscillator with the person's tuned excitability and connections, in the linear regime
the models were tuned in, oscillating at 0.05 Hz with a typical relaxation rate of 0.02 per second (the usual settings of
the Hopf whole-brain model; the tuning fixes neither). The model is advanced exactly from one second to the next, so a
10-minute run is 600 steps.

Stimulator: three electrodes at the sites the group model ranked best (left and right lateral prefrontal cortex, left
precentral cortex), updated once a second. It can read the activity of the 13 regions of the control network and of the
stimulated sites. Every stimulator gets the same energy: the same root-mean-square current over a run, set so that a steady
current of that size would shift the left prefrontal site's activity by one standard deviation.

Goal: raise the global connectivity of the left lateral prefrontal cortex, its average activity correlation with the other
99 regions over a run. Across people this predicts fluid intelligence (Cole et al. 2012, J Neurosci 32:8988); nobody has
shown that raising it raises a person's intelligence. Two versions of the goal:
  any way         raise it however the stimulator can
  selectively     raise it without raising brain-wide synchrony (the mean correlation of all pairs of other regions),
                  because driving the whole brain into one rhythm also raises it, and that is closer to a seizure than to
                  better thinking (the first version of this script found exactly that shortcut)

Stimulators, all trained by gradient descent through the simulated brain (300 rounds of 16 runs):
  fixed rhythm   each electrode plays a sine wave at the brain's own rhythm, amplitude and timing tuned (open loop)
  linear         currents are a tuned weighted sum of what it reads (closed loop)
  network        a small neural network turns what it reads into currents (closed loop)
Each is scored on 64 new runs with the same noise for all, against no stimulation. The network trained for the selective
goal in one person's model is also tried in the other two people's models.

Run (about 10 minutes on 14 cores):
    .venv/bin/python stimulation_controller.py data/mica_mics
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, solve_continuous_lyapunov
from torch import nn

torch.set_default_dtype(torch.float64)
PEOPLE = 3
STEP, STEPS = 1.0, 600       # seconds per stimulator update; updates per run (10 minutes)
RHYTHM, RELAX = 0.05, 0.02   # Hz; a typical region's relaxation rate per second
SITES = ["7Networks_LH_Cont_PFCl_1", "7Networks_RH_Cont_PFCl_2", "7Networks_LH_DorsAttn_PrCv_1"]
BATCH, ROUNDS, TEST_RUNS = 16, 300, 64
GOALS = ("any way", "selectively")


class Brain:
    """One person's model as a moving system, advanced exactly by STEP seconds at a time (linear regime)."""

    def __init__(self, weights, distance, target):
        n = len(distance)
        rate = RELAX / np.median(distance)  # model units -> per second
        k = (np.diag(weights.sum(1)) - weights + np.diag(distance)) * rate
        w = 2 * np.pi * RHYTHM * np.eye(n)
        jacobian = np.block([[-k, -w], [w, -k]])  # real and imaginary parts of each region's oscillation
        step = expm(jacobian * STEP)
        resting = solve_continuous_lyapunov(jacobian, -np.eye(2 * n))  # covariance at rest, for unit noise
        self.n = n
        self.step = torch.tensor(step.T)
        self.push = torch.tensor(np.linalg.solve(jacobian, step - np.eye(2 * n))[:, :n].T)  # a current held for one step
        self.start = torch.linalg.cholesky(torch.tensor(resting))
        self.noise = torch.linalg.cholesky(torch.tensor(resting - step @ resting @ step.T + 1e-12 * np.eye(2 * n)))
        self.sd = np.sqrt(np.diag(resting))
        self.energy = self.sd[target] / abs(np.linalg.inv(-jacobian)[target, target])  # steady current moving the target by 1 sd

    def run(self, stimulator, sites, read, runs, generator):
        """Activity (steps, runs, regions) and currents (steps, runs, sites) of `runs` runs from the resting state."""
        state = torch.randn(runs, 2 * self.n, generator=generator) @ self.start.T
        push = self.push[sites]
        activity, currents = [], []
        for t in range(STEPS):
            current = stimulator(state[:, read], t)
            state = state @ self.step + current @ push + torch.randn(runs, 2 * self.n, generator=generator) @ self.noise.T
            activity.append(state[:, : self.n])
            currents.append(current)
        return torch.stack(activity), torch.stack(currents)


def measures(activity, target):
    """Per run: the target's global connectivity, and brain-wide synchrony (mean correlation of all pairs of other regions)."""
    x = activity - activity.mean(0)
    x = x / x.norm(dim=0)
    c = torch.einsum("trn,trm->rnm", x, x)
    n = c.shape[1]
    others = ~torch.eye(n, dtype=torch.bool)
    others[target], others[:, target] = False, False
    return (c[:, target].sum(1) - 1) / (n - 1), c[:, others].mean(1)


class Rhythm(nn.Module):
    """Open loop: a sine wave per electrode at the brain's own rhythm."""

    def __init__(self, n_read, n_sites, energy, scale):
        super().__init__()
        self.waves, self.energy, self.gain = nn.Parameter(0.1 * torch.randn(2, n_sites)), energy, 1.0

    def forward(self, reading, t):
        phase = 2 * np.pi * RHYTHM * STEP * t
        return (self.gain * self.energy * (self.waves[0] * np.cos(phase) + self.waves[1] * np.sin(phase))).expand(len(reading), -1)


class Linear(nn.Module):
    """Closed loop: currents are a weighted sum of what it reads."""

    def __init__(self, n_read, n_sites, energy, scale):
        super().__init__()
        self.map, self.energy, self.scale, self.gain = nn.Linear(n_read, n_sites, bias=False), energy, scale, 1.0

    def forward(self, reading, t):
        return self.gain * self.energy * self.map(reading / self.scale)


class Network(Linear):
    """Closed loop: a small neural network turns what it reads into currents."""

    def __init__(self, n_read, n_sites, energy, scale):
        super().__init__(n_read, n_sites, energy, scale)
        self.map = nn.Sequential(nn.Linear(n_read, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, n_sites))


def train(brain, stimulator, sites, read, target, goal, resting_synchrony, seed):
    optimizer = torch.optim.Adam(stimulator.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(ROUNDS):
        activity, currents = brain.run(stimulator, sites, read, BATCH, generator)
        reach, synchrony = measures(activity, target)
        loss = -reach.mean() + 10 * torch.relu(currents.pow(2).mean().sqrt() / brain.energy - 1) ** 2  # energy budget
        if goal == "selectively":
            loss = loss + 50 * torch.relu(synchrony.mean() - resting_synchrony)
        if goal == "steadily":  # stricter: brain-wide synchrony must stay at its resting level, neither up nor down
            loss = loss + 50 * (synchrony.mean() - resting_synchrony).abs()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return stimulator


def score(brain, stimulator, sites, read, target):
    """Target connectivity, brain-wide synchrony and share of the energy budget used in 64 test runs. A stimulator that
    spends more than the budget is scaled down to it; one that spends less is left as trained."""
    with torch.no_grad():
        spend = lambda: float(brain.run(stimulator, sites, read, 16, torch.Generator().manual_seed(1))[1].pow(2).mean().sqrt())
        if stimulator is not None:
            stimulator.gain = 1.0
            if spend() > brain.energy:
                low, high = 1e-3, 1.0
                for _ in range(30):  # find the output scale that spends exactly the budget
                    stimulator.gain = (low * high) ** 0.5
                    low, high = (stimulator.gain, high) if spend() < brain.energy else (low, stimulator.gain)
        none = lambda reading, t: torch.zeros(len(reading), len(sites))
        activity, currents = brain.run(stimulator or none, sites, read, TEST_RUNS, torch.Generator().manual_seed(2))
        return [m.numpy() for m in measures(activity, target)] + [float(currents.pow(2).mean().sqrt() / brain.energy)]


def person(args):
    """Train the stimulators for both goals in one person's model; return their scores and the selective network."""
    weights, distance, target, sites, read = args
    torch.set_num_threads(3)
    brain = Brain(weights, distance, target)
    scale = torch.tensor(np.r_[brain.sd, brain.sd][read])
    results = {"none": score(brain, None, sites, read, target)}
    resting_synchrony = float(results["none"][1].mean())
    trained = {}
    for goal in GOALS:
        for name, kind in (("fixed rhythm", Rhythm), ("linear", Linear), ("network", Network)):
            torch.manual_seed(0)
            stimulator = kind(len(read), len(sites), brain.energy, scale)
            trained[goal, name] = train(brain, stimulator, sites, read, target, goal, resting_synchrony, seed=10)
            results[goal, name] = score(brain, trained[goal, name], sites, read, target)
    return results, trained["selectively", "network"].state_dict()


def main(folder):
    models = np.load(Path(folder) / "personal_models.npz")
    labels = open(Path(folder).parent / "hcp_connectome" / "strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    target, n = int(models["target"]), len(labels)
    sites = torch.tensor([labels.index(s) for s in SITES])
    regions = sorted({i for i, label in enumerate(labels) if "_Cont_" in label} | set(sites.tolist()))
    read = torch.tensor(regions + [r + n for r in regions])  # both components of each region read
    people = list(models["people"][:PEOPLE])
    print(f"people {', '.join(people)} | electrodes at {', '.join(s.replace('7Networks_', '') for s in SITES)} | "
          f"reading {len(regions)} regions | runs of {STEPS * STEP / 60:.0f} minutes")

    jobs = [(models["weights"][p], models["distance"][p], target, sites, read) for p in range(PEOPLE)]
    with Pool(PEOPLE) as pool:
        outcomes = pool.map(person, jobs)
    print("\nat rest: target connectivity " + ", ".join(f"{p} {o[0]['none'][0].mean():.3f}" for o, p in zip(outcomes, people)) +
          " | brain-wide synchrony " + ", ".join(f"{p} {o[0]['none'][1].mean():.3f}" for o, p in zip(outcomes, people)))
    for goal in GOALS:
        print(f"\ngoal: raise the target's connectivity {goal}. Change against no stimulation, mean of 64 test runs "
              "(target connectivity / brain-wide synchrony, share of the energy budget used), one column per person:")
        for name in ("fixed rhythm", "linear", "network"):
            cells = []
            for results, _ in outcomes:
                reach, synchrony, used = results[goal, name]
                cells.append(f"{(reach - results['none'][0]).mean():+.3f} / {(synchrony - results['none'][1]).mean():+.3f} ({used:.0%})")
            print(f"  {name:<13} " + " | ".join(cells))

    print("\nselective network stimulator trained in one person's model, tried in each (target connectivity / synchrony change):")
    for i, (_, state) in enumerate(outcomes):
        cells = []
        for j in range(PEOPLE):
            brain = Brain(models["weights"][j], models["distance"][j], target)
            network = Network(len(read), len(sites), brain.energy, torch.tensor(np.r_[brain.sd, brain.sd][read]))
            network.load_state_dict(state)
            reach, synchrony, used = score(brain, network, sites, read, target)
            rest = outcomes[j][0]["none"]
            cells.append(f"{people[j]} {(reach - rest[0]).mean():+.3f} / {(synchrony - rest[1]).mean():+.3f} ({used:.0%})")
        print(f"  trained on {people[i]}: " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
