"""A whole human brain, region by region, tuned to real human brain activity; then a test of where stimulation would most
strengthen the brain hub linked to intelligence.

Data: Human Connectome Project group maps from the ENIGMA toolbox (BSD 3-Clause; Lariviere et al. 2021, Nature Methods),
for the 100 cortical regions of the Schaefer atlas, each labelled with one of 7 brain networks.
  wiring:    connection strengths from diffusion MRI tractography (MRtrix3, SIFT2), group average, given on a log scale
  activity:  correlations between the regions' resting-state fMRI signals, z-transformed, negatives set to zero, averaged

Model: each region is a noisy oscillator just below the point where it would oscillate on its own (the Hopf whole-brain
model, Deco et al. 2017, Sci Rep 7:3095), linked to the others only where diffusion MRI finds a connection. In this regime
its resting activity correlations are those of the inverse of (network Laplacian + each region's distance from
oscillating), so they can be computed exactly. That makes the model quick to tune by gradient descent, the method used to
train neural networks: the strength of each real connection and each region's distance from oscillating are adjusted until
the model's activity correlations match the real ones.

Checks: the model is tuned on 80% of the region pairs and scored on the other 20%, which it never sees (5 folds), against
the wiring alone, the untuned model (one overall coupling strength), and the same tuning on randomly rewired connections.

Stimulation test: stimulation is modelled as bringing a region closer to oscillating on its own (Deco et al. 2019, PNAS
116:18088 model stimulation in this kind of model). The tuned model reports how stimulating each region changes the global
connectivity of the left lateral prefrontal cortex, its average activity correlation with every other region, which
predicts fluid intelligence across people (Cole et al. 2012, J Neurosci 32:8988; about 10% of the variance). Gradient
descent then looks for a pattern of several regions that does more with the same total amount of stimulation. These are
model predictions; nobody has shown that changing this connectivity raises a person's intelligence.

Run (about a minute):
    .venv/bin/python human_brain_model.py data/hcp_connectome
"""
import copy
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
FOLDS = 5
STIMULATION = 0.5                     # a stimulated region's distance from oscillating on its own is halved
TARGET = "7Networks_LH_Cont_PFCl_1"   # left lateral prefrontal cortex, in the control network
COUPLINGS = np.geomspace(1e-3, 10, 41)


def load(folder):
    read = lambda kind: np.loadtxt(f"{folder}/{kind}_ctx_schaefer_100.csv", delimiter=",")
    labels = open(f"{folder}/strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    log_wiring = read("strucMatrix")
    wiring = np.where(log_wiring > 0, np.exp(log_wiring), 0.0)  # undo the log scale; one pair with a negative log weight counts as unconnected
    return wiring / wiring.max(), read("funcMatrix"), labels


def correlations(weights, distance):
    """Exact resting activity correlations (z-transformed) of the linked oscillators."""
    covariance = torch.linalg.inv(torch.diag(weights.sum(1)) - weights + torch.diag(distance))
    sd = covariance.diagonal().sqrt()
    return torch.atanh((covariance / sd[:, None] / sd[None, :]).clamp(max=1 - 1e-9))


class Brain(torch.nn.Module):
    """The strength of every real connection (relative to the wiring map) and every region's distance from oscillating."""

    def __init__(self, wiring, coupling):
        super().__init__()
        self.n, self.edges = len(wiring), torch.tensor(np.argwhere(np.triu(wiring, 1) > 0).T)
        self.wiring = torch.tensor(wiring[tuple(self.edges)])
        self.log_gain = torch.nn.Parameter(torch.full((self.edges.shape[1],), float(np.log(coupling))))
        self.log_distance = torch.nn.Parameter(torch.zeros(self.n))

    def weights(self):
        w = torch.zeros(self.n, self.n).index_put(tuple(self.edges), self.wiring * self.log_gain.exp())
        return w + w.T

    def forward(self, stimulation=1.0):  # stimulation: each region's distance from oscillating is multiplied by this
        return correlations(self.weights(), self.log_distance.exp() * stimulation)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def untuned(wiring, real, pairs):
    """The single overall coupling strength whose exact correlations best match the real ones on `pairs`."""
    w, ones = torch.tensor(wiring), torch.ones(len(wiring))
    return max(COUPLINGS, key=lambda g: pearson(correlations(g * w, ones).numpy()[pairs], real[pairs]))


def tune(wiring, real, pairs, rng):
    """Gradient descent on `pairs`; 10% of them are set aside to decide when to stop."""
    order = rng.permutation(len(pairs[0]))
    stop, fit = (tuple(p[order[: len(order) // 10]] for p in pairs), tuple(p[order[len(order) // 10:]] for p in pairs))
    model = Brain(wiring, untuned(wiring, real, fit))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    target_fit, target_stop = torch.tensor(real[fit]), torch.tensor(real[stop])
    best, best_state, stale = np.inf, None, 0
    for _ in range(5000):
        optimizer.zero_grad()
        z = model()
        ((z[fit] - target_fit) ** 2).mean().backward()
        optimizer.step()
        with torch.no_grad():
            error = float(((model()[stop] - target_stop) ** 2).mean())
        best, best_state, stale = (error, copy.deepcopy(model.state_dict()), 0) if error < best else (best, best_state, stale + 1)
        if stale == 200:
            break
    model.load_state_dict(best_state)
    return model


def reach(z, target):
    """The target region's global connectivity: its mean activity correlation (z) with every other region."""
    return torch.cat([z[target, :target], z[target, target + 1:]]).mean()


def stimulation_map(model, target):
    """The target's global connectivity at rest, and its change when each region in turn is stimulated."""
    with torch.no_grad():
        base = float(reach(model(), target))
        changes = []
        for k in range(model.n):
            stimulation = torch.ones(model.n)
            stimulation[k] = STIMULATION
            changes.append(float(reach(model(stimulation), target)) - base)
    return base, np.array(changes)


def rewire(wiring, rng):
    """Control: the same connection strengths, placed between random pairs of regions."""
    upper = np.triu_indices(len(wiring), 1)
    rewired = np.zeros_like(wiring)
    rewired[upper] = rng.permutation(wiring[upper])
    return rewired + rewired.T


def main(folder):
    wiring, real, labels = load(folder)
    upper = np.triu_indices(len(real), 1)
    rng = np.random.default_rng(0)
    fold_of = rng.permutation(len(upper[0])) % FOLDS
    print(f"{len(labels)} regions, {int((wiring[upper] > 0).sum())} connected pairs of {len(upper[0])}; "
          f"{FOLDS} folds, each scored on {len(upper[0]) // FOLDS} pairs the model never sees")

    scores = {"wiring alone": [], "untuned model (one coupling strength)": [], "tuned model": [], "tuned model, randomly rewired": []}
    for fold in range(FOLDS):
        train = tuple(p[fold_of != fold] for p in upper)
        test = tuple(p[fold_of == fold] for p in upper)
        g = untuned(wiring, real, train)
        scores["wiring alone"].append(pearson(wiring[test], real[test]))
        scores["untuned model (one coupling strength)"].append(pearson(correlations(g * torch.tensor(wiring), torch.ones(len(wiring))).numpy()[test], real[test]))
        tuned, rewired = tune(wiring, real, train, rng), tune(rewire(wiring, rng), real, train, rng)
        with torch.no_grad():
            scores["tuned model"].append(pearson(tuned().numpy()[test], real[test]))
            scores["tuned model, randomly rewired"].append(pearson(rewired().numpy()[test], real[test]))
    print("match with the real activity correlations on unseen pairs (r, mean of folds, lowest to highest):")
    for name, values in scores.items():
        print(f"  {name:<38} {np.mean(values):.2f} ({min(values):.2f} to {max(values):.2f})")

    model = tune(wiring, real, upper, rng)  # tuned on every pair, for the stimulation test
    with torch.no_grad():
        print(f"tuned on every pair: r = {pearson(model().numpy()[upper], real[upper]):.2f}")
    target = labels.index(TARGET)
    base, changes = stimulation_map(model, target)
    order = np.argsort(changes)[::-1]
    network = lambda label: label.split("_")[2]
    print(f"\nglobal connectivity of {TARGET} in the tuned model: {base:.3f}; stimulating one region at a time, the largest increases:")
    for k in order[:8]:
        print(f"  {labels[k]:<34} {changes[k]:+.4f}")
    print(f"stimulating the target region itself: {changes[target]:+.4f} (rank {int(np.flatnonzero(order == target)[0]) + 1} of {len(labels)})")
    strength = wiring.sum(1)
    print(f"do the best sites simply have the most connections? correlation of increase with wiring strength: r = {pearson(changes, strength):.2f}")
    print("average increase by the stimulated region's network: " + ", ".join(
        f"{n} {changes[[network(label) == n for label in labels]].mean():+.4f}" for n in sorted({network(label) for label in labels})))

    # Several sites, same total amount of stimulation (the product of the distance multipliers equals one site's 0.5).
    budget = float(np.log(1 / STIMULATION))
    dose = torch.nn.Parameter(torch.full((len(labels),), -6.0))
    optimizer = torch.optim.Adam([dose], lr=0.05)
    for _ in range(3000):
        optimizer.zero_grad()
        spent = torch.nn.functional.softplus(dose)
        spent = spent * torch.clamp(budget / spent.sum(), max=1.0)  # never more than the budget
        (-reach(model(torch.exp(-spent)), target)).backward()
        optimizer.step()
    with torch.no_grad():
        spent = torch.nn.functional.softplus(dose)
        spent = (spent * torch.clamp(budget / spent.sum(), max=1.0)).numpy()
        gain = float(reach(model(torch.exp(-torch.tensor(spent))), target)) - base
    used = np.argsort(spent)[::-1]
    print(f"\nbest pattern found by gradient descent, same total stimulation: increase {gain:+.4f} "
          f"({gain / changes.max():.1f} times the best single site)")
    for k in used[:6]:
        print(f"  {labels[k]:<34} {spent[k] / budget:.0%} of the stimulation")
    assert np.mean(scores["tuned model"]) > np.mean(scores["wiring alone"]), "tuning did not beat the wiring alone on unseen pairs"


if __name__ == "__main__":
    main(sys.argv[1])
