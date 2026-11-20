"""A brain model for each of 50 people, built from their own scans, and where stimulation would help each one.

Data: MICA-MICs (Royer et al. 2022, Scientific Data 9:569; CC0): 50 healthy adults, each with a diffusion MRI wiring map and
an 8-minute resting fMRI scan, on the same 100-region Schaefer atlas as human_brain_model.py. Their average activity map
matches the Human Connectome Project's at r = 0.91. Each person's wiring map links almost every pair of regions, mostly by a
few streamlines, so each person keeps their 1,107 strongest connections, as many as the group model has.

1. A model of each person: the model of human_brain_model.py, given that person's wiring and tuned to the first half of
   their scan, then tested on the second half, which it never saw. Compared with the person's own first-half map copied
   as it is, and with the average of everyone else's first halves.
2. Where to stimulate: each person's model, tuned to the whole scan, is stimulated one region at a time, as in
   human_brain_model.py, and reports the change in the global connectivity of the left lateral prefrontal cortex. Are the
   best sites the same for everyone? Are a person's best sites reliable, i.e. do models from the two halves of their scan
   agree?

The models tuned to each whole scan are saved to the data folder (personal_models.npz) for stimulation_controller.py.

Run (about 5 minutes on 14 cores):
    .venv/bin/python personal_brains.py data/mica_mics
"""
import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from human_brain_model import TARGET, pearson, stimulation_map, tune

CORTEX = np.r_[15:65, 66:116]  # MICA-MICs rows: 14 deep-brain structures, then each hemisphere's medial wall and 50 regions
EDGES = 1107                   # connections kept per person: the strongest ones
UPPER = np.triu_indices(100, 1)


def activity(signals):
    """Activity correlations between regions, z-transformed, negatives set to zero, like the group map."""
    c = np.corrcoef(signals.T)
    np.fill_diagonal(c, 0)
    return np.clip(np.arctanh(np.clip(c, -0.999, 0.999)), 0, None)


def scans(folder, person):
    """A person's wiring (strongest connections, scaled to a maximum of 1) and resting fMRI signals (frames, regions)."""
    base = f"{folder}/micapipe/sub-{person}/ses-01"
    signals = np.loadtxt(glob.glob(f"{base}/func/*schaefer100_desc-timeseries.txt")[0], delimiter=",")[:, CORTEX]
    streamlines = np.loadtxt(glob.glob(f"{base}/dwi/*schaefer100_desc-sc.txt")[0], delimiter=",")
    streamlines = (streamlines + streamlines.T)[np.ix_(CORTEX, CORTEX)]  # the file holds the upper half
    cutoff = np.sort(streamlines[UPPER])[::-1][EDGES - 1]
    wiring = np.where(streamlines >= cutoff, streamlines, 0.0)
    np.fill_diagonal(wiring, 0)
    return wiring / wiring.max(), signals


def model_person(args):
    """Tune one person's model to the first half, the second half and the whole of their scan."""
    folder, person, target = args
    torch.set_num_threads(1)
    wiring, signals = scans(folder, person)
    half = len(signals) // 2
    rng = np.random.default_rng(int(person[2:]))
    result = {"first_real": activity(signals[:half]), "second_real": activity(signals[half:])}
    for part, real in (("first", result["first_real"]), ("second", result["second_real"]), ("whole", activity(signals))):
        model = tune(wiring, real, UPPER, rng)
        with torch.no_grad():
            result[part] = model().numpy()
        result[f"{part}_map"] = stimulation_map(model, target)[1]
        if part == "whole":
            result["weights"], result["distance"] = model.weights().detach().numpy(), model.log_distance.exp().detach().numpy()
    return result


def main(folder):
    labels = open(Path(folder).parent / "hcp_connectome" / "strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    target = labels.index(TARGET)
    people = sorted({path.split("/sub-")[1][:5] for path in glob.glob(f"{folder}/micapipe/sub-*")})
    with Pool(10) as pool:
        runs = pool.map(model_person, [(folder, person, target) for person in people])
    n = len(people)
    print(f"{n} people, 100 regions, {EDGES} strongest connections each; scans split into halves")

    # 1. Personal models, scored on the second half of each scan.
    second = [run["second_real"][UPPER] for run in runs]
    match = np.array([[pearson(runs[p]["first"][UPPER], second[q]) for q in range(n)] for p in range(n)])
    own, others = np.diag(match), match[~np.eye(n, dtype=bool)].reshape(n, n - 1).mean(1)
    copied = [pearson(run["first_real"][UPPER], s) for run, s in zip(runs, second)]
    group = [pearson(np.mean([runs[q]["first_real"][UPPER] for q in range(n) if q != p], 0), second[p]) for p in range(n)]
    print("\n1. matching the second half of each person's scan (r, mean over people):")
    print(f"   person's model, tuned on the first half          {own.mean():.2f}")
    print(f"   same model, against other people's second halves  {others.mean():.2f}")
    print(f"   person's first-half map, copied as it is          {np.mean(copied):.2f}")
    print(f"   average of everyone else's first halves           {np.mean(group):.2f}")
    print(f"   people whose model matches their own scan best of all {n}: {int((match.argmax(1) == np.arange(n)).sum())}")
    print(f"   people whose model beats the group average on their own scan: {int((own > np.array(group)).sum())}")

    # 2. Where to stimulate, per person.
    maps = np.array([run["whole_map"] for run in runs])
    best = maps.argmax(1)
    sites, counts = np.unique(best, return_counts=True)
    common = sites[counts.argmax()]
    print(f"\n2. best single site to stimulate, per person: {len(sites)} different sites among {n} people")
    for k in np.argsort(counts)[::-1][:5]:
        print(f"   {labels[sites[k]]:<34} best for {counts[k]} people")
    gain = maps.max(1) / np.maximum(maps[:, common], 1e-12)
    print(f"   personal best site vs the most common best site, in each person's own model: "
          f"median {np.median(gain):.2f} times the effect (range {gain.min():.2f} to {gain.max():.2f})")
    print(f"   size of the best effect: median {np.median(maps.max(1)):.4f} z")
    halves = [(run["first_map"], run["second_map"]) for run in runs]
    within = [pearson(a, b) for a, b in halves]
    between = [pearson(halves[p][0], halves[q][1]) for p in range(n) for q in range(n) if p != q]
    same_top = np.mean([a.argmax() == b.argmax() for a, b in halves])
    print(f"   do a person's two half-scans give the same stimulation map? r = {np.mean(within):.2f} "
          f"(two different people: {np.mean(between):.2f}); same best site in both halves for {same_top:.0%} of people")

    np.savez_compressed(Path(folder) / "personal_models.npz", people=people, target=target,
                        weights=np.array([run["weights"] for run in runs]), distance=np.array([run["distance"] for run in runs]),
                        maps=maps)
    assert own.mean() > others.mean(), "personal models do not match their own person better than others"


if __name__ == "__main__":
    main(sys.argv[1])
