"""Does a stimulator trained in a person's brain model raise that person's prefrontal hub connectivity while keeping the
rest of the brain normal? All 50 people of personal_brains.py, several training runs each.

For each person, the stimulators of stimulation_controller.py are trained for the steady goal: raise the global
connectivity of the left lateral prefrontal cortex while brain-wide synchrony stays at its resting level (neither up nor
down; an earlier version allowed it to fall, and some stimulators then lowered it a lot, which is no more normal).
The neural network and the linear closed-loop stimulators are trained from 3 random starts each, the fixed rhythm (open
loop) from 1, since it switched itself off in the earlier test. Each is scored on 64 new runs against no stimulation.
Each person's first network is also tried in the next person's model, to see whether control is personal.

Results are appended to stimulation_study.jsonl as they come, so an interrupted run keeps what it has and resumes where
it stopped. Run without arguments to print the summary.

Run (about 30 minutes on 14 cores):
    .venv/bin/python stimulation_study.py data/mica_mics
    .venv/bin/python stimulation_study.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from stimulation_controller import SITES, Brain, Linear, Network, Rhythm, score, train

RESULTS = Path(__file__).with_name("stimulation_study.jsonl")
KINDS = {"network": (Network, 3), "linear": (Linear, 3), "fixed rhythm": (Rhythm, 1)}  # stimulator, training runs


def job(args):
    """Train one stimulator for one person from one random start; score it there and, for the first network, in the next person."""
    folder, p, name, seed = args
    torch.set_num_threads(1)
    models = np.load(Path(folder) / "personal_models.npz")
    labels = open(Path(folder).parent / "hcp_connectome" / "strucLabels_ctx_schaefer_100.csv").read().strip().split(",")
    target, n = int(models["target"]), len(labels)
    sites = torch.tensor([labels.index(s) for s in SITES])
    regions = sorted({i for i, label in enumerate(labels) if "_Cont_" in label} | set(sites.tolist()))
    read = torch.tensor(regions + [r + n for r in regions])

    def setup(q):
        brain = Brain(models["weights"][q], models["distance"][q], target)
        rest = score(brain, None, sites, read, target)
        return brain, rest, torch.tensor(np.r_[brain.sd, brain.sd][read])

    brain, rest, scale = setup(p)
    kind = KINDS[name][0]
    torch.manual_seed(seed)
    stimulator = train(brain, kind(len(read), len(sites), brain.energy, scale), sites, read, target, "steadily", float(rest[1].mean()), seed=10 + seed)
    reach, synchrony, used = score(brain, stimulator, sites, read, target)
    change = reach - rest[0]
    row = {"person": str(models["people"][p]), "stimulator": name, "start": seed, "rest_reach": float(rest[0].mean()),
           "rest_synchrony": float(rest[1].mean()), "reach_change": float(change.mean()),
           "reach_change_se": float(change.std() / np.sqrt(len(change))), "synchrony_change": float((synchrony - rest[1]).mean()),
           "energy_used": used}
    if name == "network" and seed == 0:
        q = (p + 1) % len(models["people"])
        other, other_rest, other_scale = setup(q)
        transferred = Network(len(read), len(sites), other.energy, other_scale)
        transferred.load_state_dict(stimulator.state_dict())
        reach, synchrony, _ = score(other, transferred, sites, read, target)
        row.update(next_person=str(models["people"][q]), next_reach_change=float((reach - other_rest[0]).mean()),
                   next_synchrony_change=float((synchrony - other_rest[1]).mean()))
    return row


def summary():
    rows = [json.loads(line) for line in open(RESULTS)]
    people = sorted({r["person"] for r in rows})
    print(f"{len(rows)} trained stimulators, {len(people)} people")
    rest = {r["person"]: (r["rest_reach"], r["rest_synchrony"]) for r in rows}
    for name, (_, runs) in KINDS.items():
        per = {p: [r for r in rows if r["person"] == p and r["stimulator"] == name] for p in people}
        per = {p: v for p, v in per.items() if len(v) == runs}
        if not per:
            continue
        gain = np.array([np.mean([r["reach_change"] for r in v]) for v in per.values()])
        sync = np.array([np.mean([r["synchrony_change"] for r in v]) for v in per.values()])
        reliable = sum(all(r["reach_change"] > 2 * r["reach_change_se"] for r in v) for v in per.values())
        print(f"\n{name} ({len(per)} people, {runs} training run(s) each)")
        print(f"  change in hub connectivity: median {np.median(gain):+.3f}, mean {gain.mean():+.3f} (lowest {gain.min():+.3f}, highest {gain.max():+.3f})")
        print(f"  change in brain-wide synchrony: median {np.median(sync):+.3f} (largest size {np.abs(sync).max():.3f})")
        print(f"  people with a clear gain in every training run: {reliable} of {len(per)}")
        if runs > 1:
            starts = np.array([[r["reach_change"] for r in sorted(v, key=lambda r: r["start"])] for v in per.values()])
            print(f"  do different training runs agree on who gains? correlation between runs: {np.corrcoef(starts.T)[np.triu_indices(runs, 1)].mean():.2f}")
        resting = np.array([rest[p][0] for p in per])
        print(f"  does it help people whose hub is weakly connected at rest? correlation of gain with resting hub connectivity: {np.corrcoef(resting, gain)[0, 1]:+.2f}")
    transfer = [r for r in rows if "next_person" in r]
    if transfer:
        own = np.array([r["reach_change"] for r in transfer])
        moved = np.array([r["next_reach_change"] for r in transfer])
        print(f"\nnetwork trained on one person, tried in the next person ({len(transfer)} pairs): "
              f"gain in its own person {own.mean():+.3f}, in the next person {moved.mean():+.3f}; "
              f"better in its own person for {(own > moved).sum()} of {len(transfer)}")


def main(folder):
    done = {(r["person"], r["stimulator"], r["start"]) for r in map(json.loads, open(RESULTS))} if RESULTS.exists() else set()
    people = list(np.load(Path(folder) / "personal_models.npz")["people"])
    jobs = [(folder, p, name, seed) for seed in range(3) for name, (_, runs) in KINDS.items() if seed < runs
            for p in range(len(people)) if (people[p], name, seed) not in done]
    print(f"{len(jobs)} stimulators to train ({len(done)} already done)", flush=True)
    with Pool(12) as pool, open(RESULTS, "a") as out:
        for k, row in enumerate(pool.imap_unordered(job, jobs), 1):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if k % 25 == 0:
                print(f"{k} of {len(jobs)} done", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1]) if len(sys.argv) > 1 else summary()
