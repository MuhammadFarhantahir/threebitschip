"""How many bits do the decoder's weights need?

Trains the GRU decoder of gru_decoder.py, stores its weight matrices with fewer bits (8 down to 2), and scores
every version on the same held-out reaches. A weight stored with b bits is one of 2^b values,
step * {-2^(b-1), ..., 2^(b-1) - 1}, with one step size per weight matrix. Two ways of getting there:

  rounded:    round the trained weights to the nearest allowed value. No further training.
  retrained:  keep training with the rounding in the forward pass, so the network adapts to it. The step sizes
              are learned too (learned step size quantization, Esser et al. 2020, arXiv:1902.08153).

"retrained, 32 bits" is the reference: the same extra training with no rounding, so that a gain from the
extra training is not mistaken for a gain from the rounding. The held-out reaches are used only for scoring.

Run one network size and seed (adds its rows to precision_sweep.jsonl), then print the table of all runs:
    .venv/bin/python precision_sweep.py data/mc_maze_train.nwb 64 0
    .venv/bin/python precision_sweep.py
"""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn

from gru_decoder import Decoder, fit_gru, prepare, score

BITS = (8, 6, 5, 4, 3, 2)
DROPOUT = {256: (0.4, 0.4), 64: (0.2, 0.2)}  # per network size, each chosen earlier on the tuning reaches
RESULTS = "precision_sweep.jsonl"


def best_step(w, bits):
    """The step size that leaves the rounded matrix closest to the original (smallest squared error)."""
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    steps = w.abs().max() / -lo * torch.linspace(0.02, 1, 200)
    return min(steps, key=lambda s: ((w / s).round().clamp(lo, hi) * s - w).square().sum())


class Rounded(nn.Module):
    """A decoder whose weight matrices are rounded to `bits` bits on every forward pass."""

    # ponytail: weights only, one step per matrix, biases and the GRU's running state left at full precision.
    # A chip needs those in integers too; that takes a hand-written GRU step, the next thing to build.
    def __init__(self, decoder, bits):
        super().__init__()
        self.decoder, self.lo, self.hi = decoder, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.names = [name for name, p in decoder.named_parameters() if p.dim() == 2]
        # Learned as logarithms, so a step stays positive and the optimizer changes it by a small fraction at a time.
        self.log_steps = nn.ParameterList(
            nn.Parameter(best_step(decoder.get_parameter(name).detach(), bits).log()) for name in self.names)

    def weights(self):
        rounded = {}
        for name, log_step in zip(self.names, self.log_steps):
            step = log_step.exp()
            v = torch.clamp(self.decoder.get_parameter(name) / step, self.lo, self.hi)
            rounded[name] = (v + (v.round() - v).detach()) * step  # round going forward, pass the gradient straight through
        return rounded

    def forward(self, counts):
        return torch.func.functional_call(self.decoder, self.weights(), (counts,))


def main(path, hidden=256, seed=0):
    d = prepare(path, seed)
    held, wrong = d.held_out, np.roll(d.held_out, 1)
    model = fit_gru(Decoder(d.x.shape[-1], d.y.shape[1], hidden, DROPOUT[hidden]), d)
    matrices = sum(p.numel() for p in model.parameters() if p.dim() == 2)

    def record(method, bits, m):
        row = dict(hidden=hidden, seed=seed, method=method, bits=bits, kB=matrices * bits / 8000,
                   tune=score(m, d.x[d.tune], d.y[d.tune]), held_out=score(m, d.x[held], d.y[held]),
                   control=score(m, d.x[wrong], d.y[held]))  # control: another reach's neurons, should be 0 or below
        print(json.dumps(row), flush=True)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    full = record("trained", 32, model)
    assert full["control"] < 0.1 < full["held_out"], "decoder self-check failed"
    # Self-check of the rounding: 16 bits must change nothing, 2 bits must change the output, and a
    # training step through the rounding must reach both the weights and the step sizes.
    fine, coarse, sample = Rounded(copy.deepcopy(model), 16), Rounded(copy.deepcopy(model), 2), d.x[held[:8]]
    exact = model(sample).detach()
    assert (fine(sample) - exact).abs().max() < 1e-2 < (coarse(sample) - exact).abs().max(), "rounding self-check failed"
    coarse(sample).sum().backward()
    assert coarse.decoder.gru.weight_hh_l0.grad.abs().sum() > 0 and coarse.log_steps[0].grad is not None, "gradient self-check failed"

    for bits in BITS:
        record("rounded", bits, Rounded(copy.deepcopy(model), bits))
    for bits in (32, *BITS):
        torch.manual_seed(1000 * seed + bits)  # each retraining has its own random stream, whatever ran before it
        start = copy.deepcopy(model)
        record("retrained", bits, fit_gru(start if bits == 32 else Rounded(start, bits), d, lr=1e-3))


def summary():
    rows = {(r["hidden"], r["seed"], r["method"], r["bits"]): r for r in map(json.loads, open(RESULTS))}  # a re-run replaces its rows
    for hidden in sorted({k[0] for k in rows}, reverse=True):
        seeds = sorted({k[1] for k in rows if k[0] == hidden})
        print(f"\n{hidden} GRU units, held-out R2: mean over {len(seeds)} seed(s) (lowest to highest)")
        print(f"{'bits':>5} {'weights kB':>11}   {'rounded':<24}{'retrained':<24}control")
        for bits in (32, *BITS):
            cells, control = [], []
            for method in ("trained" if bits == 32 else "rounded", "retrained"):
                found = [rows[key] for s in seeds if (key := (hidden, s, method, bits)) in rows]
                scores = [r["held_out"] for r in found]
                control += [r["control"] for r in found]
                cells.append(f"{np.mean(scores):.3f} ({min(scores):.3f} to {max(scores):.3f})" if scores else "-")
            kb = next(r["kB"] for k, r in rows.items() if k[0] == hidden and k[3] == bits)
            print(f"{bits:>5} {kb:>11.1f}   {cells[0]:<24}{cells[1]:<24}{max(control):.2f} or lower")


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:])) if len(sys.argv) > 1 else summary()
