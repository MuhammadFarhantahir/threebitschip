"""Neural-network BCI decoder: a small GRU reads spike counts and predicts hand velocity.

Scored on the same held-out reaches and velocity samples as baseline_decoder.py, so the R2 scores
compare directly. The GRU is causal: each velocity sample is predicted only from spikes that came
before it, as a real-time BCI needs (the baseline's smoothing looks 50 ms ahead).

Not a like-for-like match with the Neural Latents Benchmark table (best model 0.91): those models are
trained without seeing hand velocity and are scored on a hidden test set.

Setup, once (after the baseline's setup):
    uv pip install --python .venv/bin/python torch
Run:
    .venv/bin/python gru_decoder.py data/mc_maze_train.nwb
"""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from torch import nn

from baseline_decoder import BIN, LAG, PAD, fit, flat, load, plot, smooth

# Chosen among six size/dropout settings by their score on the tuning reaches (which come out of the training set).
HIDDEN = 256          # GRU units
DROPOUT = (0.4, 0.4)  # applied to the input spikes, and to the GRU output
PATIENCE = 12         # stop after this many epochs without improvement on the tuning reaches


class Decoder(nn.Module):
    def __init__(self, neurons, bins, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.drop_in, self.drop_out = nn.Dropout(dropout[0]), nn.Dropout(dropout[1])
        self.gru = nn.GRU(neurons, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 2)
        # Velocity sample k lies inside spike bin PAD + LAG/BIN + k. Reading the GRU one bin earlier
        # means it has only seen spikes from before that sample.
        first = PAD + round(LAG / BIN) - 1
        self.steps = slice(first, first + bins)

    def forward(self, counts):  # (reaches, PAD + bins + PAD, neurons) -> (reaches, bins, 2)
        hidden, _ = self.gru(self.drop_in(counts))
        return self.readout(self.drop_out(hidden[:, self.steps]))


def score(model, x, y):
    model.eval()
    with torch.no_grad():
        return r2_score(flat(y.numpy()), flat(model(x).numpy()))


def history(counts, bins, blocks=5, width=10):
    """For each velocity sample, every neuron's spike count in the 5 consecutive 50 ms blocks before it."""
    total = np.pad(np.cumsum(counts, axis=1, dtype=np.float32), ((0, 0), (1, 0), (0, 0)))  # [:, j] = spikes in bins < j
    now = PAD + round(LAG / BIN) + np.arange(bins)  # the spike bin that holds each velocity sample
    return np.concatenate([total[:, now - width * m] - total[:, now - width * (m + 1)] for m in range(blocks)], axis=-1)


def prepare(path, seed=0):
    """Load the recording as tensors and set aside the tuning reaches."""
    torch.manual_seed(seed)
    counts, velocity, split = load(path)
    train, held_out = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    scale = velocity[train].std()  # the network learns velocity in units of its own spread
    x, y = torch.tensor(counts, dtype=torch.float32), torch.tensor(velocity / scale, dtype=torch.float32)

    # Early stopping needs reaches the network does not learn from. They come out of the training
    # reaches, never the held-out ones, so the held-out score stays a fair test.
    rows = torch.tensor(train)[torch.randperm(len(train))]
    tune, rows = rows[: len(rows) * 15 // 100], rows[len(rows) * 15 // 100 :]
    return SimpleNamespace(counts=counts, velocity=velocity, train=train, held_out=held_out, scale=scale,
                           x=x, y=y, rows=rows, tune=tune)


def fit_gru(model, d, lr=2e-3):
    """Train until the score on the tuning reaches stops improving, then put the best weights back."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Starting from the model as given means further training of a trained model can only keep or improve it.
    best, best_weights, stale = score(model, d.x[d.tune], d.y[d.tune]), copy.deepcopy(model.state_dict()), 0
    for epoch in range(300):
        model.train()
        for batch in d.rows[torch.randperm(len(d.rows))].split(64):
            optimizer.zero_grad()
            nn.functional.mse_loss(model(d.x[batch]), d.y[batch]).backward()
            optimizer.step()
        tuned = score(model, d.x[d.tune], d.y[d.tune])
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: R2 on tuning reaches {tuned:.3f}", flush=True)
        if tuned > best:
            best, best_weights, stale = tuned, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale == PATIENCE:
                break
    model.load_state_dict(best_weights)
    return model


def main(path):
    d = prepare(path)
    counts, velocity, train, held_out, scale, x, y, rows, tune = (
        d.counts, d.velocity, d.train, d.held_out, d.scale, d.x, d.y, d.rows, d.tune)
    model = fit_gru(Decoder(x.shape[-1], y.shape[1]), d)

    r2 = score(model, x[held_out], y[held_out])
    # Control: feed the decoder another reach's neural activity. The score should fall to 0 or below
    # (confidently decoding the wrong reach is worse than always guessing the average).
    r2_control = score(model, x[np.roll(held_out, 1)], y[held_out])

    # Two linear references on the same held-out samples: the baseline, and a linear decoder that, like the
    # GRU, sees only earlier spikes, 250 ms of them (a Wiener filter, the classic BCI decoder).
    rates, past = smooth(counts), history(counts, y.shape[1])
    r2_linear = fit(rates[train], velocity[train]).score(flat(rates[held_out]), flat(velocity[held_out]))
    wiener = RidgeCV(alphas=np.logspace(0, 5, 11), cv=5).fit(flat(past[train]), flat(velocity[train]))
    r2_wiener = wiener.score(flat(past[held_out]), flat(velocity[held_out]))

    print(f"{x.shape[-1]} neurons, {len(rows)} training reaches, {len(tune)} tuning reaches, {len(held_out)} held-out reaches")
    print(f"held-out R2, linear baseline:                  {r2_linear:.3f}")
    print(f"held-out R2, linear with 250 ms of history:    {r2_wiener:.3f}")
    print(f"held-out R2, GRU:                              {r2:.3f}")
    print(f"control, GRU given another reach's neurons:    {r2_control:.3f}")
    assert r2_control < 0.1 < r2, "decoder self-check failed"

    model.eval()
    with torch.no_grad():
        decoded = model(x[held_out]).numpy() * scale
    title = f"Hand paths decoded by a GRU from {x.shape[-1]} motor-cortex neurons, held-out reaches (velocity R² = {r2:.2f})"
    print("saved", plot(velocity[held_out], decoded, title, "decoded_reaches_gru.png"))


if __name__ == "__main__":
    main(sys.argv[1])
