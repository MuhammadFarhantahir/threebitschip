"""Baseline BCI decoder: predict hand velocity from motor-cortex spikes with ridge regression.

Data: MC_Maze (Churchland & Kaufman 2022, DANDI:000128, CC-BY-4.0). A monkey reaches through
on-screen mazes while neurons in its motor and premotor cortex are recorded.

Setup, once:
    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python numpy scipy scikit-learn h5py matplotlib
    mkdir data && curl -L -o data/mc_maze_train.nwb \
        https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/
Run:
    .venv/bin/python baseline_decoder.py data/mc_maze_train.nwb
"""
import sys

import h5py
import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV

# Preprocessing follows the Neural Latents Benchmark (Pei et al. 2021, arXiv:2109.04463). On this dataset
# that paper reports R2 0.62 for this method (smoothed spikes + ridge) and 0.91 for its best neural network.
BIN = 0.005               # s, width of a spike-count bin
SMOOTH_SD = 0.050         # s, Gaussian smoothing applied to the spike counts
LAG = 0.100               # s, the hand moves about 100 ms after the neural activity that drives it
WINDOW = (-0.250, 0.450)  # s, analysed time around movement onset
PAD = round(3 * SMOOTH_SD / BIN)  # bins of spikes kept each side of the window, so smoothing has real data at its edges


def load(path):
    """Return spike counts (trials, PAD + bins + PAD, neurons), hand velocity (trials, bins, 2), each trial's split."""
    with h5py.File(path) as f:
        trials = f["intervals/trials"]
        onset, start, stop = (trials[k][:] for k in ("move_onset_time", "start_time", "stop_time"))
        split = trials["split"][:].astype(str)
        spikes = np.split(f["units/spike_times"][:], f["units/spike_times_index"][:-1])
        vel_time, vel = (f["processing/behavior/hand_vel"][k][:] for k in ("timestamps", "data"))

    first, last = round(WINDOW[0] / BIN), round(WINDOW[1] / BIN)
    edges = onset[:, None] + np.arange(first - PAD, last + PAD + 1) * BIN  # (trials, bin edges)
    counts = np.stack([np.diff(np.searchsorted(np.sort(s), edges), axis=1) for s in spikes], axis=-1)

    centers = onset[:, None] + (np.arange(first, last) + 0.5) * BIN + LAG  # (trials, bins)
    velocity = np.stack([np.interp(centers, vel_time, v) for v in vel.T], axis=-1)

    # Recording only covers the trials, so drop any trial whose window runs past its own recording.
    ok = (edges[:, 0] >= start) & (np.maximum(edges[:, -1], centers[:, -1]) <= stop)
    ok &= np.isfinite(velocity).all(axis=(1, 2))
    return counts[ok], velocity[ok], split[ok]


def smooth(counts):
    """Spike counts -> firing rates (trials, bins, neurons): Gaussian smoothing, then the padding is cut off."""
    # ponytail: acausal smoothing (uses future spikes) is fine offline; a real-time BCI needs a causal filter
    return gaussian_filter1d(counts.astype(float), SMOOTH_SD / BIN, axis=1, truncate=3.0)[:, PAD:-PAD]


def flat(x):
    """(trials, bins, features) -> (trials * bins, features)."""
    return x.reshape(-1, x.shape[-1])


def fit(rates, velocity):
    # cv=5 picks the penalty on contiguous blocks of reaches. RidgeCV's default (leave one bin out)
    # sees the neighbouring bins of the same reach, picks too weak a penalty, and decodes worse.
    return RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5).fit(flat(rates), flat(velocity))


def plot(actual, decoded, title, path="decoded_reaches.png"):
    """Draw 8 held-out reaches spread evenly through the session, as hand paths (velocity summed over time)."""
    ink, ink2, muted, surface = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
    plt.rcParams.update({
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.labelcolor": ink2,
        "xtick.color": muted, "ytick.color": muted, "xtick.major.size": 0, "ytick.major.size": 0,
    })
    picks = np.linspace(0, len(actual) - 1, 8).astype(int)
    paths = [np.cumsum(v[picks], axis=1) * BIN for v in (actual, decoded)]  # mm/s summed over time -> mm from start
    lim = 1.08 * max(np.abs(p).max() for p in paths)  # one square window for every panel, centred on the start
    fig, axes = plt.subplots(2, 4, figsize=(10, 6), sharex=True, sharey=True)
    for k, ax in enumerate(axes.flat):
        for p, color, label in zip(paths, ("#2a78d6", "#eb6834"), ("Actual hand path", "Decoded from neurons")):
            ax.plot(*p[k].T, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", label=label)
        ax.plot(0, 0, "o", ms=7, color=ink2, mec=surface, mew=1.5, label="Start")
        ax.set(aspect="equal", xlim=(-lim, lim), ylim=(-lim, lim))
    for axis in (axes[0, 0].xaxis, axes[0, 0].yaxis):  # axes are shared, so this sets every panel
        axis.set_major_locator(MultipleLocator(100))    # same spacing both ways: square grid cells
    fig.suptitle(title, x=0.06, y=0.97, ha="left", fontsize=13, fontweight="bold", color=ink)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.05, 0.93),
               ncols=3, frameon=False, labelcolor=ink2)
    fig.supxlabel("x (mm)", color=ink2, fontsize=9)
    fig.supylabel("y (mm)", color=ink2, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.09, wspace=0.08, hspace=0.12)
    fig.savefig(path, dpi=150)
    return path


def main(path):
    counts, velocity, split = load(path)
    rates = smooth(counts)
    train, held_out = split == "train", split == "val"  # the dataset authors' own split, by trial
    X_test, y_test = flat(rates[held_out]), flat(velocity[held_out])

    model = fit(rates[train], velocity[train])
    r2 = model.score(X_test, y_test)

    # Control: train on mismatched (shuffled) reaches. Only the average reach profile can still be learned,
    # so the score should fall to about 0; a high score here would mean the pipeline leaks.
    shuffled = np.random.default_rng(0).permutation(train.sum())
    r2_control = fit(rates[train], velocity[train][shuffled]).score(X_test, y_test)

    print(f"{rates.shape[-1]} neurons, {train.sum()} training reaches, {held_out.sum()} held-out reaches")
    print(f"held-out R2 (hand velocity):     {r2:.3f}")
    print(f"shuffled-trials control R2:      {r2_control:.3f}")
    assert r2_control < 0.1 < r2, "decoder self-check failed"

    decoded = model.predict(X_test).reshape(velocity[held_out].shape)
    title = f"Hand paths decoded from {rates.shape[-1]} motor-cortex neurons, held-out reaches (velocity R² = {r2:.2f})"
    print("saved", plot(velocity[held_out], decoded, title))


if __name__ == "__main__":
    main(sys.argv[1])
