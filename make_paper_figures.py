"""Generate the manuscript figures from the aggregate experimental outputs.

Outputs (saved in figures/):
  1. fig1_ccep_validation.png      — 74-patient human CCEP validation: fMRI-vs-causal trade-off + 5-fold CCEP calibration
  2. fig2_closedloop_control.png   — 50-person closed-loop stimulation study + universal transfer & nonlinear Hopf
  3. fig3_falcon_precision_drift.png — FALCON H1 real human implant: 2–32 bit accuracy, memory, and 6-week drift recovery

Run:
    .venv/bin/python make_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).with_name("figures")
FIG_DIR.mkdir(exist_ok=True)

# Editorial AI-lab palette matching threebits.si
BG = "#f9f8f4"
INK = "#1a1916"
MUTED = "#6b675b"
LINE = "#cbc6b8"
ACCENT = "#c8502a"
BLUE = "#2f5d8a"
TEAL = "#2b7a69"
GRAY = "#9a9589"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 220,
    "savefig.facecolor": BG,
})


def fig1_ccep():
    v = json.loads(Path("validate_simulator_ccep.json").read_text())
    c = json.loads(Path("causal_brain_simulator.json").read_text())
    tf = v["fmri_vs_ccep_tradeoff"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel A: Resting fMRI r vs Causal Human CCEP rho trade-off
    fmri_r = [row["fmri_r"] for row in tf]
    ccep_rho = [row["ccep_rho"] for row in tf]
    ccep_part = [row["ccep_partial_rho"] for row in tf]
    hub_rho = [row["hub_rho"] for row in tf]

    ax1.plot(fmri_r, ccep_rho, "-o", color=ACCENT, lw=2.3, ms=6.5, label="Whole-brain CCEP (Spearman ρ)")
    ax1.plot(fmri_r, hub_rho, "--s", color=BLUE, lw=1.9, ms=5.5, label="Prefrontal hub LH_Cont_PFCl_1 (ρ)")
    ax1.plot(fmri_r, ccep_part, ":^", color=TEAL, lw=1.9, ms=5.5, label="Partial ρ | Euclidean distance")
    ax1.axhline(0, color=LINE, lw=1)
    ax1.annotate("Untuned Hopf\n(1 param g)", (fmri_r[0], ccep_rho[0]), xytext=(14, -18),
                 textcoords="offset points", fontsize=8.5, color=ACCENT, fontweight="bold")
    ax1.annotate("Free-tuned to fMRI\n(r = 0.89)", (fmri_r[-1], ccep_rho[-1]), xytext=(-58, 56),
                 textcoords="offset points", fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax1.set_xlabel("Observational resting-state fMRI fit (Pearson r)")
    ax1.set_ylabel("Real human electrical stimulation accuracy (ρ)")
    ax1.set_title("A. Resting-fMRI tuning degrades causal accuracy (74 patients)")
    ax1.set_ylim(-0.08, 0.72)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel B: 5-fold held-out human CCEP prediction
    cv = c["cv_ccep"]
    ns_pred = v["results"]["non_SOZ_only_n>=10"]["predictors"]
    bars = [
        ("Rewired\ncontrol", cv["CCEP-calibrated rewired control (unseen pairs)"]["spearman"],
         cv["CCEP-calibrated rewired control (unseen pairs)"]["partial_spearman_given_dist"], GRAY),
        ("fMRI-tuned\nHopf", ns_pred["Tuned Hopf simulator (G = K^-1)"]["spearman"],
         ns_pred["Tuned Hopf simulator (G = K^-1)"]["partial_spearman_given_dist"], GRAY),
        ("Resting\nfMRI", ns_pred["HCP resting-state fMRI alone"]["spearman"],
         ns_pred["HCP resting-state fMRI alone"]["partial_spearman_given_dist"], GRAY),
        ("MRI\nwiring", cv["HCP diffusion MRI wiring alone"]["spearman"],
         cv["HCP diffusion MRI wiring alone"]["partial_spearman_given_dist"], BLUE),
        ("Untuned\nHopf", cv["Untuned Hopf (1 param g)"]["spearman"],
         cv["Untuned Hopf (1 param g)"]["partial_spearman_given_dist"], TEAL),
        ("CCEP-Hopf\n(regions)", cv["CCEP-calibrated Hopf (5-fold unseen stim regions)"]["spearman"],
         cv["CCEP-calibrated Hopf (5-fold unseen stim regions)"]["partial_spearman_given_dist"], "#d97a4a"),
        ("CCEP-Hopf\n(pairs)", cv["CCEP-calibrated Hopf (5-fold unseen pairs)"]["spearman"],
         cv["CCEP-calibrated Hopf (5-fold unseen pairs)"]["partial_spearman_given_dist"], ACCENT),
    ]
    x = np.arange(len(bars))
    w = 0.38
    b1 = ax2.bar(x - w / 2, [b[1] for b in bars], w, color=[b[3] for b in bars], label="Raw Spearman ρ")
    b2 = ax2.bar(x + w / 2, [b[2] for b in bars], w, color=[b[3] for b in bars], alpha=0.42, edgecolor=INK, lw=0.6,
                 label="Partial ρ | distance")
    for rect in b1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
    for rect in b2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=7.2, color=MUTED)
    ax2.set_xticks(x)
    ax2.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax2.set_ylabel("Held-out correlation on 3,219 non-SOZ pairs")
    ax2.set_ylim(0, 0.82)
    ax2.set_title("B. Causal calibration beats anatomy & fMRI (5-fold CV)")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout(w_pad=2.5)
    out = FIG_DIR / "fig1_ccep_validation.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig2_closedloop():
    study = Path("stimulation_study.jsonl")
    if not study.exists():
        study = Path("stimulation_study_figure.jsonl")
    rows = [json.loads(line) for line in study.read_text().splitlines()]
    fu = json.loads(Path("stimulation_followups.json").read_text())
    people = sorted({r["person"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel A: Resting hub connectivity vs closed-loop hub gain across 50 people
    rest = np.array([next(r["rest_reach"] for r in rows if r["person"] == p) for p in people])
    for name, col, marker, label in (
        ("linear", ACCENT, "o", "Closed-loop linear (3 seeds, r = −0.87)"),
        ("network", BLUE, "s", "Closed-loop MLP (3 seeds, r = −0.72)"),
        ("fixed rhythm", GRAY, "^", "Open-loop fixed rhythm (r = −0.20)"),
    ):
        g = np.array([np.mean([r["reach_change"] for r in rows if r["person"] == p and r["stimulator"] == name]) for p in people])
        ax1.scatter(rest, g, c=col, marker=marker, s=34, alpha=0.82, label=label, edgecolors="none")
        coef = np.polyfit(rest, g, 1)
        xs = np.linspace(rest.min(), rest.max(), 50)
        ax1.plot(xs, np.polyval(coef, xs), color=col, lw=1.6, alpha=0.7)
    ax1.axhline(0, color=LINE, lw=1)
    ax1.set_xlabel("Baseline resting connectivity of LH_Cont_PFCl_1")
    ax1.set_ylabel("Hub connectivity gain Δ (64 unseen runs)")
    ax1.set_title("A. Closed-loop control helps weaker prefrontal hubs (n = 50)")
    ax1.legend(frameon=False, fontsize=8.2, loc="upper right")

    # Panel B: Universal transfer + supercritical nonlinear Hopf comparison
    lt = fu["linear_transfer"]
    nl = fu["nonlinear_hopf"]
    chosen = fu["chosen_people"]
    strong2 = chosen[-2:]

    cats = [
        ("Perm.\nctrl", np.mean([r["perm_reach_change"] for r in lt]), GRAY),
        ("Next\nperson", np.mean([r["next_reach_change"] for r in lt]), BLUE),
        ("LOO\nuniv.", np.mean([r["loo_reach_change"] for r in lt]), TEAL),
        ("Personal\nlinear", np.mean([r["own_reach_change"] for r in lt]), ACCENT),
        ("NL 6p\nlinear", np.mean([r["reach_change"] for r in nl if r["stimulator"] == "linear"]), "#3e70a3"),
        ("NL 6p\nresid.", np.mean([r["reach_change"] for r in nl if r["stimulator"] == "residual"]), "#d97a4a"),
        ("Top 2\nlinear", np.mean([r["reach_change"] for r in nl if r["stimulator"] == "linear" and r["person"] in strong2]), "#6d8fb5"),
        ("Top 2\nresid.", np.mean([r["reach_change"] for r in nl if r["stimulator"] == "residual" and r["person"] in strong2]), ACCENT),
    ]
    x = np.arange(len(cats))
    vals = [c[1] for c in cats]
    cols = [c[2] for c in cats]
    bs = ax2.bar(x, vals, width=0.64, color=cols)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.axvline(3.5, color=LINE, lw=1, ls="--")
    for rect, val in zip(bs, vals):
        va = "bottom" if val >= 0 else "top"
        off = 0.005 if val >= 0 else -0.006
        ax2.text(rect.get_x() + rect.get_width() / 2, val + off, f"{val:+.3f}", ha="center", va=va, fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[0] for c in cats], fontsize=7.8)
    ax2.set_ylim(-0.175, 0.135)
    ax2.set_ylabel("Mean hub connectivity gain Δ")
    ax2.set_title("B. Cross-brain transfer & nonlinear Hopf (α = 2)")
    ax2.text(1.5, 0.118, "Linear transfer (50p)", ha="center", fontsize=8.2, color=MUTED)
    ax2.text(5.5, 0.118, "Supercritical NL Hopf", ha="center", fontsize=8.2, color=MUTED)

    fig.tight_layout(w_pad=2.5)
    out = FIG_DIR / "fig2_closedloop_control.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig3_falcon():
    p_base = Path("falcon_precision_drift.jsonl")
    p_anch = Path("falcon_anchored_recal.jsonl")
    if not p_base.exists():
        return
    base = [json.loads(l) for l in open(p_base)]
    anch = [json.loads(l) for l in open(p_anch)] if p_anch.exists() else []
    bits = [32, 8, 4, 3, 2]
    wiener = next(r for r in base if r["model"] == "wiener_300ms")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Panel A: Held-in minival R² vs weight memory (log scale)
    kb = [next(r["kB"] for r in base if r["model"] == "gru_128" and r["bits"] == b) for b in bits]
    mv = [np.mean([r["minival_mean"] for r in base if r["model"] == "gru_128" and r["bits"] == b]) for b in bits]
    ax1.plot(kb, mv, "-o", color=ACCENT, lw=2.3, ms=7.5, label="Causal 128-unit GRU (QAT)")
    offsets = {32: (0, -28), 8: (0, -28), 4: (6, 12), 3: (-4, -28), 2: (6, 12)}
    for b, xk, ym in zip(bits, kb, mv):
        ax1.annotate(f"{b}-bit\n({xk:.0f} kB)", (xk, ym), xytext=offsets[b],
                     textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax1.scatter([wiener["kB"]], [wiener["minival_mean"]], color=GRAY, s=70, marker="D", zorder=5,
                label=f"Causal 300 ms Wiener ({wiener['kB']:.0f} kB)")
    ax1.set_xscale("log")
    ax1.set_xticks([30, 50, 100, 200, 500])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.set_xlim(22, 650)
    ax1.set_ylim(0.18, 0.94)
    ax1.set_xlabel("Weight memory footprint (kB, log scale)")
    ax1.set_ylabel("Held-in minival diagnostic R² (6 days)")
    ax1.set_title("A. Held-in diagnostic scores by precision")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    # Panel B: Later days (weeks 4–6) drift & 48-second recalibration
    x = np.arange(len(bits))
    zero = [np.mean([r["heldout_zeroshot_mean"] for r in base if r["model"] == "gru_128" and r["bits"] == b]) for b in bits]
    full_u = [np.mean([r["heldout_recal_full_mean"] for r in base if r["model"] == "gru_128" and r["bits"] == b]) for b in bits]
    if anch and len(anch) >= 15:
        unsup = [np.mean([r["unsup_align_mean"] for r in anch if r["bits"] == b]) for b in bits]
        adp = [np.mean([r["anchored_adapter_mean"] for r in anch if r["bits"] == b]) for b in bits]
        w = 0.20
        ax2.bar(x - 1.5 * w, zero, w, color=GRAY, label="Zero-shot (no recal)")
        ax2.bar(x - 0.5 * w, unsup, w, color=BLUE, label="Unsupervised mean-align (0 labels)")
        ax2.bar(x + 0.5 * w, full_u, w, color="#d97a4a", label="48 s full QAT recal")
        ax2.bar(x + 1.5 * w, adp, w, color=ACCENT, label="48 s anchored frozen-core adapter (1%)")
    else:
        adp = [np.mean([r["heldout_recal_adapter_mean"] for r in base if r["model"] == "gru_128" and r["bits"] == b]) for b in bits]
        w = 0.25
        ax2.bar(x - w, zero, w, color=GRAY, label="Zero-shot")
        ax2.bar(x, full_u, w, color=BLUE, label="48 s full QAT recal")
        ax2.bar(x + w, adp, w, color=ACCENT, label="48 s frozen-core adapter")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b}-bit\n({k:.0f} kB)" for b, k in zip(bits, kb)], fontsize=8.5)
    ax2.set_ylim(-0.045, 0.365)
    ax2.set_ylabel("Mean R² across 7 later days (weeks 4–6)")
    ax2.set_title("B. Multi-week human drift & 48-second recalibration")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout(w_pad=2.5)
    out = FIG_DIR / "fig3_falcon_precision_drift.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig4_safe_tmseeg_memory():
    tms = json.loads(Path("validate_simulator_tmseeg_expanded.json").read_text())["results"]
    safe = json.loads(Path("safe_closedloop_controller.json").read_text())
    mem = json.loads(Path("human_memory_policy.json").read_text())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.2, 4.4), gridspec_kw={"width_ratios": [1.05, 1.0, 1.1]})

    # Panel A: participant means for the two-site TMS–EEG evaluation.
    comparisons = [
        ("Wrong-site control", "Wrong-site negative control (permuted targets)", GRAY),
        ("Coil-distance null", "Coil-distance-only null (no connectome)", GRAY),
        ("Rewired control", "Rewired white-matter control", GRAY),
        ("fMRI-tuned Hopf", "Resting-fMRI-tuned Hopf (observational)", "#b56a5a"),
        ("Anatomy-based Hopf", "Untuned anatomy-Hopf (1 param g on dMRI)", TEAL),
        ("CCEP-calibrated Hopf", "CCEP-calibrated Hopf (invasive -> non-invasive zero-shot)", ACCENT),
    ]
    tms_rows = []
    for label, key, color in comparisons:
        summary = tms[key]
        count, n = summary["indiv_positive_count"], summary["indiv_n"]
        tms_rows.append((f"{label}\n({count}/{n} positive)", summary["indiv_mean_partial_given_dist"], color))
    y = np.arange(len(tms_rows))
    vals = [r[1] for r in tms_rows]
    cols = [r[2] for r in tms_rows]
    bars = ax1.barh(y, vals, height=0.62, color=cols)
    ax1.axvline(0, color=INK, lw=0.8)
    for bar, v in zip(bars, vals):
        ha = "left" if v >= 0 else "right"
        off = 0.015 if v >= 0 else -0.015
        ax1.text(v + off, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}", va="center", ha=ha, fontsize=8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r[0] for r in tms_rows], fontsize=8)
    ax1.set_xlim(-0.24, 0.38)
    ax1.set_xlabel("Mean participant-level partial ρ")
    ax1.set_title("A. Two-site TMS–EEG (46 adults)")

    # Panel B: Certified-Safe BRL Lyapunov Control on CCEP-Calibrated Human Brain (5 seeds)
    b1 = safe["bench1_ccep_brain"]["controllers"]
    c_rows = [
        ("Open-loop\nrhythm", np.mean([r["reach_change"] for r in b1["fixed_rhythm"]["runs"]]), GRAY, "16% E"),
        ("Unconstr.\nlinear", np.mean([r["reach_change"] for r in b1["unconstrained_linear"]["runs"]]), BLUE, "105% E"),
        ("Unconstr.\nresidual", np.mean([r["reach_change"] for r in b1["unconstrained_residual"]["runs"]]), "#b56a5a", "3783% E!"),
        ("SafeLinear\n(5/5 BRL)", np.mean([r["reach_change"] for r in b1["safe_linear"]["runs"]]), TEAL, "93% E"),
        ("SafeResidual\n(5/5 BRL)", np.mean([r["reach_change"] for r in b1["safe_residual"]["runs"]]), ACCENT, "77% E"),
    ]
    x = np.arange(len(c_rows))
    cv = [r[1] for r in c_rows]
    bs = ax2.bar(x, cv, width=0.60, color=[r[2] for r in c_rows])
    ax2.axhline(0, color=INK, lw=0.8)
    for rect, (_, v, _, note) in zip(bs, c_rows):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:+.3f}\n({note})",
                 ha="center", va="bottom", fontsize=7.2, color=ACCENT if "!" in note else INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r[0] for r in c_rows], fontsize=7.6)
    ax2.set_ylim(0, 0.083)
    ax2.set_ylabel("Prefrontal hub gain Δ (CCEP brain)")
    ax2.set_title("B. Lyapunov certificates for simulations")
    ax2.text(0.03, 0.94, "160/160 certified, 0 threshold violations", transform=ax2.transAxes,
             fontsize=7.6, color=TEAL, fontweight="bold", va="top")

    # Panel C: 15-Patient Real Human Memory Stimulation (4,500 words, 1,142 stims)
    pts = sorted(mem["patients"], key=lambda r: r["recall_diff_low_pp"])
    grp_col = {"LTC": TEAL, "Frontal": BLUE, "Hippocampus": ACCENT, "Insula": "#8f2d1a"}
    yp = np.arange(len(pts))
    pv = [p["recall_diff_low_pp"] for p in pts]
    pc = [grp_col.get(p["site_group"], GRAY) for p in pts]
    ax3.barh(yp, pv, height=0.66, color=pc)
    ax3.axvline(0, color=INK, lw=0.8)
    ax3.set_yticks(yp)
    ax3.set_yticklabels([f"Case {index:02d}" for index in range(1, len(pts) + 1)], fontsize=7.2)
    ax3.set_xlim(-12.5, 10.5)
    ax3.set_xlabel("Matched low-state recall Δ (pp)")
    ax3.set_title("C. Retrospective recall (15 patients)")
    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(color=TEAL, label="LTC (n=10, +0.0 pp)"),
               Patch(color=BLUE, label="Frontal (n=2)"),
               Patch(color=ACCENT, label="Hipp CA1/DG (n=2, −7.8 pp)"),
               Patch(color="#8f2d1a", label="Insula (n=1, −9.9 pp)")]
    ax3.legend(handles=handles, frameon=False, fontsize=7.2, loc="upper left")

    fig.tight_layout(w_pad=2.2)
    out = FIG_DIR / "fig4_safe_tmseeg_memory.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def fig5_prosthetic_cortex():
    b = json.loads(Path("brain_ai_coprocessor.json").read_text())
    h = json.loads(Path("human_intelligence_wm_eeg.json").read_text())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.5), gridspec_kw={"width_ratios": [1.15, 1.05, 1.0]})

    # Panel A: Capacity Curve (K = 2..16) + 117 real healthy adults empirical cliff
    ks = [2, 4, 6, 8, 12, 16]
    b1 = b["exp_b1_capacity_curve"]
    curves = [
        ("coprocessor", "4-bit coupled external memory (50 ms)", ACCENT, "-o", 2.4),
        ("state_steer", "State-Steering Only (no ext. mem.)", TEAL, "--s", 1.9),
        ("brain_alone", "Simulated recurrent model alone", BLUE, "-^", 2.0),
        ("screen_ai_40b", "Low-rate interface (200 ms)", "#8b6d5c", ":d", 1.7),
        ("sham_control", "Trial-permuted control", GRAY, ":x", 1.5),
    ]
    for key, lbl, col, fmt, lw in curves:
        ys = [b1[key][str(k)]["acc30"] * 100 for k in ks]
        ax1.plot(ks, ys, fmt, color=col, lw=lw, ms=5.5, label=lbl)
    # Overlay empirical 117 healthy adults from OpenNeuro ds008037!
    a1 = h["part_a1_g_factor_123adults"]
    ax1.plot([2, 4, 6], [a1["acc30_2"] * 100, a1["acc30_4"] * 100, a1["acc30_6"] * 100],
             "-*", color=INK, lw=2.0, ms=9, zorder=6, label="117 Healthy Humans (ds008037)")
    ax1.axhline(16.7, color=LINE, lw=1, ls="--")
    ax1.text(16.0, 12.0, "chance (16.7%)", ha="right", fontsize=7.4, color=MUTED)
    ax1.set_xticks(ks)
    ax1.set_ylim(5, 108)
    ax1.set_xlabel("Working-memory items held simultaneously (K)")
    ax1.set_ylabel("Recall precision accuracy (% within 30°)")
    ax1.set_title("A. Observed recall and simulated capacity")
    ax1.legend(frameon=False, fontsize=7.2, loc="center right")

    # Panel B: 7 x 5 Bandwidth x Latency Phase Transition Heatmap at K = 12
    grid = b["exp_b1b_phase_diagram_k12"]
    bws = ["20_bps", "40_bps", "80_bps", "160_bps", "320_bps", "640_bps", "960_bps"]
    bw_lbls = ["20", "40", "80", "160", "320", "640", "960"]
    lats = ["20ms", "50ms", "100ms", "200ms", "250ms"]
    mat = np.array([[grid[bw][lt]["acc30"] * 100 for lt in lats] for bw in bws])
    im = ax2.imshow(mat, origin="lower", aspect="auto", cmap="RdYlGn", vmin=15, vmax=100)
    for r in range(len(bws)):
        for c in range(len(lats)):
            val = mat[r, c]
            tc = "#ffffff" if (val < 30 or val > 85) else INK
            ax2.text(c, r, f"{val:.0f}%", ha="center", va="center", fontsize=7.8, fontweight="bold", color=tc)
    ax2.axhline(1.5, color="#ffffff", lw=1.2, ls="--")
    ax2.set_xticks(range(len(lats)))
    ax2.set_xticklabels(lats, fontsize=8.2)
    ax2.set_yticks(range(len(bws)))
    ax2.set_yticklabels(bw_lbls, fontsize=8.2)
    ax2.set_xlabel("Modeled feedback delay")
    ax2.set_ylabel("Nominal read cap (bits/s)")
    ax2.set_title("B. Simulated recall at K = 12")

    # Panel C: Level-4 Prosthetic Cortex Strict Outperformance (K=12 + Private Human Rule)
    b2 = b["exp_b2_prosthetic_cortex"]
    cka = b["cortical_cka"]
    bars = [
        ("Recurrent\nalone", b2["human_alone"]["acc30"] * 100, BLUE),
        ("State\nsteer", b2["state_steer"]["acc30"] * 100, TEAL),
        ("Low-rate\ninterface", b2["screen_ai_40b"]["acc30"] * 100, "#8b6d5c"),
        ("Permuted\nread", b2["sham_control"]["acc30"] * 100, GRAY),
        ("Digital\nalone", b2["ai_standalone"]["acc30"] * 100, "#5c6e8b"),
        ("Coupled\nmodel", b2["prosthetic_ctx"]["acc30"] * 100, ACCENT),
    ]
    x = np.arange(len(bars))
    ys = [r[1] for r in bars]
    bs = ax3.bar(x, ys, width=0.62, color=[r[2] for r in bars])
    ax3.axhline(16.7, color=LINE, lw=1, ls="--")
    for rect, val in zip(bs, ys):
        ax3.text(rect.get_x() + rect.get_width() / 2, val + 2.0, f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold" if val > 70 else "normal", color=ACCENT if val > 70 else INK)
    ax3.set_xticks(x)
    ax3.set_xticklabels([r[0] for r in bars], fontsize=7.5)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("Accuracy (% within 30°, K=12 + private rule)")
    ax3.set_title("C. Simulated private-rule task")
    ax3.text(0.03, 0.94, f"CKA coherence: {cka['alone_vs_alone']:.2f} → {cka['prost_vs_prost']:.2f} (10.8×)",
             transform=ax3.transAxes, fontsize=7.4, color=TEAL, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.0)
    out = FIG_DIR / "fig5_prosthetic_cortex.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig1_ccep()
    fig2_closedloop()
    fig3_falcon()
    fig4_safe_tmseeg_memory()
    fig5_prosthetic_cortex()
