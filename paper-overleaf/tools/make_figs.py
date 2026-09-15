#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2: (a) Fieller conditional coverage and (b) report rate over the 63
distribution x mechanism cells; (c) abstention-penalised score S_lambda as a
function of the abstention price lambda for the four methods on the frozen
holdout.  S_lambda is affine in lambda, S_lambda = a + (1 - report) lambda, so
the two stored values at lambda = 0.1 and lambda = 1.0 determine each line."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
                     "pdf.fonttype": 42, "font.family": "serif"})

x4 = json.load(open(os.path.join(DATA, "x4_landscape.json")))
r2 = json.load(open(os.path.join(DATA, "r2_holdout.json")))

DIST_ORDER = ["LogNormal CV 0.36", "Gamma CV 0.36", "Pareto CV 0.36", "Two-point CV 0.36",
              "LogNormal CV 2.91", "Gamma CV 2.91", "Pareto CV 2.91", "Two-point CV 2.91", "Pareto alpha 1.5"]
DIST_SHORT = ["LogN 0.36", "Gamma 0.36", "Pareto 0.36", "2-pt 0.36",
              "LogN 2.91", "Gamma 2.91", "Pareto 2.91", "2-pt 2.91", r"Pareto $\alpha$1.5"]
MECH_ORDER = ["uniform", "ucb_fieller", "ucb_delta", "thompson_ratio", "eps_greedy_ratio", "greedy_ratio", "kube_knapsack"]
MECH_SHORT = ["unif.", "UCB-F", "UCB-D", "TS", r"$\epsilon$-greedy", "greedy", "KUBE"]

cond = np.full((9, 7), np.nan)
rep = np.full((9, 7), np.nan)
fail = np.zeros((9, 7), bool)
for c in x4["cells"]:
    i = DIST_ORDER.index(c["cost_dist"])
    j = MECH_ORDER.index(c["mechanism"])
    cond[i, j] = c["conditional"][0]
    rep[i, j] = c["report"][0]
    fail[i, j] = c["conditional"][2] < 0.90

fig, axes = plt.subplots(1, 2, figsize=(7.1, 1.34), gridspec_kw={"wspace": 0.42})

for ax, mat, title, cmap, vmin, vmax in (
        (axes[0], cond, "(a) Fieller conditional coverage", "viridis", 0.84, 0.99),
        (axes[1], rep, "(b) Fieller report rate", "magma", 0.0, 1.0)):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(7))
    ax.set_xticklabels(MECH_SHORT, rotation=45, ha="right")
    ax.set_yticks(range(9))
    ax.set_yticklabels(DIST_SHORT)
    ax.set_title(title, loc="left")
    for i in range(9):
        for j in range(7):
            v = mat[i, j]
            col = "white" if (v - vmin) / (vmax - vmin) < 0.55 else "black"
            ax.text(j, i, ("%.2f" % v).replace("0.", ".", 1) if v < 1 else "1.0", ha="center", va="center", fontsize=5.6, color=col)
    if mat is cond:
        for i in range(9):
            for j in range(7):
                if fail[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, shrink=0.9)
    cb.ax.tick_params(labelsize=5.5)
axes[0].axhline(0.90, color="none")

fig.savefig(os.path.join(FIGS, "fig_landscape.pdf"), bbox_inches="tight", pad_inches=0.02)
print("figs/fig_landscape.pdf written")


# ============================================================================ Figure 2: budget sweep + alpha calibration
r4 = json.load(open(os.path.join(DATA, "r4_policy.json")))
RUNS = r4["runs"]
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.25), gridspec_kw={"wspace": 0.5})
ax = axes[0]
mults = [5, 10, 30, 100]
def rep(pol, m):
    blk = "A" if m == 30 else "B"
    return RUNS["%s|%s|env1|%dx" % (blk, pol, m)]["methods"]["Fieller"]["report"]
for pol, col, ls, lab in (("ucb_fieller_delta", "#0072B2", "-", "UCB-F, delta fallback"), ("uniform", "#D55E00", "--", "uniform")):
    y = np.array([rep(pol, m) for m in mults])
    ax.plot(mults, y[:, 0], color=col, ls=ls, marker="o", ms=2.5, lw=1.1, label=lab)
    ax.fill_between(mults, y[:, 1], y[:, 2], color=col, alpha=0.2, lw=0)
pred = RUNS["A|ucb_fieller_delta|env1|30x"]["pred_report_upper"]
ax.axhline(pred, color="gray", lw=0.7, ls=":")
ax.text(5.2, pred - 0.11, r"$1-K^{-1}\sum_a(1-\mu_a)^2$", fontsize=5.5, color="gray")
ax.set_xscale("log"); ax.set_xticks(mults); ax.set_xticklabels([str(m) + r"$\times$" for m in mults])
ax.set_ylim(0, 1.05); ax.set_xlabel("budget multiplier"); ax.set_ylabel("Fieller report rate")
ax.set_title("(a) report rate vs budget", loc="left")
ax.legend(frameon=False, loc="center right", fontsize=5.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax = axes[1]
AL = r4["alpha_sweep"]
levels = ["0.20", "0.10", "0.05", "0.01"]
nom = [1 - float(a) for a in levels]
for pol, meth, key, col, ls, lab in (("ucb_fieller", "Fieller", "conditional", "#0072B2", "-", "Fieller, adaptive"),
                                     ("ucb_fieller", "raw delta", "common_F", "#D55E00", "--", "delta on Fieller set, adaptive"),
                                     ("uniform", "Fieller", "conditional", "#009E73", "-.", "Fieller, uniform")):
    y = np.array([AL[pol][a][meth][key] for a in levels])
    ax.plot(nom, y[:, 0], color=col, ls=ls, marker="o", ms=2.5, lw=1.1, label=lab)
    ax.fill_between(nom, y[:, 1], y[:, 2], color=col, alpha=0.2, lw=0)
ax.plot([0.78, 1.0], [0.78, 1.0], color="black", lw=0.6)
ax.set_xlabel(r"nominal level $1-\alpha$"); ax.set_ylabel("conditional coverage")
ax.set_title("(b) calibration", loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=5.0, handlelength=1.8)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.savefig(os.path.join(FIGS, "fig_budget_alpha.pdf"), bbox_inches="tight", pad_inches=0.02)
print("figs/fig_budget_alpha.pdf written")


# ---------------------------------------------------------------------------
# Figure 1 (fig_sets.pdf): the existence criterion of Proposition 1 and the
# four sets returned on a zero-outcome arm.  Panel (b) endpoints come from
# ratio_ci.py on two seeded cost pairs of the benchmark distribution, so the
# picture is produced by the same code as the tables.
# ---------------------------------------------------------------------------
import sys  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "experiments")))
import ratio_ci as RC  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

ALPHA = 0.10
SIGMA = 0.35
THETA = float(np.median([p["rho"] for p in json.load(
    open(os.path.join(DATA, "r2_recomputed.json")))["adaptive"]["per_arm_fieller"]]))
TCRIT = float(tdist.ppf(1.0 - ALPHA / 2.0, 1))
THR = np.sqrt(2.0) / TCRIT            # CV-hat threshold of condition (i) at n = 2


def unit(costs):
    """Moment dict for one arm-trajectory unit with two zero outcomes."""
    c = np.asarray(costs, float).reshape(1, 2)
    S = {"n": np.array([[2.0]]), "sr": np.array([[0.0]]), "sc": c.sum(1, keepdims=True),
         "sr2": np.array([[0.0]]), "sc2": (c ** 2).sum(1, keepdims=True), "src": np.array([[0.0]])}
    return RC.moments(S)


rng = np.random.default_rng(7)
draws = rng.lognormal(0.0, SIGMA, size=(4000, 2))
cvhat = draws.std(1, ddof=1) / draws.mean(1)
tight = draws[cvhat < 0.6 * THR][0]   # condition (i) holds
wide = draws[cvhat > 1.6 * THR][0]    # condition (i) fails

fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.24), gridspec_kw={"width_ratios": [1.0, 1.28], "wspace": 0.42})

# (a) existence regions of Proposition 1
ax = axes[0]
ax.axvspan(0, THR, ymin=0.5, ymax=1.0, color="#cfe3f5")
ax.axvspan(THR, 0.65, ymin=0.5, ymax=1.0, color="#f6ddc9")
ax.axvspan(0, THR, ymin=0.0, ymax=0.5, color="#ead9ef")
ax.axvspan(THR, 0.65, ymin=0.0, ymax=0.5, color="#f6ddc9")
ax.axvline(THR, color="k", lw=0.8)
ax.axhline(0.5, color="k", lw=0.8)
ax.text(THR / 2, 0.75, r"$[l,u]$", ha="center", va="center", fontsize=8)
ax.text((THR + 0.65) / 2, 0.75, r"two rays / $\mathbb{R}$", ha="center", va="center", fontsize=6.3)
ax.text(THR / 2, 0.25, r"$\{0\}$", ha="center", va="center", fontsize=8)
ax.text((THR + 0.65) / 2, 0.25, r"$\mathbb{R}$", ha="center", va="center", fontsize=8)
ax.set_xlim(0, 0.65)
ax.set_ylim(0, 1)
ax.set_xticks([0, THR, 0.6])
ax.set_xticklabels(["0", r"$\sqrt{n}/t$", ""])
ax.set_yticks([0.25, 0.75])
ax.set_yticklabels([r"$\bar r=0$", r"$\bar r>0$"])
ax.set_xlabel(r"$\widehat{\mathrm{CV}}_n=\sqrt{s_{cc}}/\bar c$", labelpad=0.5)
ax.set_title("(a) Fieller set, Prop. 1", loc="left", pad=2.5)
ax.tick_params(length=2)

# (b) the four constructions on a zero-outcome arm (two zeros, n = 2)
ax = axes[1]
XL, XR = -0.16, 0.16
rows = []
for lab, cost in (("Fieller, (i) holds", tight), ("Fieller, (i) fails", wide)):
    M = unit(cost)
    lo, hi, kind = RC.fieller_set(M, ALPHA)[:3]
    rows.append((lab, float(lo[0, 0]), float(hi[0, 0]), str(kind[0, 0])))
M = unit(tight)
for name in ("raw delta", "delta-JV", "Jeffreys+cost"):
    lo, hi, kind = RC.all_sets(M, ALPHA)[name]
    rows.append((name, float(lo[0, 0]), float(hi[0, 0]), str(kind[0, 0])))

for k, (lab, lo, hi, kind) in enumerate(rows):
    y = len(rows) - 1 - k
    if kind == "point":
        ax.plot([0], [y], "o", ms=3.4, color="#b2182b", zorder=3)
    elif kind == "line":
        ax.annotate("", xy=(XR, y), xytext=(XL, y),
                    arrowprops=dict(arrowstyle="<->", lw=1.1, color="#2166ac"))
    else:
        pad = 0.022
        l, h = max(lo, XL + pad), min(hi, XR - pad)
        ax.plot([l, h], [y, y], lw=1.8, color="#2166ac", solid_capstyle="butt")
        if lo < XL + pad:
            ax.annotate("", xy=(XL, y), xytext=(l, y), arrowprops=dict(arrowstyle="->", lw=1.1, color="#2166ac"))
        else:
            ax.plot([l], [y], "|", ms=5, color="#2166ac")
        if hi > XR - pad:
            ax.annotate("", xy=(XR, y), xytext=(h, y), arrowprops=dict(arrowstyle="->", lw=1.1, color="#2166ac"))
        else:
            ax.plot([h], [y], "|", ms=5, color="#2166ac")
    ax.text(XL, y + 0.22, lab, fontsize=6.0, va="bottom")
ax.axvline(THETA, color="k", ls=":", lw=0.9)
ax.text(THETA + 0.006, -0.72, r"$\theta$", fontsize=7.5)
ax.set_xlim(XL, XR)
ax.set_ylim(-0.85, len(rows) - 0.25)
ax.set_yticks([])
ax.set_xticks([0, THETA])
ax.set_xticklabels(["0", ""])
ax.set_title("(b) sets on a zero-outcome arm", loc="left", pad=2.5)
ax.tick_params(length=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

fig.savefig(os.path.join(FIGS, "fig_sets.pdf"), bbox_inches="tight", pad_inches=0.02)
print("theta=%.3f threshold=%.3f" % (THETA, THR))
for r in rows:
    print("  %-19s %-9s [%.3f, %.3f]" % (r[0], r[3], r[1], r[2]))
