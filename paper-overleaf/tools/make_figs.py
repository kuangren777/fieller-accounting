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

fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.37), gridspec_kw={"hspace": 0.30})

for ax, mat, title, cmap, vmin, vmax in (
        (axes[0], cond, "(a) Fieller conditional coverage", "viridis", 0.84, 0.99),
        (axes[1], rep, "(b) Fieller report rate", "magma", 0.0, 1.0)):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(7))
    ax.set_xticklabels(MECH_SHORT if ax is axes[1] else [], rotation=45, ha="right")
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




# Method architecture is independently reproducible without loading result data.
from make_method import main as make_method
make_method()
