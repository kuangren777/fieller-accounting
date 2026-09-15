#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the compact data/ JSON files the paper macros are generated from.

Sources (read-only):
  ARCH/icassp_r2_holdout_results.json   frozen independent-seed holdout (MC=400)
  ARCH/icassp_r2_holdout_cache.pkl      its sufficient statistics (branch / zero-variance diagnostics)
  EXP/icassp_r3_selection_results.json  selected-arm + n-screen study on the same collection
  ARCH/icassp_r1_robustness_x4raw.pkl   63-cell distribution x mechanism landscape (per-trajectory sums)
  ARCH/icassp_landscape_cache.pkl       72-cell K x sigma x gamma x budget x policy landscape (unit level)
  ARCH/icassp_r1_split_results.pkl      policy-frozen sample-split control
  ARCH/icassp_realcost_out.txt          Qwen3-0.6B token-cost calibration instance (parsed, two rows)

Every proportion carries a trajectory-cluster bootstrap CI (ratio of trajectory sums).
"""
import json
import os
import pickle
import re
import sys

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import nct

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ARCH = os.path.abspath(os.path.join(ROOT, "..", "archive", "tmp", "experiments"))
EXP = os.path.abspath(os.path.join(ROOT, "..", "experiments"))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402

ALPHA = 0.10
B = 2000
os.makedirs(DATA, exist_ok=True)


def dump(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    print("wrote", os.path.relpath(path, ROOT))


def boot_ratio(num, den, seed, B=B):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if den.sum() <= 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(B, num.size))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


# ---------------------------------------------------------------- R2 + R3 copies
r2 = json.load(open(os.path.join(ARCH, "icassp_r2_holdout_results.json")))
dump("r2_holdout.json", r2)
r3 = json.load(open(os.path.join(EXP, "icassp_r3_selection_results.json")))
dump("r3_selection.json", r3)

# ---------------------------------------------------------------- anchor diagnostics from the R2 collection
cache = pickle.load(open(os.path.join(ARCH, "icassp_r2_holdout_cache.pkl"), "rb"))
rho = np.asarray(cache["rho"], float)
diag = {"source": "icassp_r2_holdout_cache.pkl", "MC": int(cache["data"]["ucb_fieller"]["stats"]["n"].shape[0])}
for pol in ("ucb_fieller", "uniform"):
    S = cache["data"][pol]["stats"]
    n = S["n"]
    rh = S["sr"] / np.maximum(n, 1.0)
    ch = S["sc"] / np.maximum(n, 1.0)
    srr, scc, src = L.moment_vars(rh, ch, S["sr2"], S["sc2"], S["src"], n)
    lo_f, hi_f = L.fieller_ci(rh, ch, srr, scc, src, n, alpha=ALPHA)
    lo_d, hi_d = L.delta_ci(rh, ch, srr, scc, src, n, alpha=ALPHA)
    th = np.broadcast_to(rho, n.shape)
    fin = np.isfinite(lo_f) & np.isfinite(hi_f)
    hit_d = (th >= lo_d) & (th <= hi_d) & np.isfinite(lo_d)
    ge2 = n >= 2
    z = ge2 & (srr <= 1e-15)
    nz = ge2 & ~z
    ns = np.maximum(n, 2.0)
    tq = tdist.ppf(1 - ALPHA / 2, ns - 1.0)
    A = ch ** 2 - tq ** 2 * scc / ns
    Bq = -2.0 * (ch * rh - tq ** 2 * src / ns)
    C = rh ** 2 - tq ** 2 * srr / ns
    D = Bq ** 2 - 4.0 * A * C
    K = n.shape[1]
    allK = np.full(n.shape[0], K, float)
    d = {
        "frac_n_eq_2": boot_ratio((n == 2).sum(1), allK, 11),
        "frac_n_lt_2": boot_ratio((n < 2).sum(1), allK, 12),
        "srr0_given_ge2": boot_ratio(z.sum(1), ge2.sum(1), 13),
        "srr0_given_n_eq_2": boot_ratio((z & (n == 2)).sum(1), (n == 2).sum(1), 14),
        "delta_cov_given_srr0": boot_ratio((hit_d & z).sum(1), z.sum(1), 15),
        "delta_cov_given_srr_pos": boot_ratio((hit_d & nz).sum(1), nz.sum(1), 16),
        "fieller_abstain_and_srr0_over_abstain": boot_ratio((ge2 & ~fin & z).sum(1), (ge2 & ~fin).sum(1), 17),
        "fieller_report_given_srr_pos": boot_ratio((fin & nz).sum(1), nz.sum(1), 18),
        "branch_finite": float(np.mean(fin[ge2])),
        "branch_A_only": float(np.mean(((A <= 0) & (D > 0))[ge2])),
        "branch_D_only": float(np.mean(((A > 0) & (D <= 0))[ge2])),
        "branch_A_and_D": float(np.mean(((A <= 0) & (D <= 0))[ge2])),
        "mean_pulls_per_arm": float(n.mean()),
        "median_n": float(np.median(n)),
        "n_q90": float(np.percentile(n, 90)),
        "n_max": float(n.max()),
        "budget": float(cache["budget"]),
    }
    diag[pol] = d
diag["all_arms_two_zeros_prob"] = float(np.prod((1.0 - np.asarray(cache["env_mu"])) ** 2))
diag["rho_min"] = float(rho.min())
diag["rho_max"] = float(rho.max())
diag["mu_min"] = float(np.min(cache["env_mu"]))
diag["mu_max"] = float(np.max(cache["env_mu"]))
diag["c_min"] = float(np.min(cache["env_c"]))
diag["c_max"] = float(np.max(cache["env_c"]))
dump("anchor_diagnostics.json", diag)

# ---------------------------------------------------------------- X4 landscape (63 cells)
x4 = pickle.load(open(os.path.join(ARCH, "icassp_r1_robustness_x4raw.pkl"), "rb"))
DIST_LABEL = {
    ("lognormal", 0.36, None): "LogNormal CV 0.36", ("gamma", 0.36, None): "Gamma CV 0.36",
    ("pareto", 0.36, None): "Pareto CV 0.36", ("twopoint", 0.36, None): "Two-point CV 0.36",
    ("lognormal", 2.91, None): "LogNormal CV 2.91", ("gamma", 2.91, None): "Gamma CV 2.91",
    ("pareto", 2.91, None): "Pareto CV 2.91", ("twopoint", 2.91, None): "Two-point CV 2.91",
    ("pareto", 0.36, 1.5): "Pareto alpha 1.5",
}
cells = []
for key, v in x4.items():
    cd, cv, rd, mech, bm, es, pa = key
    K = v["K"]
    allK = np.full(v["MC"], K, float)
    label = DIST_LABEL[(cd, round(cv, 2), pa)]
    cells.append({
        "cost_dist": label, "mechanism": mech, "budget": bm,
        "report": boot_ratio(v["f_den"], allK, 101),
        "conditional": boot_ratio(v["f_num"], v["f_den"], 102),
        "joint": boot_ratio(v["f_num"], allK, 103),
        "delta_on_F": boot_ratio(v["d_on_num"], v["f_den"], 104),
        "delta_all": boot_ratio(v["d_all_num"], v["d_all_den"], 105),
        "srr0_given_ge2": boot_ratio(v["srr0_num"], v["ge2_den"], 106),
    })
# one-sided cluster-bootstrap p-value that conditional coverage >= nominal, Holm over the 63 cells
def boot_p_below(num, den, seed, nominal=0.90, B=4000):
    num = np.asarray(num, float); den = np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(B, num.size))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float((np.mean(r >= nominal) * B + 1) / (B + 1))
pvals = []
for c, (key, v) in zip(cells, x4.items()):
    c["p_below_nominal"] = boot_p_below(v["f_num"], v["f_den"], 107)
    pvals.append(c["p_below_nominal"])
order = np.argsort(pvals)
holm = np.empty(len(pvals))
running = 0.0
for rank, i in enumerate(order):
    running = max(running, pvals[i] * (len(pvals) - rank))
    holm[i] = min(1.0, running)
for c, h in zip(cells, holm):
    c["holm_p_below_nominal"] = float(h)
cond = np.array([c["conditional"][0] for c in cells])
upper = np.array([c["conditional"][2] for c in cells])
rep = np.array([c["report"][0] for c in cells])
x4s = {
    "n_cells": len(cells), "cells": cells, "MC": int(next(iter(x4.values()))["MC"]),
    "conditional_min": float(cond.min()), "conditional_max": float(cond.max()),
    "conditional_mean": float(cond.mean()),
    "cluster_upper_below_nominal": int((upper < 0.90).sum()),
    "point_below_nominal": int((cond < 0.90).sum()),
    "report_min": float(rep.min()), "report_max": float(rep.max()),
    "delta_on_F_mean": float(np.mean([c["delta_on_F"][0] for c in cells])),
    "diff_mean": float(np.mean([c["conditional"][0] - c["delta_on_F"][0] for c in cells])),
    "fail_cells": [c for c in cells if c["conditional"][2] < 0.90],
    "holm_fail_cells": [c for c in cells if c["holm_p_below_nominal"] < 0.05],
    "expected_false_fails_at_2p5pct": float(0.025 * len(cells)),
    "worst_report_cell": min(cells, key=lambda c: c["report"][0]),
    "worst_conditional_cell": min(cells, key=lambda c: c["conditional"][0]),
}
dump("x4_landscape.json", x4s)

# ---------------------------------------------------------------- X2 landscape (72 cells)
x2 = pickle.load(open(os.path.join(ARCH, "icassp_landscape_cache.pkl"), "rb"))
cells2 = []
for key, v in x2.items():
    K, sigma, gamma, bm, strat = key
    MC = v["_mc"]
    n = v["n"].reshape(MC, K)
    fin = (np.isfinite(v["lo_f"]) & np.isfinite(v["up_f"])).reshape(MC, K)
    th = v["rho_true"].reshape(MC, K)
    hit = fin & (th >= v["lo_f"].reshape(MC, K)) & (th <= v["up_f"].reshape(MC, K))
    fin_d = (np.isfinite(v["lo_d"]) & np.isfinite(v["up_d"])).reshape(MC, K)
    hit_d = fin_d & (th >= v["lo_d"].reshape(MC, K)) & (th <= v["up_d"].reshape(MC, K))
    allK = np.full(MC, K, float)
    cv = float(np.sqrt(np.exp(sigma ** 2) - 1.0))
    cells2.append({
        "K": K, "sigma": sigma, "cv": cv, "gamma": gamma, "budget": bm, "policy": strat,
        "report": boot_ratio(fin.sum(1), allK, 201),
        "conditional": boot_ratio(hit.sum(1), fin.sum(1), 202),
        "joint": boot_ratio(hit.sum(1), allK, 203),
        "delta_on_F": boot_ratio((hit_d & fin).sum(1), fin.sum(1), 204),
        "delta_all": boot_ratio(hit_d.sum(1), fin_d.sum(1), 205),
    })
ad = [c for c in cells2 if c["policy"] == "ucb_fieller"]
un = [c for c in cells2 if c["policy"] == "uniform"]
x2s = {
    "n_cells": len(cells2), "cells": cells2,
    "adaptive": {
        "conditional_min": float(min(c["conditional"][0] for c in ad)),
        "conditional_max": float(max(c["conditional"][0] for c in ad)),
        "cluster_upper_below_nominal": int(sum(c["conditional"][2] < 0.90 for c in ad)),
        "report_min": float(min(c["report"][0] for c in ad)),
        "report_max": float(max(c["report"][0] for c in ad)),
        "report_mean_by_sigma": {str(s): float(np.mean([c["report"][0] for c in ad if c["sigma"] == s]))
                                 for s in sorted(set(c["sigma"] for c in ad))},
        "worst_report_cell": min(ad, key=lambda c: c["report"][0]),
        "worst_conditional_cell": min(ad, key=lambda c: c["conditional"][0]),
    },
    "uniform": {
        "conditional_min": float(min(c["conditional"][0] for c in un)),
        "conditional_max": float(max(c["conditional"][0] for c in un)),
        "cluster_upper_below_nominal": int(sum(c["conditional"][2] < 0.90 for c in un)),
        "report_min": float(min(c["report"][0] for c in un)),
        "report_max": float(max(c["report"][0] for c in un)),
    },
}
dump("x2_landscape.json", x2s)

# ---------------------------------------------------------------- policy-frozen split control
sp = pickle.load(open(os.path.join(ARCH, "icassp_r1_split_results.pkl"), "rb"))
split = {"source": "icassp_r1_split_results.pkl", "keys": sorted(sp.keys())}
for k, v in sp.items():
    if k == "summaries":
        split["summaries"] = {m: {kk: (list(vv) if isinstance(vv, (list, tuple, np.ndarray)) else vv)
                                  for kk, vv in s.items()} for m, s in v.items()}
    else:
        try:
            json.dumps(v)
            split[k] = v
        except TypeError:
            split[k] = str(v)
dump("split_control.json", split)

# ---------------------------------------------------------------- real-cost calibration (parsed)
rc = {"source": "icassp_realcost_out.txt", "model": "Qwen3-0.6B", "rows": []}
pat = re.compile(r"adaptive-UCB-F\+delta\s+(\d+)x\s+(\d+)\s+([0-9.]+)\[([0-9.]+),([0-9.]+)\]\s*\|\s*"
                 r"([0-9.]+)\[([0-9.]+),([0-9.]+)\]\s+([0-9.]+)\[([0-9.]+),([0-9.]+)\]\s+([0-9.]+)\[([0-9.]+),([0-9.]+)\]")
for line in open(os.path.join(ARCH, "icassp_realcost_out.txt"), encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if m and "units" not in line:
        g = m.groups()
        rc["rows"].append({"budget": int(g[0]), "units": int(g[1]),
                           "report": [float(g[2]), float(g[3]), float(g[4])],
                           "conditional": [float(g[5]), float(g[6]), float(g[7])],
                           "delta_on_F": [float(g[8]), float(g[9]), float(g[10])],
                           "delta_all": [float(g[11]), float(g[12]), float(g[13])]})
        if len(rc["rows"]) == 2:
            break
dump("realcost_calibration.json", rc)

# ---------------------------------------------------------------- R4 policy experiments (copy)
r4 = json.load(open(os.path.join(EXP, "icassp_r4_policy_results.json")))
dump("r4_policy.json", r4)

# ---------------------------------------------------------------- theory box
rng = np.random.default_rng(7)
N = 200000
n = rng.integers(2, 60, N).astype(float)
cbar = rng.uniform(0.1, 20, N)
scc = rng.uniform(1e-6, 50, N)
t = tdist.ppf(1 - ALPHA / 2, n - 1)
A = cbar ** 2 - t ** 2 * scc / n
T = cbar / np.sqrt(scc / n)
ident = np.mean((A <= 0) == (np.abs(T) <= t))
theory = {"identity_tuples": N, "identity_agreement": float(ident),
          "cv_threshold": {str(int(k)): float(np.sqrt(k) / tdist.ppf(1 - ALPHA / 2, k - 1)) for k in (2, 3, 5, 10, 20, 50)},
          "gaussian_reference": []}
for cv in (0.36, 1.0, 2.0, 2.91):
    for nn in (3, 5, 10, 20, 50):
        tq = tdist.ppf(1 - ALPHA / 2, nn - 1)
        a = tq * np.sqrt((nn - 1) / nn)
        dl = np.sqrt(nn) / cv
        p = float(nct.cdf(a, nn - 1, dl) - nct.cdf(-a, nn - 1, dl))
        theory["gaussian_reference"].append({"cv": cv, "n": nn, "p_A_le_0": p})
dump("theory.json", theory)
print("done")


# R5 design study: regularised indices, two-stage reportability design, always-existing constructions
src = os.path.join(EXP, "icassp_r5_designs_results.json")
if os.path.exists(src):
    json.dump(json.load(open(src)), open(os.path.join(DATA, "r5_designs.json"), "w"), indent=1, sort_keys=True)
    print("data/r5_designs.json")
