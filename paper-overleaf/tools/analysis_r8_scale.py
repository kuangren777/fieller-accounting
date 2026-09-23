#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostics on the frozen holdout (no new sampling), written to data/r8_scale.json.

  (1) where Fieller reports: pull counts and g = t^2 CVhat^2 / n on non-degenerate units, and
      Fieller against raw delta coverage on the units with g above 0.1;
  (2) width tails of every construction among its reports;
  (3) the S_lambda leader over a fine lambda grid.  Rescaling the cost unit by k divides every
      interval score by k, so ranking at lambda under unit k equals ranking at k*lambda under
      the original unit, and the grid covers every choice of cost unit.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_r2 as A  # noqa: E402

RC = A.RC
ALPHA = A.ALPHA
G_CUT = 0.1


def main():
    d = pickle.load(open(os.path.join(A.ARCH, "icassp_r2_holdout_cache.pkl"), "rb"))
    rho = np.asarray(d["rho"], float)
    S = d["data"]["ucb_fieller"]["stats"]
    costs = pickle.load(open(os.path.join(A.EXP, "icassp_r7_holdout_cmin.pkl"), "rb"))["sorted_costs"]
    M = RC.moments(S)
    n = M["n"]
    th = np.broadcast_to(rho, n.shape)
    sets = RC.all_sets(M, ALPHA)
    sets["Bayes"] = RC.bayes_set(M, ALPHA)
    sets["conf. seq."] = RC.cs_set(M, ALPHA, theta_max=float(rho.max()) * 6.0)
    sets["Fieller-JV"] = RC.fieller_jv_set(M, ALPHA)
    sets["MOVER-R"] = RC.mover_r_set(M, ALPHA)
    sets["positivity"] = RC.positivity_set(M, costs, ALPHA)

    lf, hf, kf = sets["Fieller"]
    ld, hd, _ = sets["raw delta"]
    nd = kf == "interval"
    tq = tdist.ppf(1.0 - ALPHA / 2.0, np.maximum(n - 1.0, 1.0))
    g = tq ** 2 * M["scc"] / np.maximum(M["ch"], 1e-300) ** 2 / np.maximum(n, 1.0)
    hi_g = nd & (g > G_CUT)
    out = {"source": "icassp_r2_holdout_cache.pkl", "alpha": ALPHA, "g_cut": G_CUT,
           "fieller_reports": {
               "n_quartiles": [float(v) for v in np.percentile(n[nd], [25, 50, 75])],
               "share_g_below_cut": float(np.mean(g[nd] <= G_CUT)),
               "n_units_g_above_cut": int(hi_g.sum()),
               "fieller_cov_g_above": float(np.mean((th >= lf) & (th <= hf), where=hi_g)),
               "delta_cov_g_above": float(np.mean((th >= ld) & (th <= hd), where=hi_g))}}

    widths, score_curve = {}, {}
    lams = np.round(np.arange(0.01, 2.0001, 0.01), 2)
    for name, (lo, hi, kind) in sets.items():
        rep = kind == "interval"
        w = (hi - lo)[rep]
        IS = RC.interval_score(lo, hi, th, ALPHA)
        widths[name] = {"median": float(np.median(w)), "mean": float(w.mean()), "p90": float(np.percentile(w, 90)),
                        "mean_IS": float(IS[rep].mean())}
        score_curve[name] = [float(np.where(rep, IS, lam).mean()) for lam in lams]
    leader = [min(score_curve, key=lambda m: score_curve[m][i]) for i in range(len(lams))]
    lead_mr = [float(l) for l, w in zip(lams, leader) if w == "MOVER-R"]
    out["widths"] = widths
    out["lambda_grid"] = [float(v) for v in lams]
    out["leader"] = leader
    out["mover_r_lead"] = {"lo": min(lead_mr), "hi": max(lead_mr),
                           "contiguous": len(lead_mr) == int(round((max(lead_mr) - min(lead_mr)) / 0.01)) + 1}
    out["rho_max"] = float(rho.max())
    path = os.path.join(A.ROOT, "data", "r8_scale.json")
    json.dump(out, open(path, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k not in ("lambda_grid", "leader")}, indent=1))


if __name__ == "__main__":
    main()
