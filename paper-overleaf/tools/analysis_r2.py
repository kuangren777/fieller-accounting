#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the frozen holdout (icassp_r2_holdout_cache.pkl) with one variance convention
(unbiased, n-1) for every construction and an explicit degenerate-set classification.

Two accountings are reported for every construction:
  'nondeg'  a report is a bounded non-degenerate interval (points, empty sets, lines are non-reports)
  'bounded' a report is any bounded set, so the singleton {0} counts as a zero-width report
Joint coverage does not depend on the accounting.  Confirmatory family (fixed in the R2 manifest):
three common-set coverage comparisons and three S_lambda(1/2) comparisons of Fieller against
raw delta, delta-JV and Jeffreys+cost, Wilcoxon signed-rank on one summary per trajectory, Holm
over six tests; recomputed here under the nondegenerate accounting and the unified convention.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ARCH = os.path.abspath(os.path.join(ROOT, "..", "archive", "tmp", "experiments"))
EXP = os.path.abspath(os.path.join(ROOT, "..", "experiments"))
sys.path.insert(0, EXP)
import ratio_ci as RC  # noqa: E402

ALPHA = 0.10
B = 4000
LAMBDAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
METHODS = ("Fieller", "raw delta", "delta-JV", "Jeffreys+cost", "Bayes", "conf. seq.")


def boot_ratio(num, den, seed, B=B):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() <= 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(B, num.size))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_mean(x, seed, B=B):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_median(x, seed, B=B):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = np.median(x[idx], axis=1)
    return [float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, p[i] * (p.size - rank))
        out[i] = min(1.0, run)
    return out


def analyse(pol, S, rho, unbiased=True):
    M = RC.moments(S, unbiased=unbiased)
    n = M["n"]
    mc, K = n.shape
    allK = np.full(mc, K, float)
    th = np.broadcast_to(rho, n.shape)
    sets = RC.all_sets(M, ALPHA)
    tmax = float(np.max(rho)) * 6.0
    sets["Bayes"] = RC.bayes_set(M, ALPHA)
    sets["conf. seq."] = RC.cs_set(M, ALPHA, theta_max=tmax)
    lo_f, hi_f, kind_f = sets["Fieller"]
    nondeg_f = kind_f == "interval"
    out = {"policy": pol, "MC": mc, "K": K, "methods": {}, "kinds": {}}
    per_traj = {}
    for i, name in enumerate(METHODS):
        lo, hi, kind = sets[name]
        bounded = np.isfinite(lo) & np.isfinite(hi)
        nondeg = kind == "interval"
        hit = bounded & (th >= lo) & (th <= hi)
        width = np.where(bounded, hi - lo, np.nan)
        rec = {"kinds": {k: float(np.mean(kind == k)) for k in ("interval", "point", "empty", "line", "n<2")}}
        for acc, rep in (("nondeg", nondeg), ("bounded", bounded)):
            rec[acc] = {
                "report": boot_ratio(rep.sum(1), allK, 100 + i),
                "conditional": boot_ratio((hit & rep).sum(1), rep.sum(1), 110 + i),
                "joint": boot_ratio((hit & rep).sum(1), allK, 120 + i),
                "width_median": boot_median(np.where(rep, hi - lo, np.nan), 130 + i),
            }
        rec["common_F"] = boot_ratio((hit & nondeg_f).sum(1), nondeg_f.sum(1), 140 + i)
        # unconditional set coverage: an unbounded set (whole line or two rays) contains theta
        line = kind == "line"
        rec["set_coverage"] = boot_ratio((hit | line).sum(1), allK, 145 + i)
        rec["coverage_on_zero_outcome"] = boot_ratio((hit & (S["sr"] == 0) & (n >= 2)).sum(1), ((S["sr"] == 0) & (n >= 2)).sum(1), 150 + i)
        # abstention-penalised score: nondegenerate accounting (points priced as abstentions) and bounded accounting
        IS = RC.interval_score(lo, hi, th, ALPHA)
        rec["score"] = {}
        for acc, rep in (("nondeg", nondeg), ("bounded", bounded)):
            rec["score"][acc] = {}
            for lam in LAMBDAS:
                sc = np.where(rep, IS, lam)
                rec["score"][acc]["%.2f" % lam] = boot_mean(sc.mean(1), 160 + i)
            rec["score"][acc]["finite_IS_mean"] = float(IS[rep].mean()) if rep.any() else float("nan")
        per_traj[name] = {
            "commonF": np.divide((hit & nondeg_f).sum(1), np.maximum(nondeg_f.sum(1), 1)),
            "S_half_nondeg": np.where(nondeg, IS, 0.5).mean(1),
        }
        out["methods"][name] = rec
    # zero-outcome decomposition
    zero = (S["sr"] == 0) & (n >= 2)
    lo_c, hi_c, kind_c = sets["Jeffreys+cost"]
    A_ = RC.fieller_set(M, ALPHA)[3]
    out["zero_outcome"] = {
        "fieller_two_ray_share": float(np.mean((kind_f == "line") & (A_["A"] < 0) & (A_["D"] > 0))),
        "jc_line_share": float(np.mean(kind_c == "line")),
        "jc_line_share_n2": float(np.mean((kind_c == "line") & (n == 2))),
        "fraction": boot_ratio(zero.sum(1), allK, 200),
        "fraction_n_eq_2": boot_ratio(((n == 2) & zero).sum(1), allK, 201),
        "fieller_point_given_zero": boot_ratio((zero & (kind_f == "point")).sum(1), zero.sum(1), 202),
        "fieller_line_given_zero": boot_ratio((zero & (kind_f == "line")).sum(1), zero.sum(1), 203),
        "delta_point_given_zero": boot_ratio((zero & (sets["raw delta"][2] == "point")).sum(1), zero.sum(1), 204),
        "fieller_nondeg_given_nonzero": boot_ratio((~zero & nondeg_f & (n >= 2)).sum(1), (~zero & (n >= 2)).sum(1), 205),
        "median_n": float(np.median(n)), "mean_n": float(n.mean()), "n_max": float(n.max()),
        "frac_n_eq_2": float(np.mean(n == 2)),
    }
    # per-arm conditional coverage (nondegenerate accounting) for Fieller, ordered by true ratio
    hit_f = nondeg_f & (th >= lo_f) & (th <= hi_f)
    per_arm = []
    for a in np.argsort(rho):
        rep_a = nondeg_f[:, a]
        per_arm.append({"arm": int(a), "rho": float(rho[a]), "report": float(rep_a.mean()),
                        "conditional": float(hit_f[:, a].sum() / max(1, rep_a.sum())), "median_n": float(np.median(n[:, a]))})
    out["per_arm_fieller"] = per_arm
    conds = [p["conditional"] for p in per_arm if p["report"] > 0]
    out["per_arm_conditional_min"] = float(min(conds)); out["per_arm_conditional_max"] = float(max(conds))
    top = sorted(per_arm, key=lambda p: -p["rho"])[:3]
    out["per_arm_top3_conditional"] = [p["conditional"] for p in top]
    out["per_arm_top3_report"] = [p["report"] for p in top]
    # confirmatory family
    tests = []
    for other in ("raw delta", "delta-JV", "Jeffreys+cost"):
        for metric, key in (("common-F coverage", "commonF"), ("S_lambda(0.50)", "S_half_nondeg")):
            x = per_traj["Fieller"][key]; y = per_traj[other][key]
            d = x - y
            try:
                p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = 1.0
            tests.append({"metric": metric, "A": "Fieller", "B": other, "difference": boot_mean(d, 300), "raw_p": p})
    hp = holm([t["raw_p"] for t in tests])
    for t, h in zip(tests, hp):
        t["holm_p"] = float(h)
    out["confirmatory_tests"] = tests
    return out


def main():
    d = pickle.load(open(os.path.join(ARCH, "icassp_r2_holdout_cache.pkl"), "rb"))
    rho = np.asarray(d["rho"], float)
    res = {"source": "icassp_r2_holdout_cache.pkl", "script_sha256": d.get("script_sha256"), "alpha": ALPHA,
           "convention": "unbiased variance for every construction", "lambdas": LAMBDAS,
           "rho_max": float(rho.max()), "rho_min": float(rho.min()), "budget": float(d["budget"]),
           "adaptive": analyse("adaptive UCB with delta fallback", d["data"]["ucb_fieller"]["stats"], rho),
           "uniform": analyse("uniform control", d["data"]["uniform"]["stats"], rho),
           "adaptive_mle": analyse("adaptive UCB with delta fallback (MLE variance)", d["data"]["ucb_fieller"]["stats"], rho, unbiased=False),
           "scheduler_fallback": json.load(open(os.path.join(ARCH, "icassp_r2_holdout_results.json")))["scheduler_fallback"]}
    for k in ("adaptive_mle",):
        res[k].pop("per_arm_fieller", None)
    out = os.path.join(ROOT, "data", "r2_recomputed.json")
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    A = res["adaptive"]
    for name in METHODS:
        r = A["methods"][name]
        print("%-14s kinds=%s | nondeg rep=%.3f cond=%.3f | bounded rep=%.3f cond=%.3f | joint=%.3f commonF=%.3f S.5(nondeg)=%.3f S.5(bounded)=%.3f" % (
            name, {k: round(v, 3) for k, v in r["kinds"].items() if v > 0}, r["nondeg"]["report"][0], r["nondeg"]["conditional"][0],
            r["bounded"]["report"][0], r["bounded"]["conditional"][0], r["nondeg"]["joint"][0], r["common_F"][0],
            r["score"]["nondeg"]["0.50"][0], r["score"]["bounded"]["0.50"][0]))
    print("zero-outcome", {k: (v if isinstance(v, float) else round(v[0], 3)) for k, v in A["zero_outcome"].items()})
    print("per-arm cond range", A["per_arm_conditional_min"], A["per_arm_conditional_max"], "top3", A["per_arm_top3_conditional"], A["per_arm_top3_report"])
    for t in A["confirmatory_tests"]:
        print("test %-18s vs %-14s diff=%+.4f [%+.4f,%+.4f] raw p=%.3g Holm=%.3g" % (t["metric"], t["B"], *t["difference"], t["raw_p"], t["holm_p"]))
    print("written", out)


if __name__ == "__main__":
    main()
