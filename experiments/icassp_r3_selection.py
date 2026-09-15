#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICASSP R3: selected-arm (post-selection) reportability and the n-screen ablation.

Reads the frozen R2 holdout collection (sufficient statistics of 400 independent
trajectories x 24 arms per policy) and answers two reviewer items without new
sampling:

  M6  selected-arm inference.  For each trajectory the analyst picks one arm by a
      data-dependent rule (largest empirical ratio among n>=2 arms, or the most
      pulled arm) and reports that arm's interval.  We report the report rate,
      the conditional coverage and the joint coverage of that single selected
      interval for Fieller, raw delta, delta-JV and Jeffreys+cost, and for the
      Bonferroni-simultaneous version of each method (per-arm level alpha/K), so
      that the reader sees the price of a valid-by-construction simultaneous band.
      The fixed true-best arm is included as the fixed-arm reference.

  F5  delta with a hand-set sample-size screen.  Raw delta reported only when
      n>=n0, for a pre-listed grid of n0.  The screen matched to Fieller's report
      rate is chosen on this same collection, so it is descriptive only.

All uncertainty intervals are percentile bootstraps over independent trajectories
(one selected unit per trajectory for the selection study; ratio-of-sums over
trajectory clusters for arm-level quantities).  No claim of adaptive validity is
made; this is a diagnostic on the frozen benchmark.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "..", "archive", "tmp", "experiments")
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402

CACHE = os.path.join(ARCH, "icassp_r2_holdout_cache.pkl")
OUT_JSON = os.path.join(HERE, "icassp_r3_selection_results.json")
ALPHA = 0.10
K = 24
B_BOOT = 4000
SEED = 930001
N0_GRID = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]


def moment(S):
    n = S["n"]
    rh = S["sr"] / np.maximum(n, 1.0)
    ch = S["sc"] / np.maximum(n, 1.0)
    srr, scc, src = L.moment_vars(rh, ch, S["sr2"], S["sc2"], S["src"], n)
    return {"n": n, "rh": rh, "ch": ch, "srr": srr, "scc": scc, "src": src, "sr": S["sr"]}


def fieller(M, alpha):
    return L.fieller_ci(M["rh"], M["ch"], M["srr"], M["scc"], M["src"], M["n"], alpha=alpha)


def raw_delta(M, alpha):
    return L.delta_ci(M["rh"], M["ch"], M["srr"], M["scc"], M["src"], M["n"], alpha=alpha)


def delta_jv(M, alpha):
    """Jeffreys-variance regularised delta (same construction as icassp_r1_baselines)."""
    n = M["n"]
    p = (M["sr"] + 0.5) / (n + 1.0)
    srr = p * (1.0 - p)
    den = np.sqrt(np.maximum(M["srr"], 0.0) * np.maximum(M["scc"], 0.0))
    corr = np.divide(M["src"], np.maximum(den, 1e-15), out=np.zeros_like(den), where=den > 1e-15)
    corr = np.clip(corr, -1.0, 1.0)
    src = corr * np.sqrt(srr * np.maximum(M["scc"], 0.0))
    return L.delta_ci(M["rh"], M["ch"], srr, M["scc"], src, n, alpha=alpha)


def jeffreys_cost(M, alpha):
    """Jeffreys reward interval + Student-t cost interval, Bonferroni split alpha/2 each."""
    n = M["n"]
    lo = np.full(n.shape, -np.inf)
    hi = np.full(n.shape, np.inf)
    ok = n >= 2
    nn = np.maximum(n, 2.0)
    s = M["sr"]
    q = tdist.ppf(1.0 - alpha / 4.0, nn - 1.0)
    rlo = beta_dist.ppf(alpha / 4.0, s + 0.5, nn - s + 0.5)
    rhi = beta_dist.ppf(1.0 - alpha / 4.0, s + 0.5, nn - s + 0.5)
    cse = np.sqrt(np.maximum(M["scc"], 0.0) / np.maximum(nn - 1.0, 1.0))
    clo = M["ch"] - q * cse
    chi = M["ch"] + q * cse
    good = ok & np.isfinite(rlo) & np.isfinite(rhi) & (clo > 0) & (chi > 0)
    lo[good] = rlo[good] / chi[good]
    hi[good] = rhi[good] / clo[good]
    return lo, hi


METHODS = {
    "Fieller": fieller,
    "raw delta": raw_delta,
    "delta-JV": delta_jv,
    "Jeffreys+cost": jeffreys_cost,
}


def boot_mean(x, rng, B=B_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def boot_ratio(num, den, rng, B=B_BOOT):
    """Ratio of sums with trajectory-cluster bootstrap. num, den: (MC,) per-trajectory sums."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    mc = num.size
    if den.sum() <= 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, mc, size=(B, mc))
    r = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-12)
    return (float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def boot_median(x, rng, B=B_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = np.median(x[idx], axis=1)
    return (float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def fmt(t):
    return "%.4f [%.4f,%.4f]" % t


def selection_study(policy, M, rho, rng):
    mc = M["n"].shape[0]
    ge2 = M["n"] >= 2
    ratio = np.where(ge2, M["rh"] / np.maximum(M["ch"], 1e-12), -np.inf)
    best_true = int(np.argmax(rho))
    rules = {
        "argmax empirical ratio (n>=2)": np.argmax(ratio, axis=1),
        "most pulled arm": np.argmax(M["n"], axis=1),
        "fixed true-best arm (reference)": np.full(mc, best_true, int),
    }
    out = {"true_best_arm": best_true, "rules": {}}
    rows = np.arange(mc)
    print("\n[%s] selected-arm study; one selected unit per trajectory; MC=%d" % (policy, mc))
    for rname, sel in rules.items():
        ident = float(np.mean(sel == best_true))
        nsel = M["n"][rows, sel]
        print("  rule: %-34s  P(selected == true best)=%.3f  median n(selected)=%.0f  min n=%.0f"
              % (rname, ident, np.median(nsel), nsel.min()))
        rres = {"identification_rate": ident,
                "selected_n_median": float(np.median(nsel)),
                "selected_n_q10": float(np.percentile(nsel, 10)),
                "methods": {}}
        for mname, fn in METHODS.items():
            for level_name, lvl in (("per-arm alpha", ALPHA), ("Bonferroni alpha/K", ALPHA / K)):
                lo, hi = fn(M, lvl)
                lo_s, hi_s = lo[rows, sel], hi[rows, sel]
                th = rho[sel]
                fin = np.isfinite(lo_s) & np.isfinite(hi_s)
                hit = fin & (th >= lo_s) & (th <= hi_s)
                rep = boot_mean(fin.astype(float), rng)
                cond = boot_ratio(hit.astype(float), fin.astype(float), rng)
                joint = boot_mean(hit.astype(float), rng)
                wid = boot_median(np.where(fin, hi_s - lo_s, np.nan), rng)
                key = "%s | %s" % (mname, level_name)
                rres["methods"][key] = {"report": rep, "conditional": cond, "joint": joint,
                                        "width_median": wid}
                print("    %-32s report=%s  conditional=%s  joint=%s  med width=%s"
                      % (key, fmt(rep), fmt(cond), fmt(joint), fmt(wid)))
        out["rules"][rname] = rres
    return out


def screen_study(policy, M, rho, rng):
    """Raw delta gated by n>=n0 versus Fieller's own reportability."""
    n = M["n"]
    mc = n.shape[0]
    lo_f, hi_f = fieller(M, ALPHA)
    lo_d, hi_d = raw_delta(M, ALPHA)
    th = np.broadcast_to(rho, n.shape)
    fin_f = np.isfinite(lo_f) & np.isfinite(hi_f)
    hit_f = fin_f & (th >= lo_f) & (th <= hi_f)
    hit_d = np.isfinite(lo_d) & np.isfinite(hi_d) & (th >= lo_d) & (th <= hi_d)
    f_rep = boot_ratio(fin_f.sum(1), np.full(mc, K), rng)
    f_cond = boot_ratio(hit_f.sum(1), fin_f.sum(1), rng)
    f_joint = boot_ratio(hit_f.sum(1), np.full(mc, K), rng)
    f_w = boot_median(np.where(fin_f, hi_f - lo_f, np.nan), rng)
    print("\n[%s] raw delta with a hand-set sample-size screen n>=n0 (descriptive ablation)" % policy)
    print("  Fieller own screen: report=%s conditional=%s joint=%s med width=%s"
          % (fmt(f_rep), fmt(f_cond), fmt(f_joint), fmt(f_w)))
    out = {"Fieller": {"report": f_rep, "conditional": f_cond, "joint": f_joint, "width_median": f_w},
           "screens": {}}
    best_n0, best_gap = None, 9.0
    for n0 in N0_GRID:
        gate = n >= n0
        rep = boot_ratio(gate.sum(1), np.full(mc, K), rng)
        cond = boot_ratio((hit_d & gate).sum(1), gate.sum(1), rng)
        joint = boot_ratio((hit_d & gate).sum(1), np.full(mc, K), rng)
        wid = boot_median(np.where(gate, hi_d - lo_d, np.nan), rng)
        agree = float(np.mean(gate == fin_f))
        inter = float((gate & fin_f).sum()); union = float((gate | fin_f).sum())
        jacc = inter / union if union > 0 else float("nan")
        # delta on units the screen admits but Fieller abstains on
        extra = gate & ~fin_f
        cov_extra = boot_ratio((hit_d & extra).sum(1), extra.sum(1), rng)
        out["screens"][str(n0)] = {"report": rep, "conditional": cond, "joint": joint,
                                   "width_median": wid, "agreement_with_Fieller_set": agree,
                                   "jaccard_with_Fieller_set": jacc,
                                   "delta_coverage_on_screen_only_units": cov_extra,
                                   "screen_only_fraction": float(extra.mean())}
        gap = abs(rep[0] - f_rep[0])
        if gap < best_gap:
            best_gap, best_n0 = gap, n0
        print("  n0=%3d report=%s conditional=%s joint=%s med width=%s agree=%.3f jaccard=%.3f"
              " delta cov on screen-only units=%s (frac %.3f)"
              % (n0, fmt(rep), fmt(cond), fmt(joint), fmt(wid), agree, jacc, fmt(cov_extra), extra.mean()))
    out["matched_n0_descriptive"] = best_n0
    print("  screen with report rate closest to Fieller: n0=%d (chosen on this collection; descriptive)" % best_n0)
    # reportability of Fieller as a function of n (bucketed)
    buckets = [(2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10 ** 9)]
    out["fieller_report_by_n"] = {}
    print("  Fieller report rate by sample size bucket:")
    for a, b in buckets:
        m = (n >= a) & (n <= b)
        if m.sum() == 0:
            continue
        r = boot_ratio((fin_f & m).sum(1), m.sum(1), rng)
        srr0 = float(np.mean(M["srr"][m] <= 1e-15))
        out["fieller_report_by_n"]["%d-%s" % (a, "inf" if b > 10 ** 8 else str(b))] = {
            "report": r, "unit_fraction": float(m.mean()), "srr0_fraction": srr0}
        print("    n in [%d,%s]: units=%.3f report=%s  P(s_rr=0)=%.3f"
              % (a, "inf" if b > 10 ** 8 else str(b), m.mean(), fmt(r), srr0))
    return out


def main():
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    rho = np.asarray(d["rho"], float)
    print("ICASSP R3 selection + screen study on the frozen R2 holdout collection")
    print("cache=%s  K=%d  MC=%d  alpha=%.2f  bootstrap B=%d" % (
        os.path.basename(CACHE), K, d["data"]["ucb_fieller"]["stats"]["n"].shape[0], ALPHA, B_BOOT))
    print("scope: diagnostic on frozen trajectories; no adaptive-validity or selection-adjusted theorem")
    results = {"alpha": ALPHA, "K": K, "bootstrap_B": B_BOOT, "source_cache": os.path.basename(CACHE),
               "source_script_sha256": d.get("script_sha256"), "policies": {}}
    for policy in ("ucb_fieller", "uniform"):
        rng = np.random.default_rng(SEED + (0 if policy == "ucb_fieller" else 1))
        M = moment(d["data"][policy]["stats"])
        pres = {"selection": selection_study(policy, M, rho, rng),
                "screen": screen_study(policy, M, rho, rng)}
        results["policies"][policy] = pres
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\nwritten: %s" % OUT_JSON)


if __name__ == "__main__":
    main()
