#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICASSP R2 frozen independent-seed holdout and scheduler-fallback audit.

Collection is permitted only when the adjacent frozen JSON manifest contains the
SHA-256 digest of this script.  The holdout keeps the benchmark environment,
policies, estimands, intervals, score penalties, tests, and seed range fixed.  It
changes only the Monte Carlo trajectory seeds relative to development runs.

This is an independent-seed simulation holdout, not an external-data validation
and not a theorem of validity under adaptive allocation, optional stopping, or
post-selection.  The estimands are the K fixed-arm ratios; selected-arm coverage
is outside scope.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expD_lib as L  # noqa: E402
import icassp_r1_baselines as R1  # noqa: E402

MANIFEST = os.path.join(HERE, "icassp_r2_holdout_frozen.json")
CACHE = os.path.join(HERE, "icassp_r2_holdout_cache.pkl")
RESULTS = os.path.join(HERE, "icassp_r2_holdout_results.json")

K = 24
MC = 400
SIGMA = 0.35
GAMMA = 0.0
ALPHA = 0.10
B_MULT = 30
ENV_SEED = 1
SEED_START = 910000
CLUSTER_B = 4000
ORACLE_N = 500_000
ORACLE_SEED = 909001
POLICIES = ("ucb_fieller", "uniform")
METHODS = ("Fieller", "raw delta", "delta-JV", "Jeffreys+cost")
SCORE_LAMBDAS = (0.10, 0.50, 1.00)

np.set_printoptions(precision=6, suppress=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        m = json.load(f)
    digest = file_sha256(__file__)
    if digest != m["script_sha256"]:
        raise RuntimeError("holdout script digest differs from frozen manifest")
    expected = {
        "K": K, "MC": MC, "sigma": SIGMA, "gamma": GAMMA,
        "alpha": ALPHA, "budget_multiplier": B_MULT,
        "environment_seed": ENV_SEED, "seed_start": SEED_START,
        "seed_end": SEED_START + MC - 1, "cluster_bootstrap_replicates": CLUSTER_B,
        "policies": list(POLICIES), "methods": list(METHODS),
        "score_lambdas": list(SCORE_LAMBDAS),
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise RuntimeError("manifest mismatch for %s: %r != %r" % (key, m.get(key), value))
    if m.get("status") != "frozen-before-execution":
        raise RuntimeError("manifest is not marked frozen-before-execution")
    return m, digest


def true_ratio(env):
    ec = env["c"] * np.exp(env["cost_sigma"] ** 2 / 2.0) * (1.0 + env["gamma"] * env["mu"])
    return env["mu"] / ec


def oracle_selfcheck(env, rho):
    rng = np.random.default_rng(ORACLE_SEED)
    est = np.empty(K, float)
    for a in range(K):
        left = ORACLE_N
        sr = 0.0
        sc = 0.0
        while left:
            n = min(left, 100_000)
            rr = rng.random(n) < env["mu"][a]
            cc = rng.lognormal(np.log(env["c"][a]), SIGMA, n)
            if GAMMA:
                cc = np.where(rr, cc * (1.0 + GAMMA), cc)
            sr += float(rr.sum())
            sc += float(cc.sum())
            left -= n
        est[a] = sr / sc
    rel = np.abs(est / rho - 1.0)
    print("[1] oracle check: N=%d iid draws/arm, max relative error=%.4f%%, mean=%.4f%%" %
          (ORACLE_N, 100 * rel.max(), 100 * rel.mean()))
    print("    threshold=1%%; verdict=%s" % ("PASS" if rel.max() < 0.01 else "FAIL"))
    if rel.max() >= 0.01:
        raise RuntimeError("analytic ratio oracle failed")


def empty_stats():
    return {key: np.zeros((MC, K), float) for key in ("n", "sr", "sc", "sr2", "sc2", "src")}


def validate_audit(audit, log_len, policy):
    if policy != "ucb_fieller":
        return
    if audit["policy_decisions"] != log_len - K:
        raise RuntimeError("policy-decision count mismatch")
    if audit["candidate_evaluations"] != K * audit["policy_decisions"]:
        raise RuntimeError("candidate-evaluation count mismatch")
    if sum(audit["quartile_decisions"]) != audit["policy_decisions"]:
        raise RuntimeError("quartile decision count mismatch")
    if sum(audit["quartile_candidate_evaluations"]) != audit["candidate_evaluations"]:
        raise RuntimeError("quartile candidate count mismatch")
    if sum(audit["quartile_fallback_candidates"]) != audit["fallback_candidate_evaluations"]:
        raise RuntimeError("quartile candidate-fallback count mismatch")
    if sum(audit["quartile_selected_fallback"]) != audit["selected_fallback_decisions"]:
        raise RuntimeError("quartile selected-fallback count mismatch")
    candidate_reasons = sum(audit[x] for x in
                            ("fallback_n_lt_2", "fallback_A_only", "fallback_D_only",
                             "fallback_A_and_D"))
    selected_reasons = sum(audit[x] for x in
                           ("selected_n_lt_2", "selected_A_only", "selected_D_only",
                            "selected_A_and_D"))
    if candidate_reasons != audit["fallback_candidate_evaluations"]:
        raise RuntimeError("candidate fallback reasons do not partition fallback events")
    if selected_reasons != audit["selected_fallback_decisions"]:
        raise RuntimeError("selected fallback reasons do not partition fallback events")


def collect(manifest, digest):
    if os.path.exists(CACHE):
        raise RuntimeError("holdout cache already exists; refusing a second collection")
    env = L.make_env(K=K, mu_range=(0.06, 0.30), cost_sigma=SIGMA,
                     cost_mean=5.0, gamma=GAMMA, nonstat="static", seed=ENV_SEED)
    rho = true_ratio(env)
    oracle_selfcheck(env, rho)
    budget = float(B_MULT * np.sum(env["base_c0"]) * np.exp(SIGMA ** 2 / 2.0))
    data = {}
    t0 = time.time()
    for policy in POLICIES:
        S = empty_stats()
        reward = np.zeros(MC, float)
        cost = np.zeros(MC, float)
        audits = []
        for i in range(MC):
            result = L.run(env, policy, budget, seed=SEED_START + i, pulls_log=True)
            for key in S:
                S[key][i] = result["arm_stats"][key]
            reward[i] = result["reward"]
            cost[i] = result["cost"]
            audit = result["scheduler_audit"]
            validate_audit(audit, len(result["pulls_log"]), policy)
            audits.append(audit)
            if (i + 1) % 50 == 0:
                print("[2] %-13s %3d/%d trajectories (%.1fs)" %
                      (policy, i + 1, MC, time.time() - t0), flush=True)
        data[policy] = {"stats": S, "reward": reward, "cost": cost, "audits": audits}
    obj = {
        "version": 1, "manifest": manifest, "script_sha256": digest,
        "rho": rho, "env_mu": env["mu"], "env_c": env["c"], "budget": budget,
        "data": data,
    }
    with open(CACHE, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print("[3] immutable holdout collection written: %s (%.3f MB)" %
          (CACHE, os.path.getsize(CACHE) / 1e6))
    return obj


def load_cache(manifest, digest):
    with open(CACHE, "rb") as f:
        obj = pickle.load(f)
    if obj.get("version") != 1 or obj.get("script_sha256") != digest:
        raise RuntimeError("holdout cache version/digest mismatch")
    if obj.get("manifest") != manifest:
        raise RuntimeError("holdout cache manifest mismatch")
    return obj


def sufficient_stats(raw, rho):
    n = raw["n"]
    rh = raw["sr"] / np.maximum(n, 1.0)
    ch = raw["sc"] / np.maximum(n, 1.0)
    srr, scc, src = L.moment_vars(rh, ch, raw["sr2"], raw["sc2"], raw["src"], n)
    return {
        "n": n, "sr": raw["sr"], "rh": rh, "ch": ch,
        "srr": srr, "scc": scc, "src": src,
        "rho": np.broadcast_to(np.asarray(rho)[None, :], (MC, K)).copy(),
    }


def interval_methods(S):
    lf, hf = L.fieller_ci(S["rh"], S["ch"], S["srr"], S["scc"], S["src"], S["n"], alpha=ALPHA)
    ld, hd = L.delta_ci(S["rh"], S["ch"], S["srr"], S["scc"], S["src"], S["n"], alpha=ALPHA)
    lj, hj = R1.delta_jv(S)
    le, he = R1.jeffreys_cost_ratio(S)
    return {
        "Fieller": (lf, hf), "raw delta": (ld, hd),
        "delta-JV": (lj, hj), "Jeffreys+cost": (le, he),
    }


def cluster_ratio(num, den, seed):
    num = np.asarray(num, float).reshape(MC, -1).sum(axis=1)
    den = np.asarray(den, float).reshape(MC, -1).sum(axis=1)
    if den.sum() <= 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, MC, size=(CLUSTER_B, MC))
    bn = num[idx].sum(axis=1)
    bd = den[idx].sum(axis=1)
    z = np.divide(bn, bd, out=np.full(CLUSTER_B, np.nan), where=bd > 0)
    z = z[np.isfinite(z)]
    return point, float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))


def cluster_mean(x, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return (float("nan"), float("nan"), float("nan"))
    point = float(x.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(CLUSTER_B, x.size))
    z = x[idx].mean(axis=1)
    return point, float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))


def cluster_quantile(values, mask, q, seed):
    values = np.asarray(values, float).reshape(MC, -1)
    mask = np.asarray(mask, bool).reshape(MC, -1)
    flat = values[mask & np.isfinite(values)]
    if not flat.size:
        return (float("nan"), float("nan"), float("nan"))
    point = float(np.percentile(flat, q))
    rng = np.random.default_rng(seed)
    z = np.empty(CLUSTER_B, float)
    for b in range(CLUSTER_B):
        idx = rng.integers(0, MC, size=MC)
        sample = values[idx]
        sample_mask = mask[idx]
        vals = sample[sample_mask & np.isfinite(sample)]
        z[b] = np.percentile(vals, q) if vals.size else np.nan
    z = z[np.isfinite(z)]
    return point, float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))


def fmt(x):
    if x is None or len(x) != 3 or not np.isfinite(x[0]):
        return "n/a"
    return "%.4f [%.4f,%.4f]" % tuple(float(v) for v in x)


def summarize(policy, raw, rho, seed0):
    S = sufficient_stats(raw, rho)
    intervals = interval_methods(S)
    base = np.ones((MC, K), bool)
    common_f = np.isfinite(intervals["Fieller"][0]) & np.isfinite(intervals["Fieller"][1])
    common_d = np.isfinite(intervals["raw delta"][0]) & np.isfinite(intervals["raw delta"][1])
    out = {"S": S, "intervals": intervals, "common_f": common_f,
           "common_d": common_d, "methods": {}}
    for j, (name, (lo, hi)) in enumerate(intervals.items()):
        finite = np.isfinite(lo) & np.isfinite(hi)
        hit = finite & (S["rho"] >= lo) & (S["rho"] <= hi)
        width = hi - lo
        scores = {}
        for k, lam in enumerate(SCORE_LAMBDAS):
            tr = R1.capped_score(lo, hi, S["rho"], lam).mean(axis=1)
            scores[str(lam)] = cluster_mean(tr, seed0 + 100 * j + 20 + k)
        out["methods"][name] = {
            "report": cluster_ratio(finite, base, seed0 + 100 * j + 1),
            "conditional": cluster_ratio(hit, finite, seed0 + 100 * j + 2),
            "joint": cluster_ratio(hit, base, seed0 + 100 * j + 3),
            "common_F": cluster_ratio(hit & common_f, common_f, seed0 + 100 * j + 4),
            "common_D_accounting": cluster_ratio(hit & common_d, common_d, seed0 + 100 * j + 5),
            "width_q25": cluster_quantile(width, finite, 25, seed0 + 100 * j + 6),
            "width_median": cluster_quantile(width, finite, 50, seed0 + 100 * j + 7),
            "width_q75": cluster_quantile(width, finite, 75, seed0 + 100 * j + 8),
            "score": scores,
            "ordinary_score": "+inf" if np.any(~finite) else float(R1.interval_score(lo, hi, S["rho"]).mean()),
            "finite": finite, "hit": hit,
        }
    return out


def trajectory_coverage(hit, common):
    den = common.sum(axis=1)
    num = (hit & common).sum(axis=1)
    return np.divide(num, den, out=np.full(MC, np.nan), where=den > 0)


def paired_tests(out):
    S = out["S"]
    common = out["common_f"]
    rows = []
    opponents = ("raw delta", "delta-JV", "Jeffreys+cost")
    f = out["methods"]["Fieller"]
    for j, other in enumerate(opponents):
        a = trajectory_coverage(f["hit"], common)
        b = trajectory_coverage(out["methods"][other]["hit"], common)
        rows.append(test_row("common-F coverage", "Fieller", other, a, b, 70000 + j))
    lf, hf = out["intervals"]["Fieller"]
    sf = R1.capped_score(lf, hf, S["rho"], 0.50).mean(axis=1)
    for j, other in enumerate(opponents):
        lo, hi = out["intervals"][other]
        so = R1.capped_score(lo, hi, S["rho"], 0.50).mean(axis=1)
        rows.append(test_row("S_lambda(0.50)", "Fieller", other, sf, so, 70100 + j))
    adjusted = R1.holm([r["raw_p"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p"] = float(value)
    return rows


def test_row(metric, a_name, b_name, a, b, seed):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    if d.size >= 5 and np.any(np.abs(d) > 1e-12):
        p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    else:
        p = 1.0
    return {"metric": metric, "A": a_name, "B": b_name,
            "difference_A_minus_B": cluster_mean(d, seed), "raw_p": p,
            "n_trajectories": int(d.size)}


def aggregate_fallback(audits):
    scalar = (
        "initialization_pulls", "policy_decisions", "candidate_evaluations",
        "fallback_candidate_evaluations", "selected_fallback_decisions",
        "fallback_n_lt_2", "fallback_A_only", "fallback_D_only", "fallback_A_and_D",
        "selected_n_lt_2", "selected_A_only", "selected_D_only", "selected_A_and_D",
    )
    total = {key: int(sum(a[key] for a in audits)) for key in scalar}
    vector = ("quartile_decisions", "quartile_fallback_candidates",
              "quartile_candidate_evaluations", "quartile_selected_fallback")
    for key in vector:
        total[key] = np.sum(np.asarray([a[key] for a in audits], int), axis=0).astype(int).tolist()
    total["candidate_fallback_rate"] = (total["fallback_candidate_evaluations"] /
                                          total["candidate_evaluations"])
    total["selected_fallback_rate"] = (total["selected_fallback_decisions"] /
                                         total["policy_decisions"])
    total["quartile_candidate_fallback_rate"] = [
        a / b for a, b in zip(total["quartile_fallback_candidates"],
                              total["quartile_candidate_evaluations"])
    ]
    total["quartile_selected_fallback_rate"] = [
        a / b for a, b in zip(total["quartile_selected_fallback"],
                              total["quartile_decisions"])
    ]
    return total


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()
                if k not in ("S", "intervals", "finite", "hit", "common_f", "common_d")}
    if isinstance(obj, (list, tuple)):
        return [json_ready(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def print_summary(label, out):
    print("\n[4] %s" % label)
    print("    fixed-arm ratio estimands; MC=%d independent trajectories; K=%d dependent arms/trajectory" % (MC, K))
    print("    all uncertainty intervals: trajectory-cluster bootstrap ratio-of-sums, B=%d" % CLUSTER_B)
    print("    common-D is an accounting set, not each method's ordinary conditional coverage.")
    print("method          | report rate          | conditional          | joint                | common-F             | common-D accounting  | median width          | S_lambda(.50)")
    for name in METHODS:
        m = out["methods"][name]
        print("%-15s | %-20s | %-20s | %-20s | %-20s | %-20s | %-20s | %s" %
              (name, fmt(m["report"]), fmt(m["conditional"]), fmt(m["joint"]),
               fmt(m["common_F"]), fmt(m["common_D_accounting"]),
               fmt(m["width_median"]), fmt(m["score"]["0.5"])))
        print("    width %-15s Q1=%s median=%s Q3=%s; ordinary score=%s" %
              (name, fmt(m["width_q25"]), fmt(m["width_median"]), fmt(m["width_q75"]),
               m["ordinary_score"]))
    print("    score sensitivity (descriptive except lambda=.50 in the frozen confirmatory family):")
    for lam in SCORE_LAMBDAS:
        print("      lambda=%.2f: %s" %
              (lam, " | ".join("%s=%s" % (name, fmt(out["methods"][name]["score"][str(lam)]))
                                  for name in METHODS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="perform the one permitted frozen holdout collection")
    args = parser.parse_args()
    manifest, digest = load_and_validate_manifest()
    print("=" * 132)
    print("ICASSP R2 frozen independent-seed holdout | reportability, common sets, score, scheduler fallback")
    print("script_sha256=%s" % digest)
    print("seeds=%d..%d MC=%d K=%d sigma=%.2f gamma=%.1f budget=%dx alpha=%.2f" %
          (SEED_START, SEED_START + MC - 1, MC, K, SIGMA, GAMMA, B_MULT, ALPHA))
    print("scope: fixed-arm ratios; no adaptive-validity, optional-stopping, post-selection, or external-validation claim")
    print("=" * 132)
    print("[0] frozen manifest validated: %s" % MANIFEST)
    if args.collect:
        obj = collect(manifest, digest)
    else:
        obj = load_cache(manifest, digest)
        print("[1] loaded frozen holdout cache: %s (%.3f MB)" %
              (CACHE, os.path.getsize(CACHE) / 1e6))

    summaries = {}
    for j, policy in enumerate(POLICIES):
        summaries[policy] = summarize(policy, obj["data"][policy]["stats"], obj["rho"],
                                      50000 + 10000 * j)
    print_summary("adaptive UCB with delta fallback, 30x", summaries["ucb_fieller"])
    print_summary("uniform control, 30x", summaries["uniform"])

    tests = paired_tests(summaries["ucb_fieller"])
    print("\n[5] frozen confirmatory family: 3 common-F coverage + 3 all-unit S_lambda(.50) comparisons")
    print("    two-sided Wilcoxon on independent trajectory summaries; Holm over exactly six tests")
    for row in tests:
        print("    %-18s Fieller vs %-15s diff=%s raw_p=%.5g Holm=%.5g n=%d" %
              (row["metric"], row["B"], fmt(row["difference_A_minus_B"]),
               row["raw_p"], row["holm_p"], row["n_trajectories"]))

    fallback = aggregate_fallback(obj["data"]["ucb_fieller"]["audits"])
    print("\n[6] scheduler-level delta-fallback audit (main-loop decisions; initialization separate)")
    print("    initialization pulls=%d; policy decisions=%d; candidate evaluations=%d" %
          (fallback["initialization_pulls"], fallback["policy_decisions"],
           fallback["candidate_evaluations"]))
    print("    candidate fallback=%d (%.4f); selected-arm fallback=%d (%.4f)" %
          (fallback["fallback_candidate_evaluations"], fallback["candidate_fallback_rate"],
           fallback["selected_fallback_decisions"], fallback["selected_fallback_rate"]))
    print("    candidate reasons n<2/A-only/D-only/A&D = %d/%d/%d/%d" %
          (fallback["fallback_n_lt_2"], fallback["fallback_A_only"],
           fallback["fallback_D_only"], fallback["fallback_A_and_D"]))
    print("    selected reasons  n<2/A-only/D-only/A&D = %d/%d/%d/%d" %
          (fallback["selected_n_lt_2"], fallback["selected_A_only"],
           fallback["selected_D_only"], fallback["selected_A_and_D"]))
    print("    budget quartile candidate-fallback rates = %s" %
          "/".join("%.4f" % x for x in fallback["quartile_candidate_fallback_rate"]))
    print("    budget quartile selected-fallback rates  = %s" %
          "/".join("%.4f" % x for x in fallback["quartile_selected_fallback_rate"]))

    result = {
        "manifest": manifest,
        "scope": "fixed-arm ratios under frozen policies; independent-seed simulation holdout",
        "adaptive": json_ready(summaries["ucb_fieller"]["methods"]),
        "uniform": json_ready(summaries["uniform"]["methods"]),
        "confirmatory_tests": json_ready(tests),
        "scheduler_fallback": json_ready(fallback),
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("\n[7] frozen derived results written: %s" % RESULTS)
    print("DONE")


if __name__ == "__main__":
    main()
