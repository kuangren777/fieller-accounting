#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICASSP R5: do better schedulers and better constructions remove the reportability problem?

Three studies, all on the anchor benchmark of R2/R4.

  (A) Regularised indices.  The benchmark index is an unregularised delta upper limit, which
      collapses to zero on an arm with two zero outcomes.  KL-UCB, an empirical-Bernstein UCB
      and a Laplace (+1/+1) smoothed delta index never collapse, so this is the direct test of
      whether the zero-outcome state is an artefact of one unregularised index.

  (B) Reportability-constrained two-stage design.  Stage one spends a fraction f of the budget
      uniformly, stage two brings every arm up to the sample size the factorised model demands,
      n0 = ceil(log eta / log(1 - mu_tilde)) with the Jeffreys-shrunk rate, capped.  Sweeping f
      traces a frontier of report rate against reward, the design-side answer to the problem the
      accounting only labels.

  (C) Two constructions that always exist, on the frozen R2 holdout (no new sampling): the
      Jeffreys posterior credible interval and an anytime-valid confidence sequence.
"""
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "..", "archive", "tmp", "experiments")
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402
sys.path.insert(0, HERE)
import ratio_ci as RC  # noqa: E402
from icassp_r4_policy import make_env, budget, tq_table, fieller_upper_fast, boot_ratio, boot_mean, true_ratio, K, MC, ALPHA  # noqa: E402

OUT = os.path.join(HERE, "icassp_r5_designs_results.json")
SEED0 = 960000
ETA = 0.05
NCAP = 40
NPROC = max(4, min(40, os.cpu_count() or 4))
NEW_POLICIES = ("kl_ucb", "bernstein_ucb", "laplace_delta")
FSTAGE = (0.2, 0.4, 0.6, 0.8)


def kl_bern(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_ucb_mu(mu_hat, n, t, iters=24):
    """Largest q with n*kl(mu_hat, q) <= log t + 3 log log t, by bisection."""
    rhs = np.log(max(t, 2.0)) + 3.0 * np.log(np.log(max(t, 3.0)))
    lo = np.clip(mu_hat, 0.0, 1.0); hi = np.ones_like(mu_hat)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = n * kl_bern(mu_hat, mid) <= rhs
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return lo


def run_traj(args):
    """One trajectory of a regularised-index policy or of the two-stage design."""
    policy, env_seed, mult, seed, frac = args
    env = make_env(env_seed)
    B = budget(env, mult)
    rng = np.random.default_rng(seed)
    n = np.zeros(K); sr = np.zeros(K); sc = np.zeros(K); sr2 = np.zeros(K); sc2 = np.zeros(K); src = np.zeros(K)
    total_cost = 0.0; total_reward = 0.0

    def pull(a):
        nonlocal total_cost, total_reward
        c, rew, _ = L.sample_arm(env, a, min(total_cost / B, 1.0), rng)
        rew = float(rew)
        n[a] += 1; sr[a] += rew; sc[a] += c; sr2[a] += rew * rew; sc2[a] += c * c; src[a] += rew * c
        total_cost += c; total_reward += rew

    for a in range(K):
        pull(a)
    if policy == "two_stage":
        stage1 = frac * B
        while total_cost < stage1:
            pull(int(rng.integers(0, K)))
        while total_cost < B:
            mut = (sr + 0.5) / (n + 1.0)
            need = np.ceil(np.log(ETA) / np.log(np.maximum(1.0 - mut, 1e-9)))
            need = np.minimum(need, NCAP)
            deficit = need - n
            if np.all(deficit <= 0):
                pull(int(np.argmin(n)))
            else:
                pull(int(np.argmax(deficit)))
    else:
        tq = tq_table(ALPHA)
        while total_cost < B:
            t = n.sum()
            ch = np.maximum(sc / np.maximum(n, 1.0), 1e-12)
            mu_hat = sr / np.maximum(n, 1.0)
            if policy == "kl_ucb":
                u = kl_ucb_mu(mu_hat, np.maximum(n, 1.0), t)
            elif policy == "bernstein_ucb":
                v = mu_hat * (1.0 - mu_hat)
                lt = np.log(max(t, 2.0))
                u = np.minimum(mu_hat + np.sqrt(2.0 * v * lt / np.maximum(n, 1.0)) + 3.0 * lt / np.maximum(n, 1.0), 1.0)
            elif policy == "laplace_delta":
                mut = (sr + 1.0) / (n + 2.0)
                lt = np.log(max(t, 2.0))
                u = np.minimum(mut + np.sqrt(2.0 * mut * (1.0 - mut) * lt / np.maximum(n, 1.0)), 1.0)
            else:
                raise ValueError(policy)
            a = int(np.argmax(u / ch))
            pull(a)
    S = {"n": n, "sr": sr, "sc": sc, "sr2": sr2, "sc2": sc2, "src": src}
    return {"stats": S, "reward": total_reward}


def stack(trajs):
    return {k: np.stack([t["stats"][k] for t in trajs]) for k in ("n", "sr", "sc", "sr2", "sc2", "src")}


def analyse(trajs, env, label, seed=0):
    S = stack(trajs)
    M = RC.moments(S)
    rho = true_ratio(env)
    th = np.broadcast_to(rho, S["n"].shape)
    mc = S["n"].shape[0]
    allK = np.full(mc, K, float)
    sets = RC.all_sets(M, ALPHA)
    out = {"label": label, "MC": mc, "methods": {}}
    for i, (name, (lo, hi, kind)) in enumerate(sets.items()):
        nondeg = kind == "interval"
        bounded = np.isfinite(lo) & np.isfinite(hi)
        hit = bounded & (th >= lo) & (th <= hi)
        line = kind == "line"
        out["methods"][name] = {
            "report": boot_ratio(nondeg.sum(1), allK, seed + 10 + i),
            "conditional": boot_ratio((hit & nondeg).sum(1), nondeg.sum(1), seed + 20 + i),
            "set_coverage": boot_ratio((hit | line).sum(1), allK, seed + 30 + i),
            "width_median": float(np.nanmedian(np.where(nondeg, hi - lo, np.nan))),
        }
    n = S["n"]
    zero = (S["sr"] == 0) & (n >= 2)
    out["zero_outcome_frac"] = boot_ratio(zero.sum(1), allK, seed + 52)
    out["reward"] = boot_mean(np.array([t["reward"] for t in trajs], float), seed + 60)
    out["median_n"] = float(np.median(n))
    out["frac_n_ge_10"] = float(np.mean(n >= 10))
    return out


def always_exist_on_holdout():
    """Study (C): the two always-existing constructions on the frozen R2 collection."""
    d = pickle.load(open(os.path.join(ARCH, "icassp_r2_holdout_cache.pkl"), "rb"))
    rho = np.asarray(d["rho"], float)
    out = {}
    for pol, key in (("adaptive", "ucb_fieller"), ("uniform", "uniform")):
        S = d["data"][key]["stats"]
        M = RC.moments(S)
        th = np.broadcast_to(rho, M["n"].shape)
        mc = M["n"].shape[0]
        allK = np.full(mc, K, float)
        rec = {}
        tmax = float(rho.max()) * 6.0
        for i, (name, fn) in enumerate((("Bayes", lambda M: RC.bayes_set(M, ALPHA)),
                                        ("conf. seq.", lambda M: RC.cs_set(M, ALPHA, theta_max=tmax)))):
            lo, hi, kind = fn(M)
            nondeg = kind == "interval"
            bounded = np.isfinite(lo) & np.isfinite(hi)
            hit = bounded & (th >= lo) & (th <= hi)
            line = kind == "line"
            rec[name] = {
                "kinds": {k: float(np.mean(kind == k)) for k in ("interval", "point", "empty", "line", "n<2")},
                "report": boot_ratio(nondeg.sum(1), allK, 400 + i),
                "conditional": boot_ratio((hit & nondeg).sum(1), nondeg.sum(1), 410 + i),
                "set_coverage": boot_ratio((hit | line).sum(1), allK, 420 + i),
                "width_median": float(np.nanmedian(np.where(nondeg, hi - lo, np.nan))),
                "point_share": float(np.mean(kind == "point")),
            }
        out[pol] = rec
    return out


def main():
    env = make_env(1)
    res = {"K": K, "MC": MC, "alpha": ALPHA, "eta": ETA, "ncap": NCAP, "budget_mult": 30}
    with Pool(NPROC) as pool:
        for pol in NEW_POLICIES:
            args = [(pol, 1, 30, SEED0 + i, 0.0) for i in range(MC)]
            res[pol] = analyse(pool.map(run_traj, args, chunksize=8), env, pol, seed=hash(pol) % 1000)
            m = res[pol]
            print("%-14s zero=%.3f report=%.3f cond=%.3f set=%.3f reward=%.0f" % (
                pol, m["zero_outcome_frac"][0], m["methods"]["Fieller"]["report"][0],
                m["methods"]["Fieller"]["conditional"][0], m["methods"]["Fieller"]["set_coverage"][0], m["reward"][0]))
        res["two_stage"] = {}
        for f in FSTAGE:
            args = [("two_stage", 1, 30, SEED0 + 50000 + int(f * 100) * 1000 + i, f) for i in range(MC)]
            r = analyse(pool.map(run_traj, args, chunksize=8), env, "two_stage f=%.1f" % f, seed=int(f * 100))
            res["two_stage"]["%.1f" % f] = r
            print("two-stage f=%.1f zero=%.3f report=%.3f cond=%.3f set=%.3f reward=%.0f med_n=%.0f" % (
                f, r["zero_outcome_frac"][0], r["methods"]["Fieller"]["report"][0],
                r["methods"]["Fieller"]["conditional"][0], r["methods"]["Fieller"]["set_coverage"][0],
                r["reward"][0], r["median_n"]))
    res["always_exist"] = always_exist_on_holdout()
    for pol, rec in res["always_exist"].items():
        for name, m in rec.items():
            print("%-9s %-11s report=%.3f cond=%.3f set=%.3f width=%.3f kinds=%s" % (
                pol, name, m["report"][0], m["conditional"][0], m["set_coverage"][0], m["width_median"],
                {k: round(v, 3) for k, v in m["kinds"].items() if v > 0}))
    json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
    print("written", OUT)


if __name__ == "__main__":
    main()
