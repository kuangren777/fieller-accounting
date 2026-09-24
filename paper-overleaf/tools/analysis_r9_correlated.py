#!/usr/bin/env python3
"""Outcome-correlated costs: the holdout environment and benchmark policy with a success pull
costing (1+gamma) times a failure pull, so r and c are dependent within an arm.

gamma = 0 is the frozen holdout design; this script never reads or writes the frozen cache.
New seeds 920000..920399 are shared across gamma values.  Every construction and accounting is
computed by analysis_r2.analyse, so each row is on the same footing as Table 1.

  python3 tools/analysis_r9_correlated.py      -> data/r9_correlated.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_r2 as A2  # noqa: E402

sys.path.insert(0, A2.ARCH)
import expD_lib as L  # noqa: E402

K, MC, SIGMA, B_MULT, ENV_SEED, SEED_START = 24, 400, 0.35, 30, 1, 920000
GAMMAS = (-0.5, 0.5, 1.0)
KEYS = ("n", "sr", "sc", "sr2", "sc2", "src")


def env_for(gamma):
    return L.make_env(K=K, mu_range=(0.06, 0.30), cost_sigma=SIGMA,
                      cost_mean=5.0, gamma=gamma, nonstat="static", seed=ENV_SEED)


def true_ratio(env):
    ec = env["c"] * np.exp(SIGMA ** 2 / 2.0) * (1.0 + env["gamma"] * env["mu"])
    return env["mu"] / ec


def one(args):
    gamma, i = args
    env = env_for(gamma)
    budget = float(B_MULT * np.sum(env["base_c0"]) * np.exp(SIGMA ** 2 / 2.0))
    res = L.run(env, "ucb_fieller", budget, seed=SEED_START + i, pulls_log=True)
    costs = [[] for _ in range(K)]
    for _, arm, cost, _, _ in res["pulls_log"]:
        costs[arm].append(cost)
    return gamma, i, {k: np.asarray(res["arm_stats"][k], float) for k in KEYS}, \
        [np.sort(np.asarray(c, float)) for c in costs], float(res["reward"])


def oracle_check(env, rho, N=400_000, seed=929001):
    rng = np.random.default_rng(seed)
    est = np.empty(K)
    for a in range(K):
        rr = rng.random(N) < env["mu"][a]
        cc = rng.lognormal(np.log(env["c"][a]), SIGMA, N)
        cc = np.where(rr, cc * (1.0 + env["gamma"]), cc)
        est[a] = rr.sum() / cc.sum()
    return float(np.max(np.abs(est / rho - 1.0)))


def main():
    jobs = [(g, i) for g in GAMMAS for i in range(MC)]
    S = {g: {k: np.zeros((MC, K)) for k in KEYS} for g in GAMMAS}
    C = {g: np.empty((MC, K), dtype=object) for g in GAMMAS}
    R = {g: np.zeros(MC) for g in GAMMAS}
    with Pool(min(40, os.cpu_count() or 1)) as pool:
        for g, i, st, cs, rew in pool.imap_unordered(one, jobs, chunksize=4):
            for k in KEYS:
                S[g][k][i] = st[k]
            for a in range(K):
                C[g][i, a] = cs[a]
            R[g][i] = rew
    out = {"seeds": [SEED_START, SEED_START + MC - 1], "MC": MC, "K": K, "gammas": list(GAMMAS),
           "design": "success pull costs (1+gamma) x failure pull; holdout environment; benchmark policy"}
    for g in GAMMAS:
        env = env_for(g)
        rho = true_ratio(env)
        err = oracle_check(env, rho)
        if err > 0.01:
            raise RuntimeError("analytic ratio oracle failed at gamma=%s: %.4f" % (g, err))
        # within-arm correlation of r and c, population value
        mu = env["mu"]
        m = np.exp(SIGMA ** 2 / 2.0)
        v = np.exp(SIGMA ** 2) * (np.exp(SIGMA ** 2) - 1.0)
        c0 = env["c"]
        ec = c0 * m * (1 + g * mu)
        erc = mu * c0 * m * (1 + g)
        ec2 = c0 ** 2 * (v + m ** 2) * (mu * (1 + g) ** 2 + 1 - mu)
        corr = (erc - mu * ec) / np.sqrt(mu * (1 - mu) * (ec2 - ec ** 2))
        res = A2.analyse("benchmark, gamma=%g" % g, S[g], rho, costs=C[g])
        res.pop("per_arm_fieller", None)
        res["oracle_max_rel_err"] = err
        res["corr_rc_range"] = [float(corr.min()), float(corr.max())]
        res["reward_mean"] = float(R[g].mean())
        out["gamma=%g" % g] = res
        print("gamma=%+.1f corr(r,c) in [%.3f, %.3f] oracle err %.4f" % (g, corr.min(), corr.max(), err))
        for name in A2.METHODS:
            r = res["methods"].get(name)
            if r is None:
                continue
            print("  %-14s nd rep=%.3f cond=%.3f | b rep=%.3f cond=%.3f | joint=%.3f set=%.3f commonF=%.3f S.5=%.3f" % (
                name, r["nondeg"]["report"][0], r["nondeg"]["conditional"][0], r["bounded"]["report"][0],
                r["bounded"]["conditional"][0], r["nondeg"]["joint"][0], r["set_coverage"][0], r["common_F"][0],
                r["score"]["nondeg"]["0.50"][0]))
        for t in res["confirmatory_tests"]:
            print("  test %-18s vs %-14s diff=%+.4f [%+.4f,%+.4f] Holm=%.3g" % (t["metric"], t["B"], *t["difference"], t["holm_p"]))
        print("  zero-outcome fraction %.3f" % res["zero_outcome"]["fraction"][0])
    path = os.path.join(A2.ROOT, "data", "r9_correlated.json")
    json.dump(out, open(path, "w"), indent=1, sort_keys=True, default=float)
    print("written", path)


if __name__ == "__main__":
    main()
