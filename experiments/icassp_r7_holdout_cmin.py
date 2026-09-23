#!/usr/bin/env python3
"""Replay the frozen R2 holdout to recover each arm's observed costs.

The frozen cache stores per-arm moments only, and a positivity-respecting lower
bound on E[c] needs the arm's order statistics.  This replays the adaptive
policy with the frozen environment and the frozen seeds 910000..910399, reads the
sorted costs from the pull log, and refuses to write anything unless every
replayed trajectory reproduces the cached n, sum of outcomes and sum of costs.
The frozen script and cache are read, never modified.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.abspath(os.path.join(HERE, "..", "archive", "tmp", "experiments"))
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402

K, MC, SIGMA, B_MULT, ENV_SEED, SEED_START = 24, 400, 0.35, 30, 1, 910000
CACHE = os.path.join(ARCH, "icassp_r2_holdout_cache.pkl")
OUT = os.path.join(HERE, "icassp_r7_holdout_cmin.pkl")

ENV = L.make_env(K=K, mu_range=(0.06, 0.30), cost_sigma=SIGMA,
                 cost_mean=5.0, gamma=0.0, nonstat="static", seed=ENV_SEED)
BUDGET = float(B_MULT * np.sum(ENV["base_c0"]) * np.exp(SIGMA ** 2 / 2.0))


def replay(i):
    res = L.run(ENV, "ucb_fieller", BUDGET, seed=SEED_START + i, pulls_log=True)
    n = np.zeros(K)
    sc = np.zeros(K)
    sr = np.zeros(K)
    costs = [[] for _ in range(K)]
    for _, arm, cost, outcome, _ in res["pulls_log"]:
        n[arm] += 1
        sc[arm] += cost
        sr[arm] += float(outcome)
        costs[arm].append(cost)
    return i, n, sr, sc, [np.sort(np.asarray(c, float)) for c in costs]


def main():
    frozen = pickle.load(open(CACHE, "rb"))["data"]["ucb_fieller"]["stats"]
    costs = np.empty((MC, K), dtype=object)
    with Pool(min(40, os.cpu_count() or 1)) as pool:
        for i, n, sr, sc, cs in pool.imap_unordered(replay, range(MC)):
            for key, got in (("n", n), ("sr", sr), ("sc", sc)):
                if not np.allclose(got, frozen[key][i], rtol=0, atol=1e-9):
                    raise RuntimeError("trajectory %d does not reproduce cached %s" % (i, key))
            for a in range(K):
                costs[i, a] = cs[a]
    pickle.dump({"source": "replay of icassp_r2_holdout.py seeds %d..%d" % (SEED_START, SEED_START + MC - 1),
                 "verified_against": "n, sr, sc of icassp_r2_holdout_cache.pkl, atol 1e-9",
                 "sorted_costs": costs}, open(OUT, "wb"), protocol=4)
    print("all %d trajectories reproduce the frozen cache; wrote %s" % (MC, OUT))


if __name__ == "__main__":
    main()
