#!/usr/bin/env python3
"""Paired reward comparison of the delta-JV fallback against the delta fallback.

Block A of icassp_r4_policy.py runs both policies on the same trajectory seeds SEED0+i,
so trajectory i of one policy is paired with trajectory i of the other.

  python3 tools/analysis_r9_paired_reward.py  -> data/r9_paired_reward.json
"""
import json
import os
import pickle

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.abspath(os.path.join(ROOT, "..", "experiments", "icassp_r4_policy_cache.pkl"))

d = pickle.load(open(CACHE, "rb"))
x = np.array([t["reward"] for t in d[("A", "ucb_fieller_jv", 1, 30)]])
y = np.array([t["reward"] for t in d[("A", "ucb_fieller_delta", 1, 30)]])
diff = x - y
rng = np.random.default_rng(61)
boot = np.array([diff[rng.integers(0, diff.size, diff.size)].mean() for _ in range(4000)])
out = {"n_pairs": int(diff.size), "mean_jv": float(x.mean()), "mean_delta": float(y.mean()),
       "paired_mean_gain": [float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
       "wilcoxon_p": float(wilcoxon(x, y).pvalue)}
json.dump(out, open(os.path.join(ROOT, "data", "r9_paired_reward.json"), "w"), indent=1)
print(out)
