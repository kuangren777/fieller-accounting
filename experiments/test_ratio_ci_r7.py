#!/usr/bin/env python3
"""Checks for the three constructions added in response to review: Fieller-JV, MOVER-R and the
positivity-bounded set.  Run: python3 test_ratio_ci_r7.py"""
import numpy as np
from scipy.stats import beta as beta_dist

import ratio_ci as RC


def moments_from(costs, outcomes):
    c = np.asarray(costs, float)
    r = np.asarray(outcomes, float)
    S = {"n": np.array([c.size], float), "sr": np.array([r.sum()]), "sc": np.array([c.sum()]),
         "sr2": np.array([(r * r).sum()]), "sc2": np.array([(c * c).sum()]), "src": np.array([(r * c).sum()])}
    return RC.moments(S)


def test_anderson_two_pulls_is_one_minus_eps_times_min():
    alpha = 0.05
    lb = RC.anderson_lower([3.0, 7.0], alpha)
    eps = np.sqrt(np.log(1 / alpha) / 4.0)
    assert abs(lb - (1 - eps) * 3.0) < 1e-12, lb


def test_anderson_is_a_valid_lower_bound():
    rng = np.random.default_rng(0)
    alpha, n, reps = 0.05, 5, 20000
    x = np.sort(rng.lognormal(1.5, 0.35, size=(reps, n)), axis=1)
    mean = np.exp(1.5 + 0.35 ** 2 / 2)
    miss = np.mean([RC.anderson_lower(row, alpha) > mean for row in x])
    assert miss <= alpha, miss


def test_fieller_jv_zero_outcomes_is_symmetric_about_zero():
    M = moments_from([4.0, 4.4], [0, 0])
    lo, hi, kind = RC.fieller_jv_set(M, 0.10)
    assert kind[0] == "interval" and abs(lo[0] + hi[0]) < 1e-12 and hi[0] > 0, (lo, hi, kind)


def test_fieller_jv_keeps_the_denominator_condition():
    M = moments_from([1.0, 9.0], [0, 0])   # CV far above sqrt(2)/t: no bounded set
    assert RC.fieller_jv_set(M, 0.10)[2][0] == "line"


def test_mover_r_zero_outcomes_starts_at_zero():
    M = moments_from([4.0, 4.4, 4.2, 3.9, 4.1], [0, 0, 0, 0, 0])
    lo, hi, kind = RC.mover_r_set(M, 0.10)
    assert kind[0] == "interval" and abs(lo[0]) < 1e-12 and hi[0] > 0, (lo, hi, kind)


def test_mover_r_abstains_when_cost_interval_reaches_zero():
    M = moments_from([1.0, 9.0], [0, 1])
    assert RC.mover_r_set(M, 0.10)[2][0] == "line"


def test_positivity_is_bounded_where_fieller_is_not():
    M = moments_from([1.0, 9.0], [0, 0])
    costs = np.empty(1, dtype=object)
    costs[0] = np.array([1.0, 9.0])
    lo, hi, kind = RC.positivity_set(M, costs, 0.10)
    assert RC.fieller_set(M, 0.10)[2][0] == "line"
    assert kind[0] == "interval" and lo[0] == 0.0 and np.isfinite(hi[0]) and hi[0] > 0


def test_positivity_covers_at_nominal():
    rng = np.random.default_rng(1)
    alpha, n, reps, mu = 0.10, 2, 20000, 0.1
    c = np.sort(rng.lognormal(1.5, 0.35, size=(reps, n)), axis=1)
    r = rng.random((reps, n)) < mu
    theta = mu / np.exp(1.5 + 0.35 ** 2 / 2)
    s = r.sum(1)
    rhi = beta_dist.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5)
    clo = np.array([RC.anderson_lower(row, alpha / 2) for row in c])
    cover = np.mean(theta <= rhi / clo)
    assert cover >= 1 - alpha, cover


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("%d/%d passed" % (len(tests), len(tests)))
