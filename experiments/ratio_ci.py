#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared interval constructions for the ratio theta = E[r]/E[c] with explicit degenerate-set
classification.  All constructions use the unbiased (n-1) variance and the Student-t quantile
t_{n-1,1-alpha/2} unless `unbiased=False` is requested (the engine's MLE convention).

Set kinds returned by `fieller_set`:
  'interval'  A>0, D>0                bounded non-degenerate interval
  'point'     A>0, D<=0, B=C=0        the singleton {0} (all outcomes zero)
  'empty'     A>0, D<0, not point     empty set
  'line'      A<=0                    unbounded (two rays or the whole real line)
  'n<2'
Delta kinds: 'interval' or 'point' (zero width, all outcomes zero) or 'n<2'.
"""
import numpy as np
from scipy.stats import t as tdist
from scipy.stats import beta as beta_dist

EPS = 1e-12


def moments(S, unbiased=True):
    n = np.asarray(S["n"], float)
    nn = np.maximum(n, 1.0)
    rh = S["sr"] / nn
    ch = S["sc"] / nn
    srr = S["sr2"] / nn - rh ** 2
    scc = S["sc2"] / nn - ch ** 2
    src = S["src"] / nn - rh * ch
    if unbiased:
        f = nn / np.maximum(nn - 1.0, 1.0)
        srr, scc, src = srr * f, scc * f, src * f
    return {"n": n, "rh": rh, "ch": ch, "srr": np.maximum(srr, 0.0), "scc": np.maximum(scc, 0.0), "src": src, "sr": S["sr"]}


def fieller_set(M, alpha=0.10):
    n = M["n"]
    ns = np.maximum(n, 2.0)
    t = tdist.ppf(1 - alpha / 2, ns - 1.0)
    A = M["ch"] ** 2 - t ** 2 * M["scc"] / ns
    B = -2.0 * (M["ch"] * M["rh"] - t ** 2 * M["src"] / ns)
    C = M["rh"] ** 2 - t ** 2 * M["srr"] / ns
    D = B ** 2 - 4.0 * A * C
    kind = np.full(n.shape, "line", dtype=object)
    lo = np.full(n.shape, -np.inf)
    hi = np.full(n.shape, np.inf)
    interval = (A > 0) & (D > 0) & (n >= 2)
    with np.errstate(all="ignore"):
        root = np.sqrt(np.maximum(D, 0.0))
        lo[interval] = ((-B - root) / (2.0 * A))[interval]
        hi[interval] = ((-B + root) / (2.0 * A))[interval]
    kind[interval] = "interval"
    point = (A > 0) & (D <= 0) & (np.abs(B) < EPS) & (np.abs(C) < EPS) & (n >= 2)
    kind[point] = "point"
    lo[point] = 0.0
    hi[point] = 0.0
    empty = (A > 0) & (D <= 0) & ~point & (n >= 2)
    kind[empty] = "empty"
    lo[empty] = np.nan
    hi[empty] = np.nan
    kind[n < 2] = "n<2"
    return lo, hi, kind, {"A": A, "B": B, "C": C, "D": D}


def delta_set(M, alpha=0.10, srr=None, src=None):
    n = M["n"]
    ns = np.maximum(n, 2.0)
    t = tdist.ppf(1 - alpha / 2, ns - 1.0)
    srr = M["srr"] if srr is None else srr
    src = M["src"] if src is None else src
    rh, ch = M["rh"], np.maximum(M["ch"], EPS)
    var = (srr / ch ** 2 + rh ** 2 * M["scc"] / ch ** 4 - 2.0 * rh * src / ch ** 3) / ns
    var = np.maximum(var, 0.0)
    lo = rh / ch - t * np.sqrt(var)
    hi = rh / ch + t * np.sqrt(var)
    kind = np.where(hi - lo <= EPS, "point", "interval").astype(object)
    kind[n < 2] = "n<2"
    lo[n < 2] = -np.inf
    hi[n < 2] = np.inf
    return lo, hi, kind


def delta_jv_set(M, alpha=0.10):
    """Delta with the Jeffreys-shrunk outcome variance p~(1-p~), p~=(s+1/2)/(n+1); the covariance keeps
    the sample correlation and is rescaled to the new variance."""
    n = M["n"]
    p = (M["sr"] + 0.5) / (n + 1.0)
    srr = p * (1.0 - p)
    den = np.sqrt(M["srr"] * M["scc"])
    corr = np.divide(M["src"], np.maximum(den, 1e-15), out=np.zeros_like(den), where=den > 1e-15)
    corr = np.clip(corr, -1.0, 1.0)
    src = corr * np.sqrt(srr * M["scc"])
    return delta_set(M, alpha, srr=srr, src=src)


def fieller_jv_set(M, alpha=0.10):
    """Fieller with the Jeffreys-shrunk outcome variance of delta-JV, so the numerator never has
    zero variance.  The covariance keeps the sample correlation, taken as zero when it is undefined."""
    n = M["n"]
    p = (M["sr"] + 0.5) / (n + 1.0)
    srr = p * (1.0 - p)
    den = np.sqrt(M["srr"] * M["scc"])
    corr = np.clip(np.divide(M["src"], np.maximum(den, 1e-15), out=np.zeros_like(den), where=den > 1e-15), -1.0, 1.0)
    M2 = dict(M, srr=srr, src=corr * np.sqrt(srr * M["scc"]))
    lo, hi, kind, _ = fieller_set(M2, alpha)
    return lo, hi, kind


def mover_r_set(M, alpha=0.10):
    """MOVER-R for a ratio of two independently estimated means: a Wilson interval for mu and a
    Student-t interval for E[c], each at level 1-alpha, combined by the method of variance estimates
    recovery.  No bounded set exists when the cost interval reaches zero."""
    n = M["n"]
    nn = np.maximum(n, 2.0)
    z = tdist.ppf(1.0 - alpha / 2.0, 1e9)
    ph = M["rh"]
    cen = (ph + z * z / (2 * nn)) / (1 + z * z / nn)
    half = z * np.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
    l1, u1 = np.maximum(cen - half, 0.0), np.minimum(cen + half, 1.0)
    q = tdist.ppf(1.0 - alpha / 2.0, nn - 1.0)
    h = q * np.sqrt(M["scc"] / nn)
    b = M["ch"]
    l2, u2 = b - h, b + h
    lo = np.full(n.shape, -np.inf)
    hi = np.full(n.shape, np.inf)
    kind = np.full(n.shape, "line", dtype=object)
    good = (n >= 2) & (l2 > 0)
    with np.errstate(all="ignore"):
        ab = ph * b
        L = (ab - np.sqrt(np.maximum(ab ** 2 - l1 * u2 * (2 * ph - l1) * (2 * b - u2), 0.0))) / (u2 * (2 * b - u2))
        U = (ab + np.sqrt(np.maximum(ab ** 2 - u1 * l2 * (2 * ph - u1) * (2 * b - l2), 0.0))) / (l2 * (2 * b - l2))
    lo[good], hi[good] = L[good], U[good]
    kind[good] = "interval"
    kind[n < 2] = "n<2"
    return lo, hi, kind


def anderson_lower(sorted_costs, alpha):
    """Anderson's lower confidence bound for the mean of a non-negative variable, from a one-sided
    DKW band with Massart's constant: the mean is at least the integral of 1 - min(1, F_n + eps)."""
    x = np.asarray(sorted_costs, float)
    n = x.size
    if n < 1:
        return 0.0
    eps = np.sqrt(np.log(1.0 / alpha) / (2.0 * n))
    steps = np.diff(np.concatenate([[0.0], x]))
    return float(np.sum(steps * np.maximum(0.0, 1.0 - (np.arange(n) / n + eps))))


def positivity_set(M, sorted_costs, alpha=0.10):
    """A set that uses only c > 0: [0, r_hi / c_lo] with r_hi a one-sided Jeffreys upper limit for mu
    and c_lo Anderson's lower limit for E[c], each at one-sided level alpha/2.  It is bounded on every
    unit with two or more pulls, and its coverage is at least 1 - alpha by the union bound."""
    n = M["n"]
    nn = np.maximum(n, 2.0)
    s = M["sr"]
    rhi = beta_dist.ppf(1.0 - alpha / 2.0, s + 0.5, nn - s + 0.5)
    clo = np.vectorize(lambda c: anderson_lower(c, alpha / 2.0), otypes=[float])(sorted_costs)
    lo = np.full(n.shape, -np.inf)
    hi = np.full(n.shape, np.inf)
    kind = np.full(n.shape, "line", dtype=object)
    good = (n >= 2) & (clo > 0)
    lo[good] = 0.0
    hi[good] = rhi[good] / clo[good]
    kind[good] = "interval"
    kind[n < 2] = "n<2"
    return lo, hi, kind


def jeffreys_cost_set(M, alpha=0.10):
    """Jeffreys interval for mu (tail alpha/4 each side) and Student-t interval for E[c] (tail alpha/4);
    ratio of endpoints when the cost lower limit is positive, otherwise no bounded set ('line')."""
    n = M["n"]
    lo = np.full(n.shape, -np.inf)
    hi = np.full(n.shape, np.inf)
    kind = np.full(n.shape, "line", dtype=object)
    ok = n >= 2
    nn = np.maximum(n, 2.0)
    s = M["sr"]
    q = tdist.ppf(1.0 - alpha / 4.0, nn - 1.0)
    rlo = beta_dist.ppf(alpha / 4.0, s + 0.5, nn - s + 0.5)
    rhi = beta_dist.ppf(1.0 - alpha / 4.0, s + 0.5, nn - s + 0.5)
    cse = np.sqrt(M["scc"] / nn)
    clo = M["ch"] - q * cse
    chi = M["ch"] + q * cse
    good = ok & np.isfinite(rlo) & np.isfinite(rhi) & (clo > 0) & (chi > 0)
    lo[good] = rlo[good] / chi[good]
    hi[good] = rhi[good] / clo[good]
    kind[good] = "interval"
    kind[n < 2] = "n<2"
    return lo, hi, kind


def bayes_set(M, alpha=0.10, ndraw=4000, seed=20260914):
    """Posterior credible interval for theta = mu / E[c].

    mu | data ~ Beta(1/2 + s_r, 1/2 + n - s_r)  (Jeffreys prior on the Bernoulli outcome)
    E[c] | data ~ cbar + (s_c / sqrt(n)) T_{n-1}  (the reference posterior for a normal mean,
    a function of the same sufficient statistics the frequentist constructions use)

    The two are independent a posteriori, so equal-tailed quantiles of the sampled ratio give
    the credible interval.  It exists for every unit with n >= 2 and is never the point {0}.
    """
    n = M["n"]
    shp = np.shape(n)
    lo = np.full(shp, -np.inf); hi = np.full(shp, np.inf)
    kind = np.full(shp, "n<2", dtype=object)
    ok = n >= 2
    if not np.any(ok):
        return lo, hi, kind
    rng = np.random.default_rng(seed)
    nn = n[ok]; s = M["sr"][ok]; ch = M["ch"][ok]; scc = M["scc"][ok]
    mu = rng.beta(0.5 + s[:, None], 0.5 + nn[:, None] - s[:, None], size=(nn.size, ndraw))
    tdraw = rng.standard_t(np.maximum(nn - 1.0, 1.0)[:, None], size=(nn.size, ndraw))
    cd = ch[:, None] + np.sqrt(np.maximum(scc, 0.0) / nn)[:, None] * tdraw
    with np.errstate(divide="ignore", invalid="ignore"):
        th = np.where(cd > 0, mu / cd, np.inf)
    q = np.percentile(np.where(np.isfinite(th), th, np.inf), [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)], axis=1)
    ql, qh = q[0], q[1]
    good = np.isfinite(ql) & np.isfinite(qh)
    lo[ok] = np.where(good, ql, -np.inf)
    hi[ok] = np.where(good, qh, np.inf)
    kind[ok] = np.where(good, "interval", "line")
    return lo, hi, kind


def cs_set(M, alpha=0.10, theta_max=1.0, ngrid=2001, brange=1.0):
    """Anytime-valid confidence sequence for theta, inverted on a grid.

    For a fixed theta the centred variable x_i = r_i - theta c_i has mean zero, sample mean
    rbar - theta cbar and sample variance s_rr - 2 theta s_rc + theta^2 s_cc, both available in
    closed form from the same sufficient statistics.  The empirical-Bernstein stitched boundary
    of Howard et al. is used, radius = sqrt(2 v ell / n) + 3 b ell / n with
    ell = log(2/alpha) + log log(e n), which holds simultaneously over all sample sizes, so the
    set is valid under the data-dependent stopping the scheduler induces.
    """
    n = np.asarray(M["n"], float)
    shp = np.shape(n)
    lo = np.full(shp, -np.inf); hi = np.full(shp, np.inf)
    kind = np.full(shp, "n<2", dtype=object)
    ok = n >= 2
    if not np.any(ok):
        return lo, hi, kind
    grid = np.linspace(0.0, theta_max, ngrid)
    nn = n[ok][:, None]; rh = M["rh"][ok][:, None]; ch = M["ch"][ok][:, None]
    srr = M["srr"][ok][:, None]; scc = M["scc"][ok][:, None]; src = M["src"][ok][:, None]
    ell = np.log(2.0 / alpha) + np.log(np.log(np.e * nn))
    g = grid[None, :]
    centre = rh - g * ch
    v = np.maximum(srr - 2.0 * g * src + g * g * scc, 0.0)
    b = brange + g * (ch + 3.0 * np.sqrt(np.maximum(scc, 0.0)))
    rad = np.sqrt(2.0 * v * ell / nn) + 3.0 * b * ell / nn
    acc = np.abs(centre) <= rad
    any_acc = acc.any(1)
    idx = np.arange(grid.size)[None, :]
    first = np.where(acc, idx, grid.size).min(1)
    last = np.where(acc, idx, -1).max(1)
    l = np.where(any_acc, grid[np.clip(first, 0, grid.size - 1)], np.nan)
    h = np.where(any_acc, grid[np.clip(last, 0, grid.size - 1)], np.nan)
    unbounded = any_acc & (last == grid.size - 1)
    k = np.where(~any_acc, "empty", np.where(unbounded, "line", "interval")).astype(object)
    l = np.where(k == "interval", l, np.where(k == "line", -np.inf, np.nan))
    h = np.where(k == "interval", h, np.where(k == "line", np.inf, np.nan))
    lo[ok] = l; hi[ok] = h; kind[ok] = k
    return lo, hi, kind


def all_sets(M, alpha=0.10):
    lf, hf, kf, _ = fieller_set(M, alpha)
    ld, hd, kd = delta_set(M, alpha)
    lj, hj, kj = delta_jv_set(M, alpha)
    lc, hc, kc = jeffreys_cost_set(M, alpha)
    return {"Fieller": (lf, hf, kf), "raw delta": (ld, hd, kd), "delta-JV": (lj, hj, kj), "Jeffreys+cost": (lc, hc, kc)}


def interval_score(lo, hi, theta, alpha=0.10):
    """Ordinary interval score on bounded sets (points included); +inf otherwise."""
    fin = np.isfinite(lo) & np.isfinite(hi)
    out = np.full(np.shape(theta), np.inf)
    out[fin] = (hi[fin] - lo[fin]) + (2.0 / alpha) * (np.maximum(lo[fin] - theta[fin], 0.0) + np.maximum(theta[fin] - hi[fin], 0.0))
    return out
