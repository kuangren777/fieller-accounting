#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ICASSP R4: policy-level experiments on the anchor benchmark.

New sampling (independent of the frozen R2 collection) with a compact re-implementation
of the trajectory loop that keeps expD_lib's environment, sampler, Fieller and delta
constructions, and adds

  (A) fallback ablation at budget 30x: the adaptive index is the Fieller upper limit;
      when Fieller is not a bounded interval the index falls back to (i) the raw delta
      upper limit [the benchmark policy], (ii) the delta-JV upper limit, (iii) +inf
      (forced exploration until Fieller reports); plus a delta-JV UCB and the uniform
      control.  Reports per method, cumulative reward, and the frozen two-zero mass.
  (B) budget sweep {5,10,30,100}x for the benchmark policy and the uniform control.
  (C) environment seeds {2,3,4} at 30x (new arm parameters mu_a, c_a).
  (D) nominal-level sweep alpha in {0.20,0.10,0.05,0.01} on the frozen R2 collection
      (no new sampling): report / conditional / joint per method, common-F delta.
  (E) denominator t-interval coverage for E[c] under the nine landscape cost
      distributions at fixed n (iid), the diagnostic for the heavy-tail failing cells.
  (F) closed-form prediction of the frozen mass: an arm is abandoned by the benchmark
      policy exactly when its first two outcomes are zero, so the predicted fraction of
      arm-trajectory units with n=2 is K^{-1} sum_a (1-mu_a)^2, compared with the
      observed fraction in every run of (A)-(C).
  (G) selected-arm inference with data splitting: select on odd-indexed pulls, report
      from even-indexed pulls, against naive all-data selection and Bonferroni.

Cluster bootstrap over independent trajectories for every proportion.  No claim of
adaptive validity is made.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(HERE, "..", "archive", "tmp", "experiments")
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402
import icassp_general as G  # noqa: E402  (cost_params / draw_cost_vec only)
sys.path.insert(0, HERE)
import ratio_ci as RC  # noqa: E402

OUT_JSON = os.path.join(HERE, "icassp_r4_policy_results.json")
CACHE = os.path.join(HERE, "icassp_r4_policy_cache.pkl")
R2CACHE = os.path.join(ARCH, "icassp_r2_holdout_cache.pkl")
K = 24
MC = 400
MC_ENV = 100
SEED0 = 940000
ALPHA = 0.10
SIGMA = 0.35
B_BOOT = 2000
NPROC = max(4, min(40, os.cpu_count() or 4))
NMAX = 60000
TQ = {}


def tq_table(alpha):
    if alpha not in TQ:
        n = np.arange(0, NMAX + 1, dtype=float)
        q = tdist.ppf(1 - alpha / 2, np.maximum(n, 2.0) - 1.0)
        TQ[alpha] = q
    return TQ[alpha]


def moments(S):
    """Unbiased-variance moments (shared convention, ratio_ci)."""
    return RC.moments(S, unbiased=True)


def _nondeg(fn):
    def wrapped(M, alpha=ALPHA):
        lo, hi, kind = fn(M, alpha)[:3]
        lo = np.where(kind == "interval", lo, -np.inf)
        hi = np.where(kind == "interval", hi, np.inf)
        return lo, hi
    return wrapped


fieller = _nondeg(lambda M, alpha=ALPHA: RC.fieller_set(M, alpha)[:3])
raw_delta = _nondeg(RC.delta_set)
delta_jv = _nondeg(RC.delta_jv_set)
jeffreys_cost = _nondeg(RC.jeffreys_cost_set)
METHODS = {"Fieller": fieller, "raw delta": raw_delta, "delta-JV": delta_jv, "Jeffreys+cost": jeffreys_cost}


# ---------------------------------------------------------------------------- trajectory
def make_env(env_seed):
    return L.make_env(K=K, cost_sigma=SIGMA, gamma=0.0, seed=env_seed)


def true_ratio(env):
    return env["mu"] / (env["c"] * np.exp(env["cost_sigma"] ** 2 / 2.0) * (1.0 + env["gamma"] * env["mu"]))


def budget(env, mult):
    base = env.get("base_c0", env["c"])
    return float(mult * np.sum(base) * np.exp(SIGMA ** 2 / 2.0))


def fieller_upper_fast(n, sr, sc, sr2, sc2, src, tq):
    """Vectorised Fieller / delta / delta-JV upper limits for the index computation."""
    nn = np.maximum(n, 1.0)
    rh = sr / nn
    ch = sc / nn
    srr = sr2 / nn - rh ** 2
    scc = sc2 / nn - ch ** 2
    srcv = src / nn - rh * ch
    ns = np.maximum(n, 2.0)
    t = tq[np.minimum(n, NMAX).astype(int)]
    A = ch ** 2 - t ** 2 * scc / ns
    Bq = -2.0 * (ch * rh - t ** 2 * srcv / ns)
    C = rh ** 2 - t ** 2 * srr / ns
    D = Bq ** 2 - 4.0 * A * C
    fin = (A > 0) & (D > 0) & (n >= 2)
    up_f = np.full(K, np.inf)
    with np.errstate(all="ignore"):
        up_f[fin] = ((-Bq + np.sqrt(np.maximum(D, 0.0))) / (2.0 * A))[fin]
        chs = np.maximum(ch, 1e-9)
        rhs = np.maximum(rh, 1e-9)
        var = (rhs / chs) ** 2 * (srr / rhs ** 2 + scc / chs ** 2 - 2.0 * srcv / (rhs * chs)) / ns
        up_d = rhs / chs + t * np.sqrt(np.maximum(var, 0.0))
        p = (sr + 0.5) / (n + 1.0)
        srr_jv = p * (1.0 - p)
        den = np.sqrt(np.maximum(srr, 0.0) * np.maximum(scc, 0.0))
        corr = np.where(den > 1e-15, srcv / np.maximum(den, 1e-15), 0.0)
        corr = np.clip(corr, -1.0, 1.0)
        src_jv = corr * np.sqrt(srr_jv * np.maximum(scc, 0.0))
        var_jv = (rhs / chs) ** 2 * (srr_jv / rhs ** 2 + scc / chs ** 2 - 2.0 * src_jv / (rhs * chs)) / ns
        up_jv = rhs / chs + t * np.sqrt(np.maximum(var_jv, 0.0))
    up_d = np.where(n < 2, np.inf, up_d)
    up_jv = np.where(n < 2, np.inf, up_jv)
    return up_f, up_d, up_jv, fin


LIB_POLICIES = ("ucb_bv1", "ucb_bv2", "kube_knapsack", "thompson_ratio", "eps_greedy_ratio", "greedy_ratio", "ucb_delta")


def run_traj_lib(args):
    """Trajectory through expD_lib.run for the library's standard budgeted policies (no interval fallback)."""
    policy, env_seed, mult, seed = args
    env = make_env(env_seed)
    B = budget(env, mult)
    res = L.run(env, policy, B, seed=seed)
    st = res["arm_stats"]
    S = {k: np.asarray(st[k], float) for k in ("n", "sr", "sc", "sr2", "sc2", "src")}
    zero = {k: np.zeros(K) for k in ("n", "sr", "sc", "sr2", "sc2", "src")}
    return {"stats": S, "half": [dict(S), dict(zero)], "reward": float(res["reward"]), "cost": float(res["cost"]),
            "fallback_selected": 0, "decisions": 0, "lib": True}


def fixed_n_calibration():
    """Fixed-n iid calibration of the implemented (MLE-variance) Fieller and delta sets against the
    unbiased-variance textbook versions, Bernoulli outcome x LogNormal cost, arm parameters of env 1."""
    env = make_env(1)
    rho = true_ratio(env)
    rng = np.random.default_rng(4242)
    R = 20000
    out = []
    for n in (2, 3, 5, 10, 30):
        rec = {"n": n}
        cov = {"engine_F": [], "textbook_F": [], "engine_D": [], "textbook_D": [], "report_F": [], "report_textbook_F": []}
        for a in range(K):
            r = (rng.random((R, n)) < env["mu"][a]).astype(float)
            c = rng.lognormal(np.log(env["c"][a]), SIGMA, (R, n))
            nn = np.full(R, float(n))
            rh = r.mean(1); ch = c.mean(1)
            srr = r.var(1); scc = c.var(1); src = ((r - rh[:, None]) * (c - ch[:, None])).mean(1)
            f = n / (n - 1.0)
            for tag, scale in (("engine", 1.0), ("textbook", f)):
                lo, hi = L.fieller_ci(rh, ch, srr * scale, scc * scale, src * scale, nn, alpha=ALPHA)
                fin = np.isfinite(lo) & np.isfinite(hi)
                cov[tag + "_F"].append(np.mean((rho[a] >= lo[fin]) & (rho[a] <= hi[fin])) if fin.any() else np.nan)
                cov["report_F" if tag == "engine" else "report_textbook_F"].append(fin.mean())
                lo, hi = L.delta_ci(rh, ch, srr * scale, scc * scale, src * scale, nn, alpha=ALPHA)
                cov[tag + "_D"].append(np.mean((rho[a] >= lo) & (rho[a] <= hi)))
        for k, v in cov.items():
            rec[k] = float(np.nanmean(v))
        out.append(rec)
    return out


def run_traj(args):
    policy, env_seed, mult, seed = args
    env = make_env(env_seed)
    B = budget(env, mult)
    tq = tq_table(ALPHA)
    rng = np.random.default_rng(seed)
    n = np.zeros(K); sr = np.zeros(K); sc = np.zeros(K); sr2 = np.zeros(K); sc2 = np.zeros(K); src = np.zeros(K)
    # odd / even split by pull index within each arm
    half = [{k: np.zeros(K) for k in ("n", "sr", "sc", "sr2", "sc2", "src")} for _ in range(2)]
    total_cost = 0.0; total_reward = 0.0; fallback_selected = 0; decisions = 0

    def pull(a):
        nonlocal total_cost, total_reward
        c, rew, _ = L.sample_arm(env, a, min(total_cost / B, 1.0), rng)
        rew = float(rew)
        h = half[int(n[a]) % 2]
        for d in (None, h):
            if d is None:
                n[a] += 1; sr[a] += rew; sc[a] += c; sr2[a] += rew * rew; sc2[a] += c * c; src[a] += rew * c
            else:
                d["n"][a] += 1; d["sr"][a] += rew; d["sc"][a] += c; d["sr2"][a] += rew * rew; d["sc2"][a] += c * c; d["src"][a] += rew * c
        total_cost += c; total_reward += rew

    for a in range(K):
        pull(a)
    while total_cost < B:
        if policy == "uniform":
            a = int(rng.integers(0, K))
        else:
            up_f, up_d, up_jv, fin = fieller_upper_fast(n, sr, sc, sr2, sc2, src, tq)
            if policy == "ucb_fieller_delta":
                idx = np.where(fin, up_f, up_d)
            elif policy == "ucb_fieller_jv":
                idx = np.where(fin, up_f, up_jv)
            elif policy == "ucb_fieller_inf":
                idx = np.where(fin, up_f, np.inf)
            elif policy == "ucb_jv":
                idx = up_jv
            else:
                raise ValueError(policy)
            idx = np.nan_to_num(idx, nan=1e9, posinf=1e9, neginf=-1e9)
            a = int(np.argmax(idx))
            decisions += 1
            if policy.startswith("ucb_fieller") and not fin[a]:
                fallback_selected += 1
        pull(a)
    S = {"n": n, "sr": sr, "sc": sc, "sr2": sr2, "sc2": sc2, "src": src}
    return {"stats": S, "half": half, "reward": total_reward, "cost": total_cost,
            "fallback_selected": fallback_selected, "decisions": decisions}


# ---------------------------------------------------------------------------- analysis
def boot_ratio(num, den, seed, B=B_BOOT):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() <= 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(B, num.size))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_mean(x, seed, B=B_BOOT):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stack(trajs, key="stats"):
    return {k: np.stack([t[key][k] for t in trajs]) for k in ("n", "sr", "sc", "sr2", "sc2", "src")}


def analyse_collection(trajs, env, label):
    S = stack(trajs)
    M = moments(S)
    rho = true_ratio(env)
    th = np.broadcast_to(rho, S["n"].shape)
    mc = S["n"].shape[0]
    allK = np.full(mc, K, float)
    out = {"label": label, "MC": mc, "methods": {}}
    lo_f, hi_f = fieller(M)
    fin_f = np.isfinite(lo_f) & np.isfinite(hi_f)
    for i, (name, fn) in enumerate(METHODS.items()):
        lo, hi = fn(M)
        fin = np.isfinite(lo) & np.isfinite(hi)
        hit = fin & (th >= lo) & (th <= hi)
        w = np.where(fin, hi - lo, np.nan)
        out["methods"][name] = {
            "report": boot_ratio(fin.sum(1), allK, 10 + i),
            "conditional": boot_ratio(hit.sum(1), fin.sum(1), 20 + i),
            "joint": boot_ratio(hit.sum(1), allK, 30 + i),
            "common_F": boot_ratio((hit & fin_f).sum(1), fin_f.sum(1), 40 + i),
            "width_median": float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan"),
        }
    sets = RC.all_sets(M, ALPHA)
    out["kinds"] = {name: {k: float(np.mean(kind == k)) for k in ("interval", "point", "empty", "line", "n<2")}
                    for name, (_, _, kind) in sets.items()}
    for name, (lo, hi, kind) in sets.items():
        bounded = np.isfinite(lo) & np.isfinite(hi)
        hit = bounded & (th >= lo) & (th <= hi)
        out["methods"][name]["bounded"] = {"report": boot_ratio(bounded.sum(1), allK, 90),
                                           "conditional": boot_ratio(hit.sum(1), bounded.sum(1), 91)}
    n = S["n"]
    zero = (S["sr"] == 0) & (n >= 2)
    out["zero_outcome_frac"] = boot_ratio(zero.sum(1), allK, 52)
    two_zero = (n == 2) & (S["sr"] == 0)
    out["frac_n2_zero"] = boot_ratio(two_zero.sum(1), allK, 50)
    out["frac_n2"] = boot_ratio((n == 2).sum(1), allK, 51)
    out["pred_frac_n2_zero"] = float(np.mean((1.0 - env["mu"]) ** 2))
    out["pred_report_upper"] = float(1.0 - np.mean((1.0 - env["mu"]) ** 2))
    out["median_n"] = float(np.median(n)); out["mean_n"] = float(n.mean())
    out["reward_mean"] = boot_mean([t["reward"] for t in trajs], 60)
    out["fallback_selected_rate"] = float(np.sum([t["fallback_selected"] for t in trajs]) /
                                          max(1, np.sum([t["decisions"] for t in trajs])))
    # selected arm: naive, Bonferroni, data split
    ratio = np.where(n >= 2, M["rh"] / np.maximum(M["ch"], 1e-12), -np.inf)
    sel = np.argmax(ratio, 1)
    rows = np.arange(mc)
    best = int(np.argmax(rho))
    out["selection"] = {"identification": boot_mean((sel == best).astype(float), 70)}
    for name in ("Fieller", "Jeffreys+cost"):
        fn = METHODS[name]
        for lvl, tag in ((ALPHA, "naive"), (ALPHA / K, "bonferroni")):
            lo, hi = fn(M, lvl)
            ls, hs = lo[rows, sel], hi[rows, sel]
            fin = np.isfinite(ls) & np.isfinite(hs)
            hit = fin & (rho[sel] >= ls) & (rho[sel] <= hs)
            out["selection"]["%s|%s" % (name, tag)] = {
                "report": boot_mean(fin.astype(float), 80), "conditional": boot_ratio(hit, fin, 81),
                "joint": boot_mean(hit.astype(float), 82),
                "width_median": float(np.nanmedian(np.where(fin, hs - ls, np.nan)))}
    # fixed true-best arm as the fixed-arm reference
    selb = np.full(mc, best, int)
    for name in ("Fieller", "raw delta", "Jeffreys+cost"):
        fn = METHODS[name]
        for lvl, tag in ((ALPHA, "naive"), (ALPHA / K, "bonferroni")):
            lo, hi = fn(M, lvl)
            ls, hs = lo[rows, selb], hi[rows, selb]
            fin = np.isfinite(ls) & np.isfinite(hs)
            hit = fin & (rho[selb] >= ls) & (rho[selb] <= hs)
            out["selection"]["truebest|%s|%s" % (name, tag)] = {
                "report": boot_mean(fin.astype(float), 90), "conditional": boot_ratio(hit, fin, 91),
                "joint": boot_mean(hit.astype(float), 92),
                "width_median": float(np.nanmedian(np.where(fin, hs - ls, np.nan))) if fin.any() else float("nan")}
    out["selection"]["truebest_n_median"] = float(np.median(n[:, best]))
    # data split: select on half 0 (odd pulls), report from half 1 (even pulls)
    if trajs[0].get("lib"):
        return out
    H0 = moments({k: np.stack([t["half"][0][k] for t in trajs]) for k in ("n", "sr", "sc", "sr2", "sc2", "src")})
    H1 = moments({k: np.stack([t["half"][1][k] for t in trajs]) for k in ("n", "sr", "sc", "sr2", "sc2", "src")})
    ratio0 = np.where(H0["n"] >= 1, H0["rh"] / np.maximum(H0["ch"], 1e-12), -np.inf)
    sel0 = np.argmax(ratio0, 1)
    out["selection"]["split_identification"] = boot_mean((sel0 == best).astype(float), 71)
    for name in ("Fieller", "Jeffreys+cost"):
        lo, hi = METHODS[name](H1, ALPHA)
        ls, hs = lo[rows, sel0], hi[rows, sel0]
        fin = np.isfinite(ls) & np.isfinite(hs)
        hit = fin & (rho[sel0] >= ls) & (rho[sel0] <= hs)
        out["selection"]["%s|split" % name] = {
            "report": boot_mean(fin.astype(float), 83), "conditional": boot_ratio(hit, fin, 84),
            "joint": boot_mean(hit.astype(float), 85),
            "width_median": float(np.nanmedian(np.where(fin, hs - ls, np.nan)))}
    return out


def alpha_sweep():
    d = pickle.load(open(R2CACHE, "rb"))
    rho = np.asarray(d["rho"], float)
    out = {}
    for pol in ("ucb_fieller", "uniform"):
        M = moments(d["data"][pol]["stats"])
        th = np.broadcast_to(rho, M["n"].shape)
        mc = M["n"].shape[0]
        allK = np.full(mc, K, float)
        out[pol] = {}
        for alpha in (0.20, 0.10, 0.05, 0.01):
            lo_f, hi_f = fieller(M, alpha)
            fin_f = np.isfinite(lo_f) & np.isfinite(hi_f)
            rec = {}
            for i, (name, fn) in enumerate(METHODS.items()):
                lo, hi = fn(M, alpha)
                fin = np.isfinite(lo) & np.isfinite(hi)
                hit = fin & (th >= lo) & (th <= hi)
                rec[name] = {"report": boot_ratio(fin.sum(1), allK, 100 + i),
                             "conditional": boot_ratio(hit.sum(1), fin.sum(1), 110 + i),
                             "joint": boot_ratio(hit.sum(1), allK, 120 + i),
                             "common_F": boot_ratio((hit & fin_f).sum(1), fin_f.sum(1), 130 + i)}
            out[pol]["%.2f" % alpha] = rec
    return out


def denominator_coverage():
    """Coverage of the cost-mean t interval (engine convention) under the landscape cost families."""
    rng = np.random.default_rng(777)
    m = 8.3506  # per-arm mean used in the landscape self-check (arm 0)
    fams = [("lognormal", 0.361, None), ("gamma", 0.361, None), ("pareto", 0.361, None), ("twopoint", 0.361, None),
            ("lognormal", 2.913, None), ("gamma", 2.913, None), ("pareto", 2.913, None), ("twopoint", 2.913, None),
            ("pareto", 0.361, 1.5)]
    out = []
    R = 20000
    for dist, cv, pa in fams:
        extra = {}
        if dist == "twopoint":
            extra["p"] = G.tp_p_for_cv(cv)
        if pa is not None:
            extra["alpha"] = pa
        par = G.cost_params(dist, m, cv, extra)
        rec = {"cost_dist": dist, "cv": cv, "pareto_alpha": pa, "n": {}}
        for n in (5, 10, 30, 100):
            X = G.draw_cost_vec(par, rng, (R, n))
            cb = X.mean(1)
            s2 = X.var(1)  # MLE
            t = tdist.ppf(1 - ALPHA / 2, n - 1)
            cov_engine = np.mean(np.abs(cb - par["E"]) <= t * np.sqrt(s2 / n))
            cov_unb = np.mean(np.abs(cb - par["E"]) <= t * np.sqrt(s2 / (n - 1)))
            rec["n"][str(n)] = {"engine": float(cov_engine), "unbiased": float(cov_unb)}
        out.append(rec)
    return out


def main():
    t0 = time.time()
    configs = []
    # (A) fallback ablation, env 1, 30x
    for pol in ("ucb_fieller_delta", "ucb_fieller_jv", "ucb_fieller_inf", "ucb_jv", "uniform"):
        configs.append(("A", pol, 1, 30))
    # (B) budget sweep
    for mult in (5, 10, 100):
        for pol in ("ucb_fieller_delta", "uniform"):
            configs.append(("B", pol, 1, mult))
    # (C) environment seeds
    for es in (2, 3, 4):
        for pol in ("ucb_fieller_delta", "uniform"):
            configs.append(("C", pol, es, 30))
    # (H) standard budgeted bandit policies from the library, no interval fallback
    for pol in LIB_POLICIES:
        configs.append(("H", pol, 1, 30))
    # (J) environment replication: 30 environment seeds, MC_ENV trajectories each, both policies
    for es in range(101, 131):
        for pol in ("ucb_fieller_delta", "uniform"):
            configs.append(("J", pol, es, 30))
    if os.path.exists(CACHE):
        raw = pickle.load(open(CACHE, "rb"))
    else:
        raw = {}
    todo = [c for c in configs if c not in raw]
    print("configs=%d todo=%d nproc=%d" % (len(configs), len(todo), NPROC))
    with Pool(NPROC) as pool:
        for c in todo:
            blk_, pol, es, mult = c
            mc_here = MC_ENV if blk_ == "J" else MC
            args = [(pol, es, mult, SEED0 + 1000 * (es if blk_ == "J" else 0) + i) for i in range(mc_here)]
            raw[c] = pool.map(run_traj_lib if pol in LIB_POLICIES else run_traj, args, chunksize=8)
            pickle.dump(raw, open(CACHE, "wb"))
            print("  done", c, "%.0fs" % (time.time() - t0), flush=True)
    results = {"MC": MC, "K": K, "alpha": ALPHA, "seed0": SEED0, "runs": {}}
    for c in configs:
        block, pol, es, mult = c
        env = make_env(es)
        key = "%s|%s|env%d|%dx" % (block, pol, es, mult)
        results["runs"][key] = analyse_collection(raw[c], env, key)
        results["runs"][key].update({"block": block, "policy": pol, "env_seed": es, "budget_mult": mult,
                                     "budget": budget(env, mult)})
        r = results["runs"][key]
        print("%-36s rep=%.3f cond=%.3f joint=%.3f | n2zero obs=%.3f pred=%.3f | reward=%.1f | dJV rep=%.3f cond=%.3f" % (
            key, r["methods"]["Fieller"]["report"][0], r["methods"]["Fieller"]["conditional"][0],
            r["methods"]["Fieller"]["joint"][0], r["frac_n2_zero"][0], r["pred_frac_n2_zero"], r["reward_mean"][0],
            r["methods"]["delta-JV"]["report"][0], r["methods"]["delta-JV"]["conditional"][0]))
        s = r["selection"]
        if "Fieller|split" not in s:
            continue
        print("   selection: ident=%.3f split-ident=%.3f | Fieller naive cond=%.3f bonf=%.3f split cond=%.3f (rep %.3f) width naive/split=%.4f/%.4f" % (
            s["identification"][0], s["split_identification"][0], s["Fieller|naive"]["conditional"][0],
            s["Fieller|bonferroni"]["conditional"][0], s["Fieller|split"]["conditional"][0], s["Fieller|split"]["report"][0],
            s["Fieller|naive"]["width_median"], s["Fieller|split"]["width_median"]))
    # environment-level replication summary: one number per environment, bootstrap over environments
    envrep = {}
    for pol in ("ucb_fieller_delta", "uniform"):
        keys = [k for k in results["runs"] if k.startswith("J|%s|" % pol)]
        per_env = {q: [] for q in ("report", "conditional", "joint", "delta_all", "common_F_gap", "zero_outcome_frac", "pred_frac", "obs_frac_n2zero",
                                  "truebest_cond", "truebest_report", "sel_cond", "reward", "jc_report", "jc_conditional", "jv_conditional")}
        for k in keys:
            r = results["runs"][k]
            per_env["report"].append(r["methods"]["Fieller"]["report"][0])
            per_env["conditional"].append(r["methods"]["Fieller"]["conditional"][0])
            per_env["joint"].append(r["methods"]["Fieller"]["joint"][0])
            per_env["delta_all"].append(r["methods"]["raw delta"]["conditional"][0])
            per_env["common_F_gap"].append(r["methods"]["Fieller"]["common_F"][0] - r["methods"]["raw delta"]["common_F"][0])
            per_env["zero_outcome_frac"].append(r["zero_outcome_frac"][0])
            per_env["pred_frac"].append(r["pred_frac_n2_zero"])
            per_env["obs_frac_n2zero"].append(r["frac_n2_zero"][0])
            per_env["truebest_cond"].append(r["selection"]["truebest|Fieller|naive"]["conditional"][0])
            per_env["truebest_report"].append(r["selection"]["truebest|Fieller|naive"]["report"][0])
            per_env["sel_cond"].append(r["selection"]["Fieller|naive"]["conditional"][0])
            per_env["reward"].append(r["reward_mean"][0])
            per_env["jc_report"].append(r["methods"]["Jeffreys+cost"]["report"][0])
            per_env["jc_conditional"].append(r["methods"]["Jeffreys+cost"]["conditional"][0])
            per_env["jv_conditional"].append(r["methods"]["delta-JV"]["conditional"][0])
        summ = {"n_env": len(keys), "MC_per_env": MC_ENV}
        for q, vals in per_env.items():
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            summ[q] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                       "env_boot": boot_mean(v, 500)} if v.size else None
        gaps = np.abs(np.asarray(per_env["obs_frac_n2zero"]) - np.asarray(per_env["pred_frac"]))
        summ["max_abs_pred_gap"] = float(gaps.max()) if gaps.size else None
        envrep[pol] = summ
        print("ENVREP %-18s n=%d report mean=%.3f [%.3f,%.3f] cond mean=%.3f [%.3f,%.3f] joint=%.3f deltaAll=%.3f cFgap=%.4f [%.4f,%.4f] truebest cond=%.3f [%.3f,%.3f] pred gap max=%.3f" % (
            pol, summ["n_env"], summ["report"]["mean"], summ["report"]["min"], summ["report"]["max"], summ["conditional"]["mean"],
            summ["conditional"]["min"], summ["conditional"]["max"], summ["joint"]["mean"], summ["delta_all"]["mean"],
            summ["common_F_gap"]["mean"], summ["common_F_gap"]["env_boot"][1], summ["common_F_gap"]["env_boot"][2],
            summ["truebest_cond"]["mean"], summ["truebest_cond"]["min"], summ["truebest_cond"]["max"], summ["max_abs_pred_gap"]))
    results["env_replication"] = envrep
    results["fixed_n_calibration"] = fixed_n_calibration()
    for rec in results["fixed_n_calibration"]:
        print("fixed-n n=%2d Fieller engine=%.3f textbook=%.3f (report %.3f/%.3f) | delta engine=%.3f textbook=%.3f" % (
            rec["n"], rec["engine_F"], rec["textbook_F"], rec["report_F"], rec["report_textbook_F"], rec["engine_D"], rec["textbook_D"]))
    results["alpha_sweep"] = alpha_sweep()
    for pol in results["alpha_sweep"]:
        for a, rec in results["alpha_sweep"][pol].items():
            print("alpha=%s %-12s Fieller rep=%.3f cond=%.3f | delta cond(all)=%.3f common-F=%.3f | JC rep=%.3f cond=%.3f" % (
                a, pol, rec["Fieller"]["report"][0], rec["Fieller"]["conditional"][0], rec["raw delta"]["conditional"][0],
                rec["raw delta"]["common_F"][0], rec["Jeffreys+cost"]["report"][0], rec["Jeffreys+cost"]["conditional"][0]))
    results["denominator_coverage"] = denominator_coverage()
    for rec in results["denominator_coverage"]:
        print("den t-interval %-10s cv=%.3f pa=%s: " % (rec["cost_dist"], rec["cv"], rec["pareto_alpha"]) +
              " ".join("n=%s eng=%.3f unb=%.3f" % (k, v["engine"], v["unbiased"]) for k, v in rec["n"].items()))
    json.dump(results, open(OUT_JSON, "w"), indent=1, sort_keys=True)
    print("written", OUT_JSON, "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
