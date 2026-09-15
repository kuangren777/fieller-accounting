#!/usr/bin/env python3
"""New experiments for the ICASSP27-6 reportability paper.

S0  oracle self-check, reusing the analytic rho of the archived scripts
S1  validation of the closed-form reportability model
S2  the implied sample size n* as a design rule
S3  replication across environment instances (the archived runs used one)
S4  arm success rates taken from real LLM evaluation traces

Writes reportability_out.txt next to this file and reportability.json for the
paper's numbers pipeline.  Heavy cells are cached in reportability_cache.npz.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(os.path.dirname(HERE), "archive", "tmp", "experiments")
sys.path.insert(0, ARCH)
import expD_lib as L  # noqa: E402

ALPHA = 0.10
OUT = open(os.path.join(HERE, "reportability_out.txt"), "w")
RESULTS: dict = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")
    OUT.flush()


def hr(title):
    say("\n" + "=" * 100)
    say(title)
    say("=" * 100)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0.0, c - h)), float(min(1.0, c + h))


def true_ratio(env):
    """rho_a = E[r]/E[c] under sample_arm's reward-dependent cost law."""
    s2 = env["cost_sigma"] ** 2
    ec = env["c"] * np.exp(s2 / 2.0) * (1.0 + env["gamma"] * env["mu"])
    return env["mu"] / ec


def budget(env, mult):
    return mult * np.sum(env["base_c0"]) * np.exp(env["cost_sigma"] ** 2 / 2.0)


def cv_of(sigma):
    return float(np.sqrt(np.exp(sigma ** 2) - 1.0))


# ======================================================================================
#  the model
# ======================================================================================
def t_crit(n, alpha=ALPHA):
    return tdist.ppf(1 - alpha / 2, np.maximum(np.asarray(n, float) - 1, 1))


def cv_threshold(n, alpha=ALPHA):
    """A > 0 is EXACTLY the event that the sample CV falls below this threshold.

    A = cbar^2 - t^2 scc / n > 0  <=>  scc / cbar^2 < n / t^2  <=>  CVhat < sqrt(n)/t.
    This is an identity, not an approximation, and it holds for any cost law.
    """
    return np.sqrt(np.asarray(n, float)) / t_crit(n, alpha)


def p_cost_gauss(n, sigma, alpha=ALPHA):
    """Gaussian refinement: treat T_n as N(sqrt(n)/CV, 1).  Accurate only when the
    sample CV is well concentrated, which fails for heavy-tailed costs."""
    n = np.asarray(n, float)
    t = t_crit(n, alpha)
    m = np.sqrt(n) / cv_of(sigma)
    return 1.0 - norm.cdf(t - m) + norm.cdf(-t - m)


def n_plugin_cost(sigma, alpha=ALPHA, nmax=4000):
    """Plug-in design rule from the identity: sqrt(n)/t > CV, i.e. n > t^2 CV^2."""
    cv = cv_of(sigma)
    for n in range(2, nmax + 1):
        if cv_threshold(n, alpha) > cv:
            return n
    return -1


def n_outcome(mu, eta=0.05):
    """Exact: P(rbar = 0) = (1-mu)^n <= eta."""
    return int(np.ceil(np.log(eta) / np.log(1.0 - mu)))


def p_num_branch(n, mu):
    """P(rbar > 0) = 1 - (1-mu)^n.  rbar = 1 keeps D > 0, so it does not abstain."""
    return 1.0 - (1.0 - np.asarray(mu, float)) ** np.asarray(n, float)


def p_report_gauss(n, mu, sigma, alpha=ALPHA):
    return p_cost_gauss(n, sigma, alpha) * p_num_branch(n, mu)


# ======================================================================================
#  S0
# ======================================================================================
def s0_oracle():
    hr("S0  ORACLE SELF-CHECK  (analytic rho vs 4e5 iid draws per arm; gate |rel err| < 1%)")
    worst = 0.0
    for (sg, gm) in [(0.35, 0.0), (0.90, 0.0), (1.50, 0.0), (0.35, 0.8)]:
        env = L.make_env(K=24, seed=1, cost_sigma=sg, gamma=gm)
        rt = true_ratio(env)
        rng = np.random.default_rng(12345)
        NS = 400000
        emp = np.zeros(env["K"])
        for a in range(env["K"]):
            rew = rng.random(NS) < env["mu"][a]
            c = rng.lognormal(np.log(env["c"][a]), sg, NS)
            c = np.where(rew, c * (1.0 + gm), c)
            emp[a] = rew.mean() / c.mean()
        rel = np.abs(emp - rt) / rt
        worst = max(worst, rel.max())
        say("  sigma=%.2f gamma=%.1f : max|rel err|=%.5f  mean=%.5f" % (sg, gm, rel.max(), rel.mean()))
    say("  WORST = %.5f -> %s" % (worst, "PASS" if worst < 0.01 else "FAIL"))
    assert worst < 0.01
    RESULTS["oracle_worst"] = worst


# ======================================================================================
#  S1  model validation
# ======================================================================================
NMAX = 60
N_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
MU_GRID = [0.05, 0.10, 0.18, 0.30]
SIG_GRID = [0.35, 0.90, 1.50]
NREP = 200000


def _s1_cell(args):
    mu, sigma, gamma, seed = args
    rng = np.random.default_rng(seed)
    r = (rng.random((NREP, NMAX)) < mu).astype(np.float64)
    c = rng.lognormal(0.0, sigma, (NREP, NMAX))
    if gamma:
        c = np.where(r > 0, c * (1.0 + gamma), c)
    cr = np.cumsum(r, 1)
    cc = np.cumsum(c, 1)
    cr2 = np.cumsum(r * r, 1)
    cc2 = np.cumsum(c * c, 1)
    crc = np.cumsum(r * c, 1)
    out = []
    for n in N_GRID:
        i = n - 1
        rb = cr[:, i] / n
        cb = cc[:, i] / n
        srr = cr2[:, i] / n - rb ** 2
        scc = cc2[:, i] / n - cb ** 2
        src = crc[:, i] / n - rb * cb
        tq = tdist.ppf(1 - ALPHA / 2, n - 1.0)
        A = cb ** 2 - tq ** 2 * scc / n
        B = -2.0 * (cb * rb - tq ** 2 * src / n)
        C = rb ** 2 - tq ** 2 * srr / n
        D = B ** 2 - 4.0 * A * C
        rep = (A > 0) & (D > 0)
        cvhat = np.sqrt(np.maximum(scc, 0.0)) / np.maximum(cb, 1e-12)
        ident = (cvhat < cv_threshold(n)) == (A > 0)
        mc = float((A > 0).mean())
        mn = float((rb > 0).mean())
        out.append(dict(mu=mu, sigma=sigma, gamma=gamma, n=n,
                        meas=float(rep.mean()),
                        meas_cost=mc,
                        meas_num=mn,
                        ident_ok=float(ident.mean()),
                        cvhat_over_cv=float(cvhat.mean() / cv_of(sigma)),
                        factor_err=float(rep.mean() - mc * mn),
                        pred_gauss=float(p_report_gauss(n, mu, sigma)),
                        pred_cost_gauss=float(p_cost_gauss(n, sigma)),
                        pred_num=float(p_num_branch(n, mu))))
    return out


def s1_model():
    hr("S1  WHEN A RATIO INTERVAL EXISTS   report  <=>  (CVhat_n < sqrt(n)/t)  and  (rbar > 0)")
    say("  %d replicates per cell, LogNormal costs, Bernoulli outcomes, independent" % NREP)
    jobs = [(mu, sg, 0.0, 900 + 7 * i)
            for i, (mu, sg) in enumerate([(m, s) for m in MU_GRID for s in SIG_GRID])]
    t0 = time.time()
    with Pool(min(12, len(jobs))) as p:
        rows = [r for chunk in p.map(_s1_cell, jobs) for r in chunk]
    say("  %d cells in %.1fs" % (len(rows), time.time() - t0))

    ident = np.array([r["ident_ok"] for r in rows])
    say("\n  [1] the identity  A > 0  <=>  CVhat < sqrt(n)/t")
    say("      agreement over %d cells x %d replicates: min %.6f, mean %.6f"
        % (len(rows), NREP, ident.min(), ident.mean()))

    fe = np.array([r["factor_err"] for r in rows])
    say("\n  [2] factorisation  P(report) = P(cost branch) x P(outcome branch)")
    say("      independent costs and outcomes: max |err| = %.5f, mean |err| = %.5f"
        % (np.abs(fe).max(), np.abs(fe).mean()))

    ne = np.array([r["meas_num"] - r["pred_num"] for r in rows])
    say("\n  [3] outcome branch  P(rbar > 0) = 1 - (1-mu)^n")
    say("      max |err| = %.5f over %d cells" % (np.abs(ne).max(), len(rows)))

    ge = np.array([r["meas_cost"] - r["pred_cost_gauss"] for r in rows])
    say("\n  [4] Gaussian refinement of the cost branch, treating T_n as N(sqrt(n)/CV, 1)")
    say("      max |err| = %.4f, mean |err| = %.4f  -- it is NOT reliable" %
        (np.abs(ge).max(), np.abs(ge).mean()))
    for sg in SIG_GRID:
        sub = [r for r in rows if r["sigma"] == sg and r["mu"] == MU_GRID[0]]
        e = np.array([r["meas_cost"] - r["pred_cost_gauss"] for r in sub])
        say("      sigma=%.2f (CV=%.2f): max |err| = %.4f" % (sg, cv_of(sg), np.abs(e).max()))

    say("\n  [5] why it fails: the sample CV is biased low, E[CVhat]/CV by n")
    say("      %-8s %s" % ("sigma", "".join("%8d" % n for n in [2, 4, 8, 15, 30, 60])))
    cvb = {}
    for sg in SIG_GRID:
        vals = []
        for n in [2, 4, 8, 15, 30, 60]:
            r = next(r for r in rows if r["sigma"] == sg and r["n"] == n and r["mu"] == MU_GRID[0])
            vals.append(r["cvhat_over_cv"])
        cvb[sg] = vals
        say("      %-8.2f %s" % (sg, "".join("%8.3f" % v for v in vals)))

    say("\n  [6] which branch binds, measured P(cost) and P(outcome) at n=10")
    for mu in MU_GRID:
        line = "      mu=%.2f :" % mu
        for sg in SIG_GRID:
            r = next(r for r in rows if r["sigma"] == sg and r["n"] == 10 and r["mu"] == mu)
            line += "   sigma=%.2f cost=%.3f out=%.3f" % (sg, r["meas_cost"], r["meas_num"])
        say(line)

    RESULTS["s1"] = dict(
        cells=len(rows), nrep=NREP,
        ident_min=float(ident.min()), ident_mean=float(ident.mean()),
        factor_max=float(np.abs(fe).max()), factor_mean=float(np.abs(fe).mean()),
        num_max=float(np.abs(ne).max()),
        gauss_max=float(np.abs(ge).max()), gauss_mean=float(np.abs(ge).mean()),
        cv_bias=cvb, rows=rows)

    say("\n  [7] coupling check: gamma = 0.8 makes cost depend on the outcome")
    jobs_g = [(mu, sg, 0.8, 500 + 3 * i)
              for i, (mu, sg) in enumerate([(m, s) for m in (0.10, 0.30) for s in (0.35, 1.50)])]
    with Pool(4) as p:
        rows_g = [r for chunk in p.map(_s1_cell, jobs_g) for r in chunk]
    fg = np.array([r["factor_err"] for r in rows_g])
    ig = np.array([r["ident_ok"] for r in rows_g])
    say("      %d coupled cells: factorisation max |err| = %.5f, identity agreement %.6f"
        % (len(rows_g), np.abs(fg).max(), ig.min()))
    RESULTS["s1_gamma"] = dict(cells=len(rows_g), factor_max=float(np.abs(fg).max()),
                               ident_min=float(ig.min()))


# ======================================================================================
#  S2  design rule
# ======================================================================================
def s2_design():
    hr("S2  DESIGN RULE   how many pulls per arm before a ratio is reportable")
    rows = RESULTS["s1"]["rows"]
    say("  outcome side is exact: n >= log(eta)/log(1-mu) for P(rbar=0) <= eta")
    say("  cost side plug-in from the identity: smallest n with sqrt(n)/t > CV")
    say("\n    %-6s %-7s | %-9s %-9s %-9s | %-9s" %
        ("mu", "sigma", "n outcome", "n cost", "n rule", "n measured"))
    recs = []
    for mu in MU_GRID:
        for sg in SIG_GRID:
            sub = sorted([r for r in rows if r["mu"] == mu and r["sigma"] == sg],
                         key=lambda r: r["n"])
            meas = next((r["n"] for r in sub if r["meas"] >= 0.90), None)
            n_out = n_outcome(mu, 0.05)
            n_cost = n_plugin_cost(sg)
            rule = max(n_out, n_cost)
            recs.append(dict(mu=mu, sigma=sg, n_outcome=n_out, n_cost=n_cost,
                             n_rule=rule, n_meas=meas))
            say("    %-6.2f %-7.2f | %-9d %-9d %-9d | %-9s"
                % (mu, sg, n_out, n_cost, rule, meas))
    ok = [r for r in recs if r["n_meas"] is not None]
    cons = [r for r in ok if r["n_rule"] >= r["n_meas"]]
    say("\n  the rule is conservative (n_rule >= n_measured) in %d of %d grid points"
        % (len(cons), len(ok)))
    ratio = np.array([r["n_rule"] / r["n_meas"] for r in ok])
    say("  n_rule / n_measured: median %.2f, range [%.2f, %.2f]"
        % (np.median(ratio), ratio.min(), ratio.max()))
    say("  conservativeness comes from the cost side, since E[CVhat] < CV at small n")

    lo = [r for r in ok if r["sigma"] == 0.35]
    rl = np.array([r["n_rule"] / r["n_meas"] for r in lo])
    say("  restricted to sigma=0.35 where the outcome side binds: median %.2f, range [%.2f, %.2f]"
        % (np.median(rl), rl.min(), rl.max()))
    RESULTS["s2"] = dict(records=recs, conservative=len(cons), total=len(ok),
                         ratio_med=float(np.median(ratio)),
                         ratio_lo=float(ratio.min()), ratio_hi=float(ratio.max()),
                         low_sigma_med=float(np.median(rl)))


# ======================================================================================
#  S3  environment-instance replication
# ======================================================================================
N_ENV = 20
MC_S3 = 200
SIG_S3 = 0.35
MULT_S3 = 30


def _s3_one(args):
    base_seed, strat = args
    env = L.make_env(K=24, seed=1, cost_sigma=SIG_S3, gamma=0.0, base_seed=base_seed)
    rt = true_ratio(env)
    B = budget(env, MULT_S3)
    f_fin = f_cov = 0
    d_same = 0
    d_all_n = d_all_cov = 0
    degen = degen_n = 0
    for m in range(MC_S3):
        res = L.run(env, strat, B, seed=910000 + m)
        st = res["arm_stats"]
        n = st["n"].astype(float)
        rb = np.where(n > 0, st["sr"] / np.maximum(n, 1), 0.0)
        cb = np.where(n > 0, st["sc"] / np.maximum(n, 1), 0.0)
        srr, scc, src = L.moment_vars(rb, cb, st["sr2"], st["sc2"], st["src"], n)
        lo_f, hi_f = L.fieller_ci(rb, cb, srr, scc, src, n, ALPHA)
        lo_d, hi_d = L.delta_ci(rb, cb, srr, scc, src, n, ALPHA)
        use = n >= 2
        fin = np.isfinite(lo_f) & np.isfinite(hi_f) & use
        covf = (lo_f <= rt) & (rt <= hi_f)
        covd = (lo_d <= rt) & (rt <= hi_d) & np.isfinite(lo_d) & np.isfinite(hi_d)
        f_fin += int(fin.sum())
        f_cov += int((fin & covf).sum())
        d_same += int((fin & covd).sum())
        d_all_n += int(use.sum())
        d_all_cov += int((use & covd).sum())
        degen += int((use & (srr <= 1e-12)).sum())
        degen_n += int(use.sum())
    return dict(base_seed=base_seed, strat=strat,
                f_fin=f_fin, f_cov=f_cov, d_same=d_same,
                d_all_n=d_all_n, d_all_cov=d_all_cov,
                degen=degen, degen_n=degen_n)


def s3_envs():
    hr("S3  REPLICATION ACROSS ENVIRONMENT INSTANCES  (%d instances, MC=%d, sigma=%.2f, %dx)"
       % (N_ENV, MC_S3, SIG_S3, MULT_S3))
    say("  the archived runs all used base_seed=20260813; here the arm parameters")
    say("  themselves are redrawn, which is the variation the earlier grid did not cover")
    jobs = [(20260813 + 1000 * i, s) for i in range(N_ENV) for s in ("ucb_fieller", "uniform")]
    t0 = time.time()
    with Pool(20) as p:
        rows = p.map(_s3_one, jobs)
    say("  %d runs in %.1fs" % (len(rows), time.time() - t0))

    for strat in ("ucb_fieller", "uniform"):
        rs = [r for r in rows if r["strat"] == strat]
        fc = np.array([r["f_cov"] / r["f_fin"] for r in rs])
        ds = np.array([r["d_same"] / r["f_fin"] for r in rs])
        da = np.array([r["d_all_cov"] / r["d_all_n"] for r in rs])
        dg = np.array([r["degen"] / r["degen_n"] for r in rs])
        gap = fc - ds
        say("\n  %s" % strat)
        say("    Fieller coverage on its reported set   %.4f  [%.4f, %.4f] over instances"
            % (fc.mean(), fc.min(), fc.max()))
        say("    delta   coverage on that same set      %.4f  [%.4f, %.4f]"
            % (ds.mean(), ds.min(), ds.max()))
        say("    gap between them                       %+.4f [%+.4f, %+.4f]"
            % (gap.mean(), gap.min(), gap.max()))
        say("    delta   coverage on all units          %.4f  [%.4f, %.4f]"
            % (da.mean(), da.min(), da.max()))
        say("    degenerate share                       %.4f  [%.4f, %.4f]"
            % (dg.mean(), dg.min(), dg.max()))
        RESULTS.setdefault("s3", {})[strat] = dict(
            n_env=len(rs),
            f_cov=float(fc.mean()), f_cov_lo=float(fc.min()), f_cov_hi=float(fc.max()),
            d_same=float(ds.mean()), d_same_lo=float(ds.min()), d_same_hi=float(ds.max()),
            gap=float(gap.mean()), gap_lo=float(gap.min()), gap_hi=float(gap.max()),
            gap_absmax=float(np.abs(gap).max()),
            d_all=float(da.mean()), d_all_lo=float(da.min()), d_all_hi=float(da.max()),
            degen=float(dg.mean()), degen_lo=float(dg.min()), degen_hi=float(dg.max()))


# ======================================================================================
#  S4  real LLM success rates
# ======================================================================================
def load_real_mu():
    out = {}
    p7 = os.path.join(ARCH, "e7_traces.json")
    d = json.load(open(p7))
    mu7 = [t["phat"] for t in d["traces"] if t.get("phat") is not None]
    out["e7"] = np.array(mu7, float)
    p8 = os.path.join(ARCH, "e8_traces_qwen15b.json")
    d8 = json.load(open(p8))
    cand = d8.get("candidates", [])
    # e8 stores the empirical success rate under the key "rate"
    mu8 = [c.get("phat", c.get("rate")) for c in cand
           if c.get("phat") is not None or c.get("rate") is not None]
    out["e8"] = np.array(mu8, float)
    return out


def cost_branch_lookup(n, sigma):
    """Measured P(cost branch) from S1, interpolated in n.  It depends on sigma and n
    only, so one table serves every arm."""
    rows = [r for r in RESULTS["s1"]["rows"] if r["sigma"] == sigma and r["mu"] == MU_GRID[0]]
    rows.sort(key=lambda r: r["n"])
    xs = [r["n"] for r in rows]
    ys = [r["meas_cost"] for r in rows]
    return float(np.interp(n, xs, ys))


def s4_real():
    hr("S4  ARM SUCCESS RATES FROM REAL LLM EVALUATION TRACES")
    real = load_real_mu()
    for k, v in real.items():
        if len(v) == 0:
            say("  %s: no usable success rates found" % k)
            continue
        say("  %s: %d candidates, rate in [%.3f, %.3f], median %.3f, share exactly 0 = %.3f"
            % (k, len(v), v.min(), v.max(), np.median(v), float((v == 0).mean())))
        RESULTS.setdefault("s4_src", {})[k] = dict(
            n=len(v), lo=float(v.min()), hi=float(v.max()),
            med=float(np.median(v)), zero=float((v == 0).mean()))
    mu = np.concatenate([real["e7"], real["e8"]])
    keep = (mu > 0) & (mu < 1)
    say("  pooled %d candidates, %d with 0 < phat < 1 (the rest cannot define a positive ratio)"
        % (len(mu), int(keep.sum())))
    mu = mu[keep]
    K = len(mu)

    for sigma in (0.35, 1.50):
        env = L.make_env(K=K, seed=1, cost_sigma=sigma, gamma=0.0)
        env["mu"] = mu.copy()
        rt = true_ratio(env)
        B = budget(env, MULT_S3)
        f_fin = f_cov = d_same = 0
        tot = 0
        pred_sum = 0.0
        nn_all = []
        for m in range(MC_S3):
            res = L.run(env, "ucb_fieller", B, seed=920000 + m)
            st = res["arm_stats"]
            n = st["n"].astype(float)
            rb = np.where(n > 0, st["sr"] / np.maximum(n, 1), 0.0)
            cb = np.where(n > 0, st["sc"] / np.maximum(n, 1), 0.0)
            srr, scc, src = L.moment_vars(rb, cb, st["sr2"], st["sc2"], st["src"], n)
            lo_f, hi_f = L.fieller_ci(rb, cb, srr, scc, src, n, ALPHA)
            lo_d, hi_d = L.delta_ci(rb, cb, srr, scc, src, n, ALPHA)
            use = n >= 2
            fin = np.isfinite(lo_f) & use
            f_fin += int(fin.sum())
            f_cov += int((fin & (lo_f <= rt) & (rt <= hi_f)).sum())
            d_same += int((fin & (lo_d <= rt) & (rt <= hi_d)).sum())
            tot += int(use.sum())
            cb_tab = np.array([cost_branch_lookup(int(v), sigma) for v in n[use]])
            pred_sum += float((cb_tab * p_num_branch(n[use], mu[use])).sum())
            nn_all.append(n[use])
        nn = np.concatenate(nn_all)
        meas_rate = f_fin / tot
        pred_rate = pred_sum / tot
        fc = wilson(f_cov, f_fin)
        ds = wilson(d_same, f_fin)
        say("\n  sigma=%.2f, K=%d real candidates, budget %dx, MC=%d" % (sigma, K, MULT_S3, MC_S3))
        say("    median pulls per arm %.0f, quartiles %.0f / %.0f"
            % (np.median(nn), np.percentile(nn, 25), np.percentile(nn, 75)))
        say("    reportability measured %.4f   predicted by the model %.4f   error %+.4f"
            % (meas_rate, pred_rate, meas_rate - pred_rate))
        say("    Fieller coverage on reported set %.4f [%.4f, %.4f]" % fc)
        say("    delta   coverage on the same set %.4f [%.4f, %.4f]" % ds)
        RESULTS.setdefault("s4", {})["sigma%.2f" % sigma] = dict(
            K=K, meas=meas_rate, pred=pred_rate, err=meas_rate - pred_rate,
            f_cov=fc[0], f_lo=fc[1], f_hi=fc[2], d_cov=ds[0], d_lo=ds[1], d_hi=ds[2],
            n_med=float(np.median(nn)), n_units=tot)
    RESULTS["s4"]["n_candidates_total"] = int(len(np.concatenate([real["e7"], real["e8"]])))
    RESULTS["s4"]["n_candidates_used"] = K


if __name__ == "__main__":
    t0 = time.time()
    s0_oracle()
    s1_model()
    s2_design()
    s3_envs()
    s4_real()
    say("\nTOTAL %.1fs" % (time.time() - t0))
    with open(os.path.join(HERE, "reportability.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    say("wrote reportability.json")
    OUT.close()
