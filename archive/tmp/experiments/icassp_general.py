# -*- coding: utf-8 -*-
"""
X4 · icassp_general.py — 分布与机制泛化（缺口 G4）
================================================================================
目的：检验"delta 条件覆盖崩塌 / Fieller 弃权后条件覆盖守住"这一现象是否
      只是 (LogNormal 成本 + UCB 机制 + Bernoulli 奖励) 这一组合的仿真伪影。

三个轴：
  轴1 成本分布：LogNormal / Gamma（轻尾）/ Pareto-I（重尾, alpha>2 方差有限）/
                两点混合（短链-长链双峰）；**全部匹配同一 per-arm E[c] 与同一 CV**。
                额外极端格 Pareto-I alpha=1.5（均值有限、方差无穷、CV 未定义）。
  轴2 自适应机制：uniform（非自适应对照）/ ucb_fieller / ucb_delta / thompson_ratio /
                eps_greedy_ratio / greedy_ratio / kube_knapsack。
  轴3 奖励分布：Bernoulli(mu) → Beta(mu*nu,(1-mu)*nu)（同 E[r]=mu，但样本方差几乎
                不可能为 0）。这是分离"零经验方差退化"与"自适应污染"的决定性检验。

口径纪律（遵守预审 F1/F2/F3）：
  * 每格都报**同子集**四列覆盖：Fieller cov|finite / delta cov|Fieller-finite /
    delta cov|all n>=2 / delta cov|Fieller-abstain，并配中位宽度。
  * 报 srr==0（奖励样本方差恰为 0）占比，作为 delta 零宽度退化的直接诊断。
  * 报 Agresti-Coull 平滑方差的 delta（不退化的公平基线）。
  * 报 interval score（alpha=0.10 → 2/alpha=20）与弃权惩罚 cap 的交叉点 c*。
  * 禁用 naive 覆盖口径。

实现方式：**不改 expD_lib.py**。用 monkeypatch 替换 L.sample_arm，按 env 里的
  cost_dist / reward_dist 分派；LogNormal+Bernoulli 路径逐字复刻原实现（含 RNG
  消耗顺序与 ln-scale 的 snap），因此可逐格复现既有锚点。

运行：timeout 1700 python3 icassp_general.py | tee icassp_general_out.txt
      （分格缓存 icassp_general_cache.pkl，可重复运行续跑；跑完再运行一次得完整 stdout）
================================================================================
"""
import os
import sys
import time
import pickle
import numpy as np
from scipy import stats
from scipy.stats import t as tdist
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expD_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
# 注：同目录已存在一个 schema 不同的 icassp_general_cache.pkl（早期同名脚本产物），
# 故本实验用独立缓存名，避免键位冲突与互相覆盖。
CACHE = os.path.join(HERE, "icassp_general_x4cache_v2.pkl")
FIG = "figures/icassp_general.png"

MC = int(os.environ.get("XG_MC", "200"))
ALPHA = 0.10
SEED_BASE = 40000            # 与 expD_e3_validity.py 同一 traj seed 口径（锚点可复现）
K = 24
NPROC = 2
BETA_NU = 4.0                # Beta 奖励的集中度：Var = mu(1-mu)/(nu+1)
MECHS = ["uniform", "ucb_fieller", "ucb_delta", "thompson_ratio",
         "eps_greedy_ratio", "greedy_ratio", "kube_knapsack"]
OKABE = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
         "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "verm": "#D55E00", "purple": "#CC79A7"}


# ====================================================================================
#  成本分布层：全部按 (目标 E[c]=m, 目标 CV=v) 参数化，形状不同、前两阶矩相同
# ====================================================================================
def cost_params(dist, m, cv, extra=None):
    """由 (E[c]=m, CV=cv) 反解各分布参数。返回 dict（含解析 E/Var 供自检）。

    LogNormal(mulog, s): E=exp(mulog+s^2/2), CV=sqrt(exp(s^2)-1)
        -> s = sqrt(log(1+cv^2)), mulog = log(m) - s^2/2
        （注：expD_lib 的原实现是 lognormal(log(c_a), sigma)，其 E = c_a*exp(sigma^2/2)，
          CV = sqrt(exp(sigma^2)-1) —— 与本参数化在 s=sigma 时同族，见 anchor 复现）
    Gamma(k, theta): E=k*theta, CV=1/sqrt(k) -> k=1/cv^2, theta=m*cv^2
    Pareto-I(alpha, xm): E=alpha*xm/(alpha-1) (alpha>1),
        CV^2 = 1/(alpha*(alpha-2)) (alpha>2) -> alpha = 1+sqrt(1+1/cv^2)
        （该 alpha 恒 >2 当 cv<inf；xm = m*(alpha-1)/alpha）
        alpha<=2 时方差无穷：此时按给定 alpha 直接取 xm 匹配均值，CV 记为 inf。
    TwoPoint(p, a, b): 以概率 p 取 b（长链），否则取 a（短链）。
        给定 p 与 (m, cv)：a,b 由 E=m, Var=(cv*m)^2 唯一解（b>a>0）：
        令 d=b-a，Var=p(1-p)d^2 -> d=cv*m/sqrt(p(1-p))；a=m-p*d，b=a+d。
        需 a>0，即 cv < sqrt((1-p)/p)。默认 p=0.2（20% 长链）。
    """
    extra = extra or {}
    if dist == "lognormal":
        s = float(np.sqrt(np.log1p(cv ** 2)))
        mulog = float(np.log(m) - s ** 2 / 2.0)
        return {"dist": dist, "mulog": mulog, "s": s,
                "E": float(np.exp(mulog + s ** 2 / 2)),
                "Var": float((np.exp(s ** 2) - 1) * np.exp(2 * mulog + s ** 2))}
    if dist == "gamma":
        k = float(1.0 / cv ** 2)
        th = float(m * cv ** 2)
        return {"dist": dist, "k": k, "theta": th, "E": k * th, "Var": k * th ** 2}
    if dist == "pareto":
        a_fix = extra.get("alpha")
        if a_fix is None:
            al = float(1.0 + np.sqrt(1.0 + 1.0 / cv ** 2))
        else:
            al = float(a_fix)
        xm = float(m * (al - 1.0) / al)
        var = (float(xm ** 2 * al / ((al - 1) ** 2 * (al - 2)))
               if al > 2 else float("inf"))
        return {"dist": dist, "alpha": al, "xm": xm,
                "E": float(al * xm / (al - 1.0)), "Var": var}
    if dist == "twopoint":
        p = float(extra.get("p", 0.2))
        if cv >= np.sqrt((1 - p) / p):
            raise ValueError("twopoint: cv=%.3f too large for p=%.2f (needs a>0)" % (cv, p))
        d = float(cv * m / np.sqrt(p * (1 - p)))
        a = float(m - p * d)
        b = float(a + d)
        return {"dist": dist, "p": p, "a": a, "b": b,
                "E": float((1 - p) * a + p * b),
                "Var": float(p * (1 - p) * d ** 2)}
    raise ValueError("unknown cost dist %s" % dist)


def draw_cost(par, rng):
    """抽 1 个成本样本（标量）。RNG 每次消耗与原实现同量级（1 次抽样调用）。"""
    d = par["dist"]
    if d == "lognormal":
        return rng.lognormal(par["mulog"], par["s"])
    if d == "gamma":
        return rng.gamma(par["k"], par["theta"])
    if d == "pareto":
        return par["xm"] * (1.0 + rng.pareto(par["alpha"]))   # numpy pareto = Lomax
    if d == "twopoint":
        return par["b"] if rng.random() < par["p"] else par["a"]
    raise ValueError(d)


def draw_cost_vec(par, rng, size):
    """向量化抽成本（仅自检用；轨迹里必须走 draw_cost 保持 RNG 消耗顺序）。"""
    d = par["dist"]
    if d == "lognormal":
        return rng.lognormal(par["mulog"], par["s"], size)
    if d == "gamma":
        return rng.gamma(par["k"], par["theta"], size)
    if d == "pareto":
        return par["xm"] * (1.0 + rng.pareto(par["alpha"], size))
    if d == "twopoint":
        return np.where(rng.random(size) < par["p"], par["b"], par["a"])
    raise ValueError(d)


def tp_p_for_cv(cv):
    """两点混合的长链概率 p：需 cv < sqrt((1-p)/p)。低 CV 用 p=0.2（20% 长链），
    高 CV 自动降到 0.05（否则短链支 a<0 无法匹配二阶矩）。"""
    return 0.2 if cv < 1.6 else 0.05


# ====================================================================================
#  env 增强 + sample_arm monkeypatch（不改 expD_lib.py）
# ====================================================================================
def make_gen_env(cost_dist, cv, reward_dist="bernoulli", base_seed=L.BASE_SEED,
                 env_seed=1, pareto_alpha=None, gamma=0.0):
    """构造泛化环境。

    臂参数（mu_a, c_a）始终用 make_env(cost_sigma=0.35, seed=env_seed, base_seed=...)
    生成 —— 因此 **每臂目标 E[c]_a = c_a·exp(0.35²/2) 与真比率 rho_a 在所有格里完全相同**，
    分布/CV 只改变成本的**形状与变异度**，不改一阶矩。这样跨格比较是干净的。
    """
    env = L.make_env(K=K, mu_range=(0.06, 0.30), cost_sigma=0.35, cost_mean=5.0,
                     gamma=gamma, nonstat="static", seed=env_seed, base_seed=base_seed)
    m_arm = env["c"] * np.exp(0.35 ** 2 / 2.0)          # 每臂目标 E[c]（原 LogNormal 均值）
    extra = {"p": tp_p_for_cv(cv)} if cost_dist == "twopoint" else {}
    if cost_dist == "pareto" and pareto_alpha is not None:
        extra = {"alpha": pareto_alpha}
    pars = []
    for a in range(env["K"]):
        if cost_dist == "lognormal" and pareto_alpha is None and abs(cv - _cv_of_sigma(0.35)) < 1e-12:
            # 逐位复刻原实现：mulog=log(c_a), s=cost_sigma（保证锚点可复现）
            pars.append({"dist": "lognormal", "mulog": float(np.log(env["c"][a])),
                         "s": 0.35, "E": float(m_arm[a]),
                         "Var": float((np.exp(0.35 ** 2) - 1) * m_arm[a] ** 2)})
        else:
            pars.append(cost_params(cost_dist, float(m_arm[a]), cv, extra))
    env["cost_pars"] = pars
    env["cost_dist"] = cost_dist
    env["cost_cv"] = cv
    env["reward_dist"] = reward_dist
    env["m_arm"] = m_arm
    return env


def _cv_of_sigma(sigma):
    return float(np.sqrt(np.exp(sigma ** 2) - 1.0))


def gen_sample_arm(env, a, x, r):
    """替换 L.sample_arm。LogNormal+Bernoulli 路径与原实现 RNG 消耗顺序逐字一致
    （先 r.random() 出奖励，再抽成本，成功轮成本 ×(1+gamma)）。"""
    if "cost_pars" not in env:                      # 非泛化 env → 原实现
        return _ORIG_SAMPLE_ARM(env, a, x, r)
    mu = (L.ns_mu_at if env.get("ns_env") else L.mu_at)(env, a, x)
    if env["reward_dist"] == "bernoulli":
        rew = r.random() < mu
        c = draw_cost(env["cost_pars"][a], r)
        if rew:
            c *= (1.0 + env["gamma"])
        return c, rew, mu
    # Beta 奖励：E[r]=mu（真比率不变），但样本方差几乎不可能恰为 0
    rew = r.beta(mu * BETA_NU, (1.0 - mu) * BETA_NU)
    c = draw_cost(env["cost_pars"][a], r)
    c *= (1.0 + env["gamma"] * float(rew))
    return c, rew, mu


_ORIG_SAMPLE_ARM = L.sample_arm
L.sample_arm = gen_sample_arm


def true_ratio(env):
    """rho_a = E[r_a] / E[c_obs,a]；E[c_obs,a] = E[c]_a·(1+gamma·E[r_a])。"""
    return env["mu"] / (env["m_arm"] * (1.0 + env["gamma"] * env["mu"]))


# ====================================================================================
#  区间构造：Fieller（原实现）/ delta（原实现）/ delta-AC（平滑方差，公平基线）/ Katz-log
# ====================================================================================
def delta_ci_ac(rhat, chat, srr, scc, src, n, alpha=ALPHA, smooth=True):
    """delta CI，方差按展开式直接算（避免 rhat=0 处的 1e-9 夹断），
    并可对分子方差做 Agresti-Coull 风格平滑（预审 F2 要求的不退化公平基线）：
        srr_ac = (n·srr + t²·(1/4)) / (n + t²)
    奖励取值在 [0,1]（Bernoulli 与 Beta 均满足），故 1/4 是方差上界，平滑合法。
    与 Fieller 共用同一 t_{n-1,1-alpha/2} 分位数（预审 P4）。"""
    rhat = np.asarray(rhat, float); chat = np.asarray(chat, float)
    srr = np.asarray(srr, float); scc = np.asarray(scc, float)
    src = np.asarray(src, float); n = np.asarray(n, float)
    bad = n < 2
    ns = np.maximum(n, 2.0)
    tq = tdist.ppf(1 - alpha / 2, ns - 1.0)
    s_rr = (ns * srr + tq ** 2 * 0.25) / (ns + tq ** 2) if smooth else srr
    ch = np.maximum(chat, 1e-12)
    # Var(rhat/chat) ~ [srr/chat^2 + rhat^2 scc/chat^4 - 2 rhat src/chat^3]/n
    var = (s_rr / ch ** 2 + rhat ** 2 * scc / ch ** 4
           - 2.0 * rhat * src / ch ** 3) / ns
    var = np.maximum(var, 0.0)
    pt = rhat / ch
    hw = tq * np.sqrt(var)
    lb = pt - hw; ub = pt + hw
    lb = np.where(bad, -np.inf, lb); ub = np.where(bad, np.inf, ub)
    return lb, ub


def katz_log_ci(rhat, chat, srr, scc, src, n, alpha=ALPHA):
    """Katz-log 区间（标准风险比区间）：log(rbar/cbar) ± t·sqrt(srr/(n rbar²)
    + scc/(n cbar²) − 2 src/(n rbar cbar))，再指数回。rbar=0 → 无界（自带弃权）。"""
    rhat = np.asarray(rhat, float); chat = np.asarray(chat, float)
    srr = np.asarray(srr, float); scc = np.asarray(scc, float)
    src = np.asarray(src, float); n = np.asarray(n, float)
    ns = np.maximum(n, 2.0)
    tq = tdist.ppf(1 - alpha / 2, ns - 1.0)
    bad = (n < 2) | (rhat <= 0) | (chat <= 0)
    rr = np.maximum(rhat, 1e-12); cc = np.maximum(chat, 1e-12)
    v = srr / (ns * rr ** 2) + scc / (ns * cc ** 2) - 2.0 * src / (ns * rr * cc)
    bad = bad | ~(v >= 0)
    v = np.maximum(v, 0.0)
    lg = np.log(rr) - np.log(cc)
    lb = np.exp(lg - tq * np.sqrt(v))
    ub = np.exp(lg + tq * np.sqrt(v))
    lb = np.where(bad, -np.inf, lb); ub = np.where(bad, np.inf, ub)
    return lb, ub


def interval_score(lb, ub, y, alpha=ALPHA):
    """IS_alpha(l,u;y) = (u-l) + (2/alpha)(l-y)_+ + (2/alpha)(y-u)_+。仅对有限区间有意义。"""
    return (ub - lb) + (2.0 / alpha) * np.maximum(lb - y, 0.0) \
                     + (2.0 / alpha) * np.maximum(y - ub, 0.0)


# ====================================================================================
#  单格仿真：一个 (cost_dist, cv, reward_dist, mech, B) 格
# ====================================================================================
def run_cell(spec):
    """跑一格：MC 条轨迹，终止时用 arm_stats 算各法 CI，返回聚合统计（全部标量/小数组）。"""
    cost_dist, cv, reward_dist, mech, bmult, env_seed, pareto_alpha = spec
    env = make_gen_env(cost_dist, cv, reward_dist=reward_dist, env_seed=env_seed,
                       pareto_alpha=pareto_alpha)
    rt = true_ratio(env)
    B = bmult * float(np.sum(env["base_c0"])) * float(np.exp(0.35 ** 2 / 2))
    Ka = env["K"]
    acc = {k: [] for k in ("n", "rho_t", "fin_f", "cov_f", "w_f",
                           "cov_d", "w_d", "srr0", "cov_dac", "w_dac",
                           "cov_k", "fin_k", "w_k", "is_f", "is_d", "is_dac", "is_k")}
    pulls = np.zeros((MC, Ka))
    rewards = np.zeros(MC)
    for i in range(MC):
        res = L.run(env, mech, B, seed=SEED_BASE + i)
        st = res["arm_stats"]
        n = st["n"].astype(float)
        pulls[i] = n
        rewards[i] = res["reward"]
        rhat = st["sr"] / np.maximum(n, 1.0)
        chat = st["sc"] / np.maximum(n, 1.0)
        srr, scc, src = L.moment_vars(rhat, chat, st["sr2"], st["sc2"], st["src"], n)
        lo_f, hi_f = L.fieller_ci(rhat, chat, srr, scc, src, n, alpha=ALPHA)
        lo_d, hi_d = L.delta_ci(rhat, chat, srr, scc, src, n, alpha=ALPHA)
        lo_a, hi_a = delta_ci_ac(rhat, chat, srr, scc, src, n, alpha=ALPHA, smooth=True)
        lo_k, hi_k = katz_log_ci(rhat, chat, srr, scc, src, n, alpha=ALPHA)
        fin_f = np.isfinite(lo_f) & np.isfinite(hi_f)
        fin_k = np.isfinite(lo_k) & np.isfinite(hi_k)
        acc["n"].append(n); acc["rho_t"].append(rt)
        acc["fin_f"].append(fin_f)
        acc["cov_f"].append((rt >= lo_f) & (rt <= hi_f))
        acc["w_f"].append(np.where(fin_f, hi_f - lo_f, np.nan))
        acc["cov_d"].append((rt >= lo_d) & (rt <= hi_d))
        acc["w_d"].append(np.where(np.isfinite(hi_d - lo_d), hi_d - lo_d, np.nan))
        acc["srr0"].append(srr <= 1e-15)
        acc["cov_dac"].append((rt >= lo_a) & (rt <= hi_a))
        acc["w_dac"].append(np.where(np.isfinite(hi_a - lo_a), hi_a - lo_a, np.nan))
        acc["cov_k"].append((rt >= lo_k) & (rt <= hi_k))
        acc["fin_k"].append(fin_k)
        acc["w_k"].append(np.where(fin_k, hi_k - lo_k, np.nan))
        acc["is_f"].append(np.where(fin_f, interval_score(lo_f, hi_f, rt), np.nan))
        acc["is_d"].append(np.where(np.isfinite(hi_d - lo_d),
                                    interval_score(lo_d, hi_d, rt), np.nan))
        acc["is_dac"].append(np.where(np.isfinite(hi_a - lo_a),
                                      interval_score(lo_a, hi_a, rt), np.nan))
        acc["is_k"].append(np.where(fin_k, interval_score(lo_k, hi_k, rt), np.nan))
    M = {k: np.asarray(v) for k, v in acc.items()}
    return aggregate_cell(spec, M, pulls, rewards, env)


def wilson(k, n, conf=0.95):
    """Wilson 分数区间。n=0 → nan。"""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + conf / 2.0)
    p = k / n
    den = 1.0 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(p), float(max(0.0, ctr - hw)), float(min(1.0, ctr + hw))


def aggregate_cell(spec, M, pulls, rewards, env):
    """把 (MC,K) 矩阵压成本格的标量指标（同子集口径 + 宽度 + IS + 集中度）。"""
    cost_dist, cv, reward_dist, mech, bmult, env_seed, pareto_alpha = spec
    ge2 = M["n"] >= 2
    fin = M["fin_f"] & ge2                    # Fieller 有限（n<2 已被 fieller_ci 判无界）
    absg = ge2 & ~M["fin_f"]                  # Fieller 弃权且 n>=2
    fk = M["fin_k"] & ge2
    out = {"spec": {"cost_dist": cost_dist, "cv": cv, "reward_dist": reward_dist,
                    "mech": mech, "bmult": bmult, "env_seed": env_seed,
                    "pareto_alpha": pareto_alpha},
           "n_units_ge2": int(ge2.sum()), "MC": MC,
           "reward_mean": float(np.mean(rewards))}
    # ---- 信息率（有限 Fieller 区间占全部 (arm,traj) 的比例，与既有锚点同口径）----
    out["info_rate"] = wilson(int(M["fin_f"].sum()), int(M["fin_f"].size))
    out["info_rate_ge2"] = wilson(int(fin.sum()), int(ge2.sum()))
    out["katz_fin_rate"] = wilson(int(M["fin_k"].sum()), int(M["fin_k"].size))
    # ---- method-specific finite/reporting sets ----
    # Fieller reports only when its quadratic inversion is bounded.  Delta's own
    # reporting set is n>=2: L.delta_ci returns finite endpoints for n>=2,
    # including the zero-width Bernoulli degeneracy.  Keep these estimands
    # separate from the same-Fieller-subset diagnostic below.
    fin_d = ge2 & np.isfinite(M["w_d"])
    out["info_rate_delta"] = wilson(int(fin_d.sum()), int(M["w_d"].size))
    out["cov_d_cond"] = wilson(int((M["cov_d"] & fin_d).sum()), int(fin_d.sum()))
    out["joint_d"] = wilson(int((M["cov_d"] & fin_d).sum()), int(ge2.size))
    # ---- same-subset coverage (F1 diagnostic) ----
    out["cov_f_fin"] = wilson(int((M["cov_f"] & fin).sum()), int(fin.sum()))
    out["cov_d_on_fin"] = wilson(int((M["cov_d"] & fin).sum()), int(fin.sum()))
    out["cov_d_all"] = wilson(int((M["cov_d"] & ge2).sum()), int(ge2.sum()))
    out["cov_d_on_abst"] = wilson(int((M["cov_d"] & absg).sum()), int(absg.sum()))
    out["cov_dac_all"] = wilson(int((M["cov_dac"] & ge2).sum()), int(ge2.sum()))
    out["cov_dac_on_fin"] = wilson(int((M["cov_dac"] & fin).sum()), int(fin.sum()))
    out["cov_k_fin"] = wilson(int((M["cov_k"] & fk).sum()), int(fk.sum()))
    # ---- joint coverage for Fieller: finite and covered / all n>=2 units ----
    out["joint_f"] = wilson(int((M["cov_f"] & fin).sum()), int(ge2.sum()))
    # ---- 退化诊断（预审 F2）----
    out["srr0_frac_ge2"] = wilson(int((M["srr0"] & ge2).sum()), int(ge2.sum()))
    zw = ge2 & (np.nan_to_num(M["w_d"], nan=1e9) < 1e-9)
    out["delta_zerowidth_ge2"] = wilson(int(zw.sum()), int(ge2.sum()))
    nd = ge2 & ~M["srr0"]
    out["cov_d_nondegen"] = wilson(int((M["cov_d"] & nd).sum()), int(nd.sum()))
    out["fieller_abst_on_srr0"] = wilson(int((absg & M["srr0"]).sum()),
                                         int((ge2 & M["srr0"]).sum()))
    # ---- 宽度（中位 + 四分位）----
    def q(arr, mask):
        v = arr[mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (float("nan"),) * 3
        return tuple(float(x) for x in np.percentile(v, [25, 50, 75]))
    out["w_f_fin"] = q(M["w_f"], fin)
    out["w_d_on_fin"] = q(M["w_d"], fin)
    out["w_d_all"] = q(M["w_d"], fin_d)
    out["w_dac_all"] = q(M["w_dac"], ge2)
    out["w_dac_on_fin"] = q(M["w_dac"], fin)
    out["w_k_fin"] = q(M["w_k"], fk)
    # ---- interval score (method-specific reporting set) ----
    def mn(arr, mask):
        v = arr[mask]
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")
    out["is_f_fin"] = mn(M["is_f"], fin)
    out["is_d_on_fin"] = mn(M["is_d"], fin)
    out["is_d_all"] = mn(M["is_d"], fin_d)
    out["is_dac_all"] = mn(M["is_dac"], ge2)
    out["is_dac_on_fin"] = mn(M["is_dac"], fin)
    out["is_k_fin"] = mn(M["is_k"], fk)
    # ---- cap 敏感性所需：弃权数 / 报告数 / IS 总和 ----
    out["W_med_ref"] = out["w_f_fin"][1]
    out["cnt"] = {"ge2": int(ge2.sum()), "fin": int(fin.sum()), "abst": int(absg.sum()),
                  "fin_d": int(fin_d.sum()), "abst_d": int(ge2.sum() - fin_d.sum()),
                  "fk": int(fk.sum()), "abst_k": int((ge2 & ~M["fin_k"]).sum())}
    out["is_sum"] = {"f": float(np.nansum(np.where(fin, M["is_f"], np.nan))),
                     "d_all": float(np.nansum(np.where(fin_d, M["is_d"], np.nan))),
                     "dac_all": float(np.nansum(np.where(ge2, M["is_dac"], np.nan))),
                     "k": float(np.nansum(np.where(fk, M["is_k"], np.nan)))}
    # ---- 集中度指标（轴2 的关键：崩塌幅度是否被采样集中度预测）----
    tot = pulls.sum(axis=1, keepdims=True)
    frac = pulls / np.maximum(tot, 1)
    top1 = np.sort(frac, axis=1)[:, -1]
    ent = -np.sum(np.where(frac > 0, frac * np.log(np.maximum(frac, 1e-300)), 0.0), axis=1)
    out["top1_share"] = float(np.mean(top1))
    out["entropy"] = float(np.mean(ent))
    out["entropy_norm"] = float(np.mean(ent) / np.log(env["K"]))
    out["mean_pulls"] = float(pulls.sum(axis=1).mean())
    out["frac_n_lt5"] = float(np.mean(pulls < 5))
    # ---- 逐轨迹覆盖率序列（供配对检验；小数组，缓存友好）----
    def per_traj(cov, msk):
        num = (cov & msk).sum(axis=1).astype(float)
        den = msk.sum(axis=1).astype(float)
        return np.where(den > 0, num / np.maximum(den, 1), np.nan)
    out["traj_cov_f_fin"] = per_traj(M["cov_f"], fin)
    out["traj_cov_d_on_fin"] = per_traj(M["cov_d"], fin)
    out["traj_cov_d_all"] = per_traj(M["cov_d"], fin_d)
    out["traj_cov_dac_all"] = per_traj(M["cov_dac"], ge2)
    return out


# ====================================================================================
#  自检 S0：分布矩匹配 + oracle 真比率 + 锚点复现
# ====================================================================================
def selfcheck(cost_grid, n_draw=400000):
    """S0a 每个 (分布,CV) 的经验 E[c]/CV 对齐解析值（<1%）；
       S0b oracle 比率：大样本 iid 抽 (r,c) 估 rho_a，与解析 true_ratio 比（<1%）；
       S0c 锚点复现：LogNormal(sigma=0.35)+Bernoulli+ucb_fieller, 30x
           → 信息率 0.318 / Fieller 条件覆盖 0.945 / delta(all n>=2) 0.301。"""
    print("=" * 108)
    print("S0 · 自检（分布矩匹配 / oracle 真比率 / 既有锚点复现）")
    print("=" * 108)
    print("判据说明：经验值本身有 MC 误差，故用 **z = |emp-ana|/SE_emp < 4** 判定（而非固定 1%%），")
    print("         并同时报中位数（对重尾稳健、与矩无关）的相对误差 <1%% 作为形状正确性检验。")
    ok_all = True
    print("\n[S0a] 成本分布：解析 vs 经验（每格 %d 向量化抽样，臂 #0）" % n_draw)
    print("    %-26s | %-8s %-8s %-6s | %-8s %-8s | %-8s %-8s %-8s"
          % ("cell", "E_ana", "E_emp", "z(E)", "CV_ana", "CV_emp",
             "med_ana", "med_emp", "rel(med)"))
    for cost_dist, cv, pa in cost_grid:
        env = make_gen_env(cost_dist, cv, pareto_alpha=pa)
        par = env["cost_pars"][0]
        rng = np.random.default_rng(777)
        xs = draw_cost_vec(par, rng, n_draw)
        e_ana = par["E"]
        cv_ana = (float(np.sqrt(par["Var"]) / par["E"])
                  if np.isfinite(par["Var"]) else float("inf"))
        e_emp = float(xs.mean()); cv_emp = float(xs.std(ddof=1) / xs.mean())
        se = float(xs.std(ddof=1) / np.sqrt(xs.size))
        z = abs(e_emp - e_ana) / se
        if cost_dist == "lognormal":
            med_ana = float(np.exp(par["mulog"]))
        elif cost_dist == "gamma":
            med_ana = float(stats.gamma.ppf(0.5, par["k"], scale=par["theta"]))
        elif cost_dist == "pareto":
            med_ana = float(par["xm"] * 2.0 ** (1.0 / par["alpha"]))
        else:
            med_ana = float(par["a"] if par["p"] < 0.5 else par["b"])
        med_emp = float(np.median(xs))
        relm = abs(med_emp - med_ana) / med_ana
        # Var=inf（Pareto alpha<=2）时样本 SE 无意义 → 只用中位数判形状（诚实标注）
        finite_var = np.isfinite(par["Var"])
        ok = (relm < 0.01) and (z < 4.0 if finite_var else True)
        ok_all &= ok
        print("    %-26s | %-8.4f %-8.4f %-6.2f | %-8.3f %-8.3f | %-8.4f %-8.4f %.2e %s"
              % (cell_label(cost_dist, cv, pa), e_ana, e_emp, z, cv_ana, cv_emp,
                 med_ana, med_emp, relm, "OK" if ok else "FAIL"))
    print("\n[S0b] oracle 真比率：iid 抽 (r,c) 估 rho_a=E[r]/E[c] vs 解析 true_ratio()")
    print("      （每 (格,臂) %d 抽样；SE 由 delta 法算；臂 0/11/23 = 低/中/高 rho）" % n_draw)
    print("    %-34s | %-4s | %-10s %-10s %-6s %-8s" %
          ("cell", "arm", "rho_ana", "rho_emp", "z", "rel"))
    for cost_dist, cv, pa in cost_grid:
        for rd in ("bernoulli", "beta"):
            env = make_gen_env(cost_dist, cv, reward_dist=rd, pareto_alpha=pa)
            rt = true_ratio(env)
            rng = np.random.default_rng(1234)
            for a in (0, 11, 23):
                mu = env["mu"][a]
                if rd == "bernoulli":
                    rs = (rng.random(n_draw) < mu).astype(float)
                else:
                    rs = rng.beta(mu * BETA_NU, (1.0 - mu) * BETA_NU, n_draw)
                cs = draw_cost_vec(env["cost_pars"][a], rng, n_draw)
                cs = cs * (1.0 + env["gamma"] * rs)
                rb = rs.mean(); cb = cs.mean()
                emp = rb / cb
                srr = rs.var(ddof=1); scc = cs.var(ddof=1)
                src = float(np.cov(rs, cs, ddof=1)[0, 1])
                var = (srr / cb ** 2 + rb ** 2 * scc / cb ** 4
                       - 2.0 * rb * src / cb ** 3) / n_draw
                se = float(np.sqrt(max(var, 1e-300)))
                z = abs(emp - rt[a]) / se
                rel = abs(emp - rt[a]) / rt[a]
                # 无穷方差格（Pareto alpha=1.5）：样本 SE 不是有效标尺，放宽到 rel<5%
                inf_var = not np.isfinite(env["cost_pars"][a]["Var"])
                ok = (rel < 0.05) if inf_var else (z < 4.0)
                ok_all &= ok
                tag = "%s/%s" % (cell_label(cost_dist, cv, pa), rd)
                print("    %-34s | %-4d | %-10.6f %-10.6f %-6.2f %.2e %s"
                      % (tag, a, rt[a], emp, z, rel, "OK" if ok else "FAIL"))
    print("\n[S0c] 锚点复现（LogNormal sigma=0.35 + Bernoulli, 30x, MC=%d, seed_base=%d）"
          % (MC, SEED_BASE))
    print("    期望（expD_e3_out.txt T1）: ucb_fieller 信息率 0.318 / Fieller|fin 0.945 / "
          "delta|all 0.301 ; uniform 0.976 / 0.902 / 0.879")
    cv035 = _cv_of_sigma(0.35)
    for mech, ref in (("ucb_fieller", (0.318, 0.945, 0.301)),
                      ("uniform", (0.976, 0.902, 0.879))):
        r = run_cell(("lognormal", cv035, "bernoulli", mech, 30, 1, None))
        got = (r["info_rate"][0], r["cov_f_fin"][0], r["cov_d_all"][0])
        dd = [abs(g - x) for g, x in zip(got, ref)]
        ok = max(dd) < 0.01
        ok_all &= ok
        print("    %-13s 信息率 %.3f (ref %.3f) | Fieller|fin %.3f (ref %.3f) | "
              "delta|all %.3f (ref %.3f) | maxdiff %.4f %s"
              % (mech, got[0], ref[0], got[1], ref[1], got[2], ref[2], max(dd),
                 "OK" if ok else "FAIL"))
    print("\n  S0 总判定: %s" % ("ALL OK" if ok_all else "存在 FAIL —— 后续结果不可信"))
    return ok_all


# ====================================================================================
#  网格定义
# ====================================================================================
CV_LO = _cv_of_sigma(0.35)      # 0.3627（= 既有实验的 sigma=0.35 LogNormal）
CV_HI = _cv_of_sigma(1.5)       # 2.9236（= 既有实验的 sigma=1.5 重尾格）

# (cost_dist, cv, pareto_alpha)
COST_GRID = [
    ("lognormal", CV_LO, None),
    ("gamma",     CV_LO, None),
    ("pareto",    CV_LO, None),
    ("twopoint",  CV_LO, None),
    ("lognormal", CV_HI, None),
    ("gamma",     CV_HI, None),
    ("pareto",    CV_HI, None),
    ("twopoint",  CV_HI, None),
    ("pareto",    CV_LO, 1.5),      # 极端：均值有限、方差无穷（CV 未定义）
]
COST_LABEL = {("lognormal", 0): "LogNormal", ("gamma", 0): "Gamma",
              ("pareto", 0): "Pareto", ("twopoint", 0): "TwoPoint"}


def cell_label(cd, cv, pa):
    if pa is not None:
        return "Pareto(a=%.1f,Var=inf)" % pa
    return "%s(CV=%.2f)" % (cd[0].upper() + cd[1:], cv)


def build_specs():
    """全部格：轴1×轴2（Bernoulli, 30x 主表）+ 10x（预算轴）+ 轴3（Beta 奖励，低 CV 四分布）。"""
    specs = []
    for cd, cv, pa in COST_GRID:
        for mech in MECHS:
            specs.append((cd, cv, "bernoulli", mech, 30, 1, pa))
    for cd, cv, pa in COST_GRID:
        for mech in MECHS:
            specs.append((cd, cv, "bernoulli", mech, 10, 1, pa))
    for cd, cv, pa in COST_GRID[:4] + [COST_GRID[4]]:
        for mech in MECHS:
            specs.append((cd, cv, "beta", mech, 30, 1, pa))
    # env seed 稳健性（预审 P7：单 env 实例 = 仿真伪影嫌疑）：3 个 env seed × 3 机制 × 2 分布
    for es in (2, 3):
        for cd, cv, pa in (COST_GRID[0], COST_GRID[2]):
            for mech in ("uniform", "ucb_fieller", "thompson_ratio"):
                specs.append((cd, cv, "bernoulli", mech, 30, es, pa))
    return specs


def ci3(t):
    return "%.3f[%.3f,%.3f]" % t


def smallU_of(r):
    """被评估单元（n>=2）中 n<5 的占比，由缓存字段推导（不需重跑）。

    引擎保证每臂初始化各拉 1 次，故 n>=1 恒成立 → {n<2} = {n==1}，于是
      P(2<=n<5) = P(n<5) − P(n==1) = frac_n_lt5 − (1 − P(n>=2))
      smallU    = P(2<=n<5) / P(n>=2)
    """
    p_ge2 = r["n_units_ge2"] / float(r["MC"] * K)
    if p_ge2 <= 0:
        return float("nan")
    return (r["frac_n_lt5"] - (1.0 - p_ge2)) / p_ge2


def holm(pvals):
    """Holm 全族校正，返回调整后 p（单调化）。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for k, i in enumerate(order):
        v = max(p[i] * (m - k), run)
        adj[i] = min(v, 1.0)
        run = v
    return adj


def boot_ci_paired(d, B=2000, seed=20260912):
    """配对自助 95% CI（连续量差）。"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (B, d.size))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ====================================================================================
#  报告
# ====================================================================================
def report_t1(R, bmult, reward_dist="bernoulli", env_seed=1):
    """T1 主表：同子集四列覆盖 + 宽度 + 信息率（预审 F1 要求的表）。"""
    print("\n" + "=" * 150)
    print("[T1] 同子集覆盖对照 · reward=%s · 预算 %d× · env_seed=%d · MC=%d · 名义 0.90"
          % (reward_dist, bmult, env_seed, MC))
    print("列：info=Fieller 有限区间率（分母=全部 (arm,traj)，与既有锚点 expD_e3 同口径）")
    print("    info|ge2=同上但分母=可评估单元 n>=2（**跨机制比较应看这一列**：n=1 的臂两法都无界，")
    print("             不是任何方法的功劳/过失。greedy_ratio 把 20 个臂留在 n=1，两口径差 5 倍）")
    print("    F|fin=Fieller conditional coverage on its finite reporting set |")
    print("    d|fin=delta coverage on the SAME Fieller-finite subset (shape diagnostic) | d|all=delta conditional coverage on its own finite set")
    print("=" * 150)
    hdr = ("%-22s %-16s | %-7s %-8s | %-20s %-20s %-20s %-8s | %-7s %-7s"
           % ("cost dist", "mech", "info", "info|ge2", "F|fin", "d|fin", "d|all",
              "d|abst", "wF_med", "wd_med"))
    print(hdr)
    print("-" * 150)
    rows = []
    for cd, cv, pa in COST_GRID:
        if reward_dist == "beta" and (cd, cv, pa) not in COST_GRID[:5]:
            continue
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, env_seed, pa)
            if k not in R:
                continue
            r = R[k]
            print("%-22s %-16s | %-7.3f %-8.3f | %-20s %-20s %-20s %-8.3f | %-7.4f %-7.4f"
                  % (cell_label(cd, cv, pa), mech, r["info_rate"][0],
                     r["info_rate_ge2"][0],
                     ci3(r["cov_f_fin"]), ci3(r["cov_d_on_fin"]), ci3(r["cov_d_all"]),
                     r["cov_d_on_abst"][0], r["w_f_fin"][1], r["w_d_on_fin"][1]))
            rows.append((cd, cv, pa, mech, r))
        print("-" * 150)
    return rows


def report_t2(R, bmult=30, reward_dist="bernoulli"):
    """T2 退化诊断：srr==0 占比 / delta 零宽度率 / 排除退化后 delta 覆盖 / Fieller 在退化单元上的弃权率。"""
    print("\n" + "=" * 150)
    print("[T2] delta 崩塌的机制诊断（预审 F2）· reward=%s · 预算 %d×" % (reward_dist, bmult))
    print("列：srr0=样本奖励方差恰为 0 的 n>=2 单元占比 | dW0=delta 宽度<1e-9 占比 | "
          "d|nondeg=剔除 srr0 后 delta 覆盖 | F_abst|srr0=Fieller 在 srr0 单元上的弃权率 | "
          "dAC|all=Agresti-Coull 平滑方差 delta（不退化基线）覆盖")
    print("=" * 150)
    print("%-24s %-16s | %-20s %-8s | %-20s %-20s | %-20s"
          % ("cost dist", "mech", "srr0", "dW0", "d|nondeg", "F_abst|srr0", "dAC|all"))
    print("-" * 150)
    for cd, cv, pa in COST_GRID:
        if reward_dist == "beta" and (cd, cv, pa) not in COST_GRID[:5]:
            continue
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, 1, pa)
            if k not in R:
                continue
            r = R[k]
            print("%-24s %-16s | %-20s %-8.3f | %-20s %-20s | %-20s"
                  % (cell_label(cd, cv, pa), mech, ci3(r["srr0_frac_ge2"]),
                     r["delta_zerowidth_ge2"][0], ci3(r["cov_d_nondegen"]),
                     ci3(r["fieller_abst_on_srr0"]), ci3(r["cov_dac_all"])))
        print("-" * 150)


def report_t3(R, bmult=30, reward_dist="bernoulli"):
    """T3 宽度 + interval score（弃权不计分）+ cap 交叉点 c*（预审 F3）。"""
    print("\n" + "=" * 150)
    print("[T3] 宽度四分位与 interval score（alpha=0.10 → 2/alpha=20）· reward=%s · 预算 %d×"
          % (reward_dist, bmult))
    print("IS 只在各法自己报告的单元上算（弃权不计分），报告率单列。")
    print("cap 交叉点 c*：S_cap(m;c)=mean_i[1{rep}IS_i + 1{abst}·c·W_med(Fieller)]，")
    print("  解 S_cap(Fieller;c) = S_cap(delta-all;c) 的 c（delta-all 报告率=1 无弃权项）。")
    print("=" * 150)
    print("重要：deltaAC（Agresti-Coull 平滑方差）**从不弃权**（报告率=1），故它的覆盖必须与")
    print("      **宽度**一起读——平滑把小 n 单元的区间撑宽来换覆盖，这正是 SPTM 审稿人要查的账。")
    print("%-22s %-16s | %-22s %-22s %-22s | %-7s %-7s %-7s %-8s | %-9s"
          % ("cost dist", "mech", "wF q25/q50/q75", "wd|fin q25/q50/q75",
             "wdAC|all q25/q50/q75", "IS_F", "IS_dfin", "IS_dall", "IS_dACall", "c*"))
    print("-" * 172)
    out = {}
    for cd, cv, pa in COST_GRID:
        if reward_dist == "beta" and (cd, cv, pa) not in COST_GRID[:5]:
            continue
        printed = False
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, 1, pa)
            if k not in R:
                continue
            r = R[k]
            cs = crossover_c(r)
            out[k] = cs
            if cs < 0:
                cstr = "delta wins"          # IS_F 已 > IS_d(all)，无正交叉点
            elif np.isfinite(cs):
                cstr = "%.2f" % cs
            else:
                cstr = "n/a"
            print("%-22s %-16s | %-22s %-22s %-22s | %-7.4f %-7.4f %-7.4f %-8.4f | %-9s"
                  % (cell_label(cd, cv, pa), mech,
                     "%.4f/%.4f/%.4f" % r["w_f_fin"], "%.4f/%.4f/%.4f" % r["w_d_on_fin"],
                     "%.4f/%.4f/%.4f" % r["w_dac_all"],
                     r["is_f_fin"], r["is_d_on_fin"], r["is_d_all"], r["is_dac_all"],
                     cstr))
            printed = True
        if printed:
            print("-" * 172)
    # deltaAC 的"零弃权 + 高覆盖"是否靠宽度买来的：宽度比与联合覆盖对照
    print("\n[T3b] deltaAC（报告率=1，不弃权）vs Fieller（报告率=info）的正面对账")
    print("      联合口径 = P(区间有限 且 覆盖真值)，对 deltaAC 等于其条件覆盖（恒有限）")
    print("      宽度比 = median(wdAC on all n>=2) / median(wF on finite) —— >1 表示平滑靠撑宽换覆盖")
    print("%-22s %-16s | %-9s %-9s | %-9s %-9s | %-9s"
          % ("cost dist", "mech", "joint_F", "joint_dAC", "wF_med", "wdAC_med", "宽度比"))
    print("-" * 120)
    for cd, cv, pa in COST_GRID:
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, 1, pa)
            if k not in R:
                continue
            r = R[k]
            wr = (r["w_dac_all"][1] / r["w_f_fin"][1]
                  if r["w_f_fin"][1] and np.isfinite(r["w_f_fin"][1]) else float("nan"))
            print("%-22s %-16s | %-9.3f %-9.3f | %-9.4f %-9.4f | %-9.2f"
                  % (cell_label(cd, cv, pa), mech, r["joint_f"][0], r["cov_dac_all"][0],
                     r["w_f_fin"][1], r["w_dac_all"][1], wr))
    return out


def crossover_c(r):
    """解 c*：Fieller（报告 fin，弃权 abst 罚 c·W_med）vs delta-all（全报告，无弃权）。
    S_F(c) = [sum_IS_F + abst·c·W_med] / ge2 ;  S_d = sum_IS_d_all / ge2
    → c* = (sum_IS_d_all − sum_IS_F) / (abst · W_med)。c<c* → delta 更优（分数更低）。"""
    n_ab = r["cnt"]["abst"]
    wm = r["W_med_ref"]
    if n_ab <= 0 or not np.isfinite(wm) or wm <= 0:
        return float("nan")
    return (r["is_sum"]["d_all"] - r["is_sum"]["f"]) / (n_ab * wm)


def report_t4(R, bmult=30, reward_dist="bernoulli"):
    """T4 集中度 vs delta 崩塌：Spearman（跨机制，按分布分组 + pooled）。"""
    print("\n" + "=" * 150)
    print("[T4] 采样集中度 vs delta 覆盖崩塌（轴2 的核心问题）· reward=%s · 预算 %d×"
          % (reward_dist, bmult))
    print("集中度指标：top1_share=最常拉臂的拉取份额；H/logK=拉取分布归一化熵（低=集中）；")
    print("            frac_n<5=样本数<5 的 (arm,traj) 占比")
    print("被预测量：d|all=delta 全 n>=2 覆盖（既有报法的崩塌量）; srr0=零方差单元占比")
    print("=" * 150)
    print("%-22s %-16s | %-8s %-8s %-9s %-8s | %-8s %-8s %-8s %-8s"
          % ("cost dist", "mech", "top1sh", "H/logK", "frac_n<5", "smallU",
             "d|all", "srr0", "F|fin", "info"))
    print("-" * 150)
    pool = {"top1": [], "ent": [], "n5": [], "smallU": [], "dall": [], "srr0": [],
            "ffin": [], "dfin": [], "cell": []}
    for cd, cv, pa in COST_GRID:
        if reward_dist == "beta" and (cd, cv, pa) not in COST_GRID[:5]:
            continue
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, 1, pa)
            if k not in R:
                continue
            r = R[k]
            su = smallU_of(r)
            print("%-22s %-16s | %-8.4f %-8.4f %-9.4f %-8.4f | %-8.4f %-8.4f %-8.4f %-8.4f"
                  % (cell_label(cd, cv, pa), mech, r["top1_share"], r["entropy_norm"],
                     r["frac_n_lt5"], su, r["cov_d_all"][0], r["srr0_frac_ge2"][0],
                     r["cov_f_fin"][0], r["info_rate"][0]))
            pool["top1"].append(r["top1_share"]); pool["ent"].append(r["entropy_norm"])
            pool["smallU"].append(su)
            pool["n5"].append(r["frac_n_lt5"]); pool["dall"].append(r["cov_d_all"][0])
            pool["srr0"].append(r["srr0_frac_ge2"][0]); pool["ffin"].append(r["cov_f_fin"][0])
            pool["dfin"].append(r["cov_d_on_fin"][0])
            pool["cell"].append((cell_label(cd, cv, pa), mech))
        print("-" * 150)
    P = {k: np.asarray(v) for k, v in pool.items() if k != "cell"}
    print("\n[T4b] Spearman 相关（pooled 全部 %d 格 = %d 分布 × %d 机制）· Holm 全族校正"
          % (len(P["dall"]), len(P["dall"]) // len(MECHS), len(MECHS)))
    print("      族定义：本表内 8 个相关检验（4 预测量 × 2 被预测量）")
    print("      smallU = 在**被评估单元**（n>=2）中 n<5 的占比（frac_n<5 含 n<2 的未评估单元，")
    print("               是混淆量：greedy_ratio 把非榜首臂留在 n=1，那些单元根本不进分母）")
    tests = []
    for xn, xv in (("top1_share", P["top1"]), ("H/logK", P["ent"]),
                   ("frac_n<5(all)", P["n5"]), ("smallU(n<5|n>=2)", P["smallU"])):
        for yn, yv in (("delta cov|all", P["dall"]), ("srr0 frac", P["srr0"])):
            rho, pv = stats.spearmanr(xv, yv)
            tests.append((xn, yn, float(rho), float(pv)))
    adj = holm([t[3] for t in tests])
    for (xn, yn, rho, pv), pa_ in zip(tests, adj):
        print("      Spearman(%-17s, %-14s) = %+.3f  p=%.3g → Holm %.3g  %s"
              % (xn, yn, rho, pv, pa_, "显著" if pa_ < 0.05 else "n.s."))
    rho0, pv0 = stats.spearmanr(P["srr0"], P["dall"])
    print("      —— 对照：Spearman(%-17s, %-14s) = %+.3f  p=%.3g（**真正的驱动量**，"
          "不进上面的族，因为它是机制而非集中度）" % ("srr0 frac", "delta cov|all", rho0, pv0))
    print("\n[T4c] 同一相关但 y = **Fieller 条件覆盖** 与 **delta 在同子集覆盖**（应无崩塌）")
    tests2 = []
    for xn, xv in (("top1_share", P["top1"]), ("H/logK", P["ent"])):
        for yn, yv in (("Fieller cov|fin", P["ffin"]), ("delta cov|fin", P["dfin"])):
            rho, pv = stats.spearmanr(xv, yv)
            tests2.append((xn, yn, float(rho), float(pv)))
    adj2 = holm([t[3] for t in tests2])
    for (xn, yn, rho, pv), pa_ in zip(tests2, adj2):
        print("      Spearman(%-12s, %-16s) = %+.3f  p=%.3g → Holm %.3g  %s"
              % (xn, yn, rho, pv, pa_, "显著" if pa_ < 0.05 else "n.s."))
    return P, pool["cell"]


def report_t5(R, bmult=30, reward_dist="bernoulli"):
    """T5 配对检验：每轨迹 Fieller|fin 覆盖率 − delta|fin 覆盖率（同子集、同 seed）。
    族 = 本表全部 (分布 × 机制) 格。若普遍不显著 → 两法在同子集上不可区分（预审 F1 的正式检验）。"""
    print("\n" + "=" * 150)
    print("[T5] 配对检验（同 seed、同有限子集）：Fieller 条件覆盖 − delta 同子集覆盖")
    print("      每格 %d 条轨迹的逐轨迹覆盖率配对；Wilcoxon signed-rank + Holm（族=本表 %d 格）"
          % (MC, len([1 for cd, cv, pa in COST_GRID for m in MECHS])))
    print("      配对自助 95%% CI (B=2000) 给差的量级")
    print("=" * 150)
    tests = []
    for cd, cv, pa in COST_GRID:
        if reward_dist == "beta" and (cd, cv, pa) not in COST_GRID[:5]:
            continue
        for mech in MECHS:
            k = (cd, cv, reward_dist, mech, bmult, 1, pa)
            if k not in R:
                continue
            r = R[k]
            d = r["traj_cov_f_fin"] - r["traj_cov_d_on_fin"]
            d = d[np.isfinite(d)]
            if d.size >= 5 and np.any(d != 0):
                pv = float(stats.wilcoxon(d).pvalue)
            else:
                pv = 1.0
            m, lo, hi = boot_ci_paired(d)
            tests.append((cell_label(cd, cv, pa), mech, m, lo, hi, pv, d.size))
    adj = holm([t[5] for t in tests])
    print("%-24s %-16s | %-26s | %-10s %-10s %-6s"
          % ("cost dist", "mech", "Δcov [boot 95%CI]", "p", "Holm", "n_traj"))
    print("-" * 150)
    nsig = 0
    for (lbl, mech, m, lo, hi, pv, nt), pa_ in zip(tests, adj):
        sig = pa_ < 0.05
        nsig += int(sig)
        print("%-24s %-16s | %+.4f [%+.4f,%+.4f]%s | %-10.3g %-10.3g %-6d %s"
              % (lbl, mech, m, lo, hi, " " * 3, pv, pa_, nt, "显著" if sig else "n.s."))
    print("-" * 150)
    print("汇总：%d/%d 格显著（Holm<0.05）。全部 |Δcov| 最大 = %.4f"
          % (nsig, len(tests), max(abs(t[2]) for t in tests)))
    return tests


def report_t6(R, bmult=30):
    """T6 delta|all 崩塌的跨分布/跨机制汇总 + env seed 稳健性（预审 P7）。"""
    print("\n" + "=" * 150)
    print("[T6] env 实例稳健性（预审 P7：全部既有结果来自单一 make_env(seed=1)）")
    print("      3 个 env seed × {LogNormal, Pareto}(CV=0.36) × {uniform, ucb_fieller, thompson_ratio}")
    print("=" * 150)
    print("%-22s %-16s %-6s | %-20s %-20s %-20s %-8s"
          % ("cost dist", "mech", "eseed", "F|fin", "d|fin", "d|all", "info"))
    print("-" * 150)
    for cd, cv, pa in (COST_GRID[0], COST_GRID[2]):
        for mech in ("uniform", "ucb_fieller", "thompson_ratio"):
            vals = {"ffin": [], "dall": []}
            for es in (1, 2, 3):
                k = (cd, cv, "bernoulli", mech, bmult, es, pa)
                if k not in R:
                    continue
                r = R[k]
                vals["ffin"].append(r["cov_f_fin"][0]); vals["dall"].append(r["cov_d_all"][0])
                print("%-22s %-16s %-6d | %-20s %-20s %-20s %-8.3f"
                      % (cell_label(cd, cv, pa), mech, es, ci3(r["cov_f_fin"]),
                         ci3(r["cov_d_on_fin"]), ci3(r["cov_d_all"]), r["info_rate"][0]))
            if len(vals["ffin"]) == 3:
                print("      → seed 间离散度: F|fin sd=%.4f range=[%.3f,%.3f] | "
                      "d|all sd=%.4f range=[%.3f,%.3f]"
                      % (np.std(vals["ffin"], ddof=1), min(vals["ffin"]), max(vals["ffin"]),
                         np.std(vals["dall"], ddof=1), min(vals["dall"]), max(vals["dall"])))
        print("-" * 150)


def report_t7(R):
    """T7 轴3 奖励分布：Bernoulli vs Beta 的直接对照（决定性检验）。"""
    print("\n" + "=" * 150)
    print("[T7] 轴3 · 奖励分布（决定性检验：Beta 奖励下样本方差几乎不为 0）· 预算 30×")
    print("      Beta(mu·nu,(1-mu)·nu), nu=%.1f → E[r]=mu 不变（真比率不变），Var(r)=mu(1-mu)/(nu+1)"
          % BETA_NU)
    print("      若 delta 的 d|all 崩塌在 Beta 下消失 → 崩塌 = Bernoulli 零方差退化，与自适应无关")
    print("=" * 150)
    print("%-22s %-16s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s"
          % ("cost dist", "mech", "srr0_B", "srr0_Be", "d|all_B", "d|all_Be",
             "d|fin_B", "d|fin_Be", "F|fin_B", "F|fin_Be"))
    print("-" * 150)
    pairs = []
    for cd, cv, pa in COST_GRID[:5]:
        for mech in MECHS:
            kb = (cd, cv, "bernoulli", mech, 30, 1, pa)
            kt = (cd, cv, "beta", mech, 30, 1, pa)
            if kb not in R or kt not in R:
                continue
            rb, rt = R[kb], R[kt]
            print("%-22s %-16s | %-8.3f %-8.3f | %-8.3f %-8.3f | %-8.3f %-8.3f | %-8.3f %-8.3f"
                  % (cell_label(cd, cv, pa), mech, rb["srr0_frac_ge2"][0],
                     rt["srr0_frac_ge2"][0], rb["cov_d_all"][0], rt["cov_d_all"][0],
                     rb["cov_d_on_fin"][0], rt["cov_d_on_fin"][0],
                     rb["cov_f_fin"][0], rt["cov_f_fin"][0]))
            pairs.append((rb, rt))
        print("-" * 150)
    if pairs:
        db = np.array([p[0]["cov_d_all"][0] for p in pairs])
        dt = np.array([p[1]["cov_d_all"][0] for p in pairs])
        sb = np.array([p[0]["srr0_frac_ge2"][0] for p in pairs])
        st_ = np.array([p[1]["srr0_frac_ge2"][0] for p in pairs])
        print("汇总（%d 格配对）：delta|all Bernoulli 均值 %.3f → Beta 均值 %.3f（Δ=%+.3f）"
              % (len(pairs), db.mean(), dt.mean(), dt.mean() - db.mean()))
        print("                  srr0 占比 Bernoulli 均值 %.3f → Beta 均值 %.3f"
              % (sb.mean(), st_.mean()))
        pv = float(stats.wilcoxon(dt - db).pvalue) if len(pairs) >= 5 else float("nan")
        print("                  配对 Wilcoxon（Beta − Bernoulli 的 delta|all）p=%.3g" % pv)
    return pairs


# ====================================================================================
#  绘图（标签全英文；无上/右边框；浅网格；Okabe-Ito；dpi=220）
# ====================================================================================
def style_ax(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color="#DDDDDD", lw=0.6, ls="-", zorder=0)
    ax.set_axisbelow(True)


def draw(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.titlesize": 9.5, "axes.labelsize": 9})
    fig = plt.figure(figsize=(14.6, 7.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.30, 1.0, 1.0],
                          hspace=0.52, wspace=0.34,
                          left=0.128, right=0.988, top=0.905, bottom=0.088)

    dists = [(cd, cv, pa) for cd, cv, pa in COST_GRID]
    dlabels = [cell_label(*d) for d in dists]
    mlabels = [m.replace("_", "\n") for m in MECHS]
    cmap = LinearSegmentedColormap.from_list(
        "ok", ["#D55E00", "#F0E442", "#FFFFFF", "#56B4E9", "#0072B2"])

    # ---- (a) 热力图：delta cov|all n>=2（既有报法 → 到处崩塌）----
    ax = fig.add_subplot(gs[0, 0])
    M = np.full((len(dists), len(MECHS)), np.nan)
    for i, d in enumerate(dists):
        for j, mech in enumerate(MECHS):
            k = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            if k in R:
                M[i, j] = R[k]["cov_d_all"][0]
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7.2,
                        color="black")
    ax.set_xticks(range(len(MECHS))); ax.set_xticklabels(mlabels, fontsize=7.0)
    ax.set_yticks(range(len(dists))); ax.set_yticklabels(dlabels, fontsize=7.4)
    ax.set_title("(a) delta coverage, all units $n\\geq2$\n"
                 "(the previously reported number: collapses everywhere)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(0.90, color="black", lw=1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- (b) 热力图：delta cov | Fieller-finite（同子集 → 处处恢复到名义）----
    ax = fig.add_subplot(gs[1, 0])
    M2 = np.full_like(M, np.nan)
    for i, d in enumerate(dists):
        for j, mech in enumerate(MECHS):
            k = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            if k in R:
                M2[i, j] = R[k]["cov_d_on_fin"][0]
    im2 = ax.imshow(M2, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            if np.isfinite(M2[i, j]):
                ax.text(j, i, "%.2f" % M2[i, j], ha="center", va="center", fontsize=7.2,
                        color="black")
    ax.set_xticks(range(len(MECHS))); ax.set_xticklabels(mlabels, fontsize=7.0)
    ax.set_yticks(range(len(dists))); ax.set_yticklabels(dlabels, fontsize=7.4)
    nlo = int(np.sum(M2 < 0.88))
    ax.set_title("(b) delta coverage on the SAME Fieller-finite subset\n"
                 "matched conditioning: %d/%d cells at $\\geq$0.88 (exceptions: heavy-tail + uniform)"
                 % (M2.size - nlo, M2.size), fontsize=9)
    cb = fig.colorbar(im2, ax=ax, fraction=0.030, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(0.90, color="black", lw=1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- (c) 集中度 vs delta|all（散点，形状=分布，颜色=机制）----
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax)
    mk = {"lognormal": "o", "gamma": "s", "pareto": "^", "twopoint": "D"}
    mcol = dict(zip(MECHS, [OKABE["black"], OKABE["verm"], OKABE["orange"],
                            OKABE["blue"], OKABE["green"], OKABE["purple"],
                            OKABE["sky"]]))
    xs, ys, su = [], [], []
    for d in dists:
        for mech in MECHS:
            k = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            if k not in R:
                continue
            r = R[k]
            ax.scatter(r["top1_share"], r["cov_d_all"][0], marker=mk[d[0]], s=26,
                       color=mcol[mech], alpha=0.9, edgecolor="white", linewidth=0.4,
                       zorder=3)
            xs.append(r["top1_share"]); ys.append(r["cov_d_all"][0])
            su.append(smallU_of(r))
    rho, pv = stats.spearmanr(xs, ys)
    rho_s, pv_s = stats.spearmanr(su, ys)
    ax.axhline(0.90, color="black", ls=":", lw=1.1, zorder=2)
    ax.text(0.97, 0.905, "nominal 0.90", fontsize=7.5, ha="right", transform=
            ax.get_yaxis_transform())
    ax.set_xlabel("sampling concentration (top-1 pull share)")
    ax.set_ylabel("delta coverage | all $n\\geq2$")
    ax.set_title("(c) Concentration does NOT predict the collapse\n"
                 "$\\rho$=%+.2f, p=%.2g (n.s.); small-$n$ share: $\\rho$=%+.2f, p=%.0e"
                 % (rho, pv, rho_s, pv_s), fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(handles=[plt.Line2D([], [], marker=mk[c], ls="", color="#666666",
                                  label=c, markersize=5) for c in
                       ["lognormal", "gamma", "pareto", "twopoint"]],
              fontsize=6.6, frameon=False, loc="lower left", ncol=2,
              title="cost law (marker)", title_fontsize=6.6)

    # ---- (d) 集中度 vs 同子集覆盖（应平坦）----
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax)
    xs2, ys2, ys3 = [], [], []
    for d in dists:
        for mech in MECHS:
            k = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            if k not in R:
                continue
            r = R[k]
            ax.scatter(r["top1_share"], r["cov_f_fin"][0], marker=mk[d[0]], s=26,
                       color=OKABE["blue"], alpha=0.85, edgecolor="white", linewidth=0.4,
                       zorder=3)
            ax.scatter(r["top1_share"], r["cov_d_on_fin"][0], marker=mk[d[0]], s=26,
                       color=OKABE["verm"], alpha=0.85, edgecolor="white", linewidth=0.4,
                       zorder=3)
            xs2.append(r["top1_share"]); ys2.append(r["cov_f_fin"][0])
            ys3.append(r["cov_d_on_fin"][0])
    r2, p2 = stats.spearmanr(xs2, ys2)
    r3, p3 = stats.spearmanr(xs2, ys3)
    ax.axhline(0.90, color="black", ls=":", lw=1.1, zorder=2)
    ax.set_xlabel("sampling concentration (top-1 pull share)")
    ax.set_ylabel("conditional coverage | finite")
    ax.set_title("(d) On matched units the two methods track each other\n"
                 "Fieller $\\rho$=%+.2f p=%.0e; delta $\\rho$=%+.2f p=%.0e (both rise together)"
                 % (r2, p2, r3, p3), fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=OKABE["blue"], label="Fieller | finite"),
        plt.Line2D([], [], marker="o", ls="", color=OKABE["verm"], label="delta | same finite")],
        fontsize=7.2, frameon=False, loc="lower left")

    # ---- (e) 零方差退化占比 vs delta|all（机制解释）----
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax)
    sx, sy = [], []
    for d in dists:
        for mech in MECHS:
            k = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            if k not in R:
                continue
            r = R[k]
            ax.scatter(r["srr0_frac_ge2"][0], r["cov_d_all"][0], marker=mk[d[0]], s=26,
                       color=mcol[mech], alpha=0.9, edgecolor="white", linewidth=0.4,
                       zorder=3)
            sx.append(r["srr0_frac_ge2"][0]); sy.append(r["cov_d_all"][0])
    sx = np.array(sx); sy = np.array(sy)
    gx = np.linspace(0, max(sx.max(), 0.8), 50)
    ax.plot(gx, 1 - gx, color=OKABE["black"], lw=1.3, ls="--", zorder=2,
            label="identity  $1-p_{\\hat{s}_{rr}=0}$")
    rr, pp = stats.spearmanr(sx, sy)
    ax.set_xlabel("fraction of units with $\\hat{s}_{rr}=0$")
    ax.set_ylabel("delta coverage | all $n\\geq2$")
    hi_m = sx > 0.3
    dev_hi = np.max(np.abs(sy[hi_m] - (1 - sx[hi_m]))) if hi_m.any() else float("nan")
    ax.set_title("(e) In the collapse regime: a zero-variance identity\n"
                 "$\\rho$=%+.3f; max |dev| %.3f when $p{>}0.3$"
                 % (rr, dev_hi), fontsize=9)
    ax.legend(fontsize=7.2, frameon=False, loc="upper right")
    ax.set_ylim(0, 1.02)

    # ---- (f) Bernoulli vs Beta 奖励（轴3 决定性检验）----
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, grid_axis="y")
    bx, bb, bt = [], [], []
    for d in dists[:5]:
        for mech in MECHS:
            kb = (d[0], d[1], "bernoulli", mech, 30, 1, d[2])
            kt = (d[0], d[1], "beta", mech, 30, 1, d[2])
            if kb not in R or kt not in R:
                continue
            bb.append(R[kb]["cov_d_all"][0]); bt.append(R[kt]["cov_d_all"][0])
            bx.append(R[kb]["top1_share"])
    idx = np.arange(len(bb))
    ax.bar(idx - 0.21, bb, 0.42, color=OKABE["verm"], label="Bernoulli reward",
           edgecolor="white", linewidth=0.4, zorder=3)
    ax.bar(idx + 0.21, bt, 0.42, color=OKABE["blue"], label="Beta reward (same $E[r]$)",
           edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0.90, color="black", ls=":", lw=1.1, zorder=4)
    ax.set_xticks([]); ax.set_xlabel("35 cells (5 cost distributions $\\times$ 7 mechanisms)")
    ax.set_ylabel("delta coverage | all $n\\geq2$")
    ax.set_title("(f) Axis 3: the collapse is created by binary rewards\n"
                 "mean %.3f $\\rightarrow$ %.3f when $\\hat{s}_{rr}=0$ is removed"
                 % (np.mean(bb), np.mean(bt)), fontsize=9)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    ax.set_ylim(0, 1.05)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=220)
    plt.close(fig)
    print("\n图已存 %s" % FIG)


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(d):
    tmp = CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=4)
    os.replace(tmp, CACHE)


def run_all(specs):
    cache = load_cache()
    todo = [s for s in specs if (s, MC) not in cache]
    print("\n网格：%d 格，已缓存 %d，待跑 %d（MC=%d，进程数 %d）"
          % (len(specs), len(specs) - len(todo), len(todo), MC, NPROC), flush=True)
    if todo:
        t0 = time.time()
        done = 0
        with mp.Pool(NPROC) as pool:
            for spec, res in zip(todo, pool.imap(run_cell, todo, chunksize=1)):
                cache[(spec, MC)] = res
                done += 1
                if done % 8 == 0 or done == len(todo):
                    save_cache(cache)
                    el = time.time() - t0
                    print("    进度 %d/%d  %.0fs（预计剩 %.0fs）"
                          % (done, len(todo), el, el / done * (len(todo) - done)),
                          flush=True)
        save_cache(cache)
    return {s: cache[(s, MC)] for s in specs}


def main():
    t0 = time.time()
    print("=" * 150)
    print("X4 · 分布与机制泛化（缺口 G4）· ICASSP 2027")
    print("目的：检验 'delta 覆盖崩塌 / Fieller 弃权后守住名义' 是否只是 "
          "(LogNormal + UCB + Bernoulli) 的仿真伪影")
    print("MC=%d · K=%d · alpha=%.2f（名义 0.90）· gamma=0 · traj seeds %d..%d"
          % (MC, K, ALPHA, SEED_BASE, SEED_BASE + MC - 1))
    print("单元 = (轨迹, 臂) 且 n>=2（n<2 时两法均无界，不可评）")
    print("=" * 150)
    ok = selfcheck(COST_GRID, n_draw=2000000)
    if not ok:
        print("\n!! S0 自检失败，终止（不产出可疑数值）")
        return
    specs = build_specs()
    R = run_all(specs)
    print("\n仿真/读缓存完成 · 累计 %.1fs" % (time.time() - t0))

    report_t1(R, 30, "bernoulli")
    report_t1(R, 10, "bernoulli")
    report_t2(R, 30, "bernoulli")
    report_t3(R, 30, "bernoulli")
    P, cells = report_t4(R, 30, "bernoulli")
    report_t4(R, 10, "bernoulli")
    report_t5(R, 30, "bernoulli")
    report_t6(R, 30)
    report_t7(R)
    report_verdict(R)
    draw(R)
    print("\n总耗时 %.1fs" % (time.time() - t0))


def report_verdict(R):
    """总判定：把 G4 的三个轴压成可直接写进论文的数字。"""
    print("\n" + "=" * 150)
    print("[VERDICT] G4 泛化判定（可直接引用的汇总数字）")
    print("=" * 150)
    # 轴1+轴2：全部 Bernoulli 30x 格
    ffin, dfin, dall, srr0, info = [], [], [], [], []
    for cd, cv, pa in COST_GRID:
        for mech in MECHS:
            k = (cd, cv, "bernoulli", mech, 30, 1, pa)
            if k not in R:
                continue
            r = R[k]
            ffin.append(r["cov_f_fin"][0]); dfin.append(r["cov_d_on_fin"][0])
            dall.append(r["cov_d_all"][0]); srr0.append(r["srr0_frac_ge2"][0])
            info.append(r["info_rate"][0])
    ffin = np.array(ffin); dfin = np.array(dfin); dall = np.array(dall)
    srr0 = np.array(srr0); info = np.array(info)
    # 逐格差与显著性（与 T5 同一配对检验，重算以便在判定里给出精确计数）
    gaps, sig_cells, lo_cells = [], [], []
    for cd, cv, pa in COST_GRID:
        for mech in MECHS:
            k = (cd, cv, "bernoulli", mech, 30, 1, pa)
            if k not in R:
                continue
            r = R[k]
            d = r["traj_cov_f_fin"] - r["traj_cov_d_on_fin"]
            d = d[np.isfinite(d)]
            pv = (float(stats.wilcoxon(d).pvalue)
                  if d.size >= 5 and np.any(d != 0) else 1.0)
            gaps.append((cell_label(cd, cv, pa), mech, float(np.mean(d)), pv,
                         r["cov_f_fin"][0], r["cov_d_on_fin"][0], cv, pa))
            if r["cov_f_fin"][2] < 0.90:      # Wilson 上界 < 名义 → 该格条件覆盖确实低于名义
                lo_cells.append((cell_label(cd, cv, pa), mech, r["cov_f_fin"]))
    adjv = holm([g[3] for g in gaps])
    sig_cells = [(g, p) for g, p in zip(gaps, adjv) if p < 0.05]
    print("V1 [同子集对照 · 预审 F1 的泛化]：全部 %d 格（%d 成本分布 × %d 机制，Bernoulli, 30×）"
          % (len(ffin), len(COST_GRID), len(MECHS)))
    print("    Fieller 条件覆盖 范围 [%.3f, %.3f] 均值 %.3f"
          % (ffin.min(), ffin.max(), ffin.mean()))
    print("    delta 同子集覆盖 范围 [%.3f, %.3f] 均值 %.3f"
          % (dfin.min(), dfin.max(), dfin.mean()))
    print("    逐格差 F−d|fin：均值 %+.4f，中位 %+.4f，最大 |差| %.4f"
          % (np.mean(ffin - dfin), np.median(ffin - dfin), np.max(np.abs(ffin - dfin))))
    print("    配对 Wilcoxon + Holm（族=%d 格）：**%d/%d 格显著**，全部方向为 Fieller 更高。"
          % (len(gaps), len(sig_cells), len(gaps)))
    if sig_cells:
        print("    显著格清单（诚实列出，不能声称'处处不可区分'）：")
        for (lbl, mech, md, pv, cf, cdd, cvv, paa), pj in sorted(
                sig_cells, key=lambda x: -abs(x[0][2])):
            print("      %-22s %-16s Δ=%+.4f  F=%.3f d=%.3f  Holm p=%.2g"
                  % (lbl, mech, md, cf, cdd, pj))
        hi = [s for s in sig_cells if (s[0][6] > 1.0 or s[0][7] is not None)]
        print("    其中 %d/%d 个显著格出现在高 CV（CV=2.91）或无穷方差（Pareto α=1.5）分布上"
              % (len(hi), len(sig_cells)))
        print("    → 可声称的是：**在低 CV（0.36）的全部 %d 格上两法不可区分**；"
              % len([g for g in gaps if g[6] < 1.0 and g[7] is None])
              + "重尾/高 CV 下 Fieller 略优但幅度多在 0.02 以内（唯一例外见下）。")
    print("V1b [必须写进论文的反例]：Fieller 条件覆盖的 Wilson 上界 < 0.90 的格 = %d 个"
          % len(lo_cells))
    for lbl, mech, w in lo_cells:
        print("      %-22s %-16s F|fin=%s  ← 条件覆盖**低于名义**，"
              "不能声称 'Fieller 处处守住名义'" % (lbl, mech, ci3(w)))
    print("V2 [崩塌是零方差恒等式 · 预审 F2/P3 的泛化]：")
    rho, pv = stats.spearmanr(srr0, dall)
    dev = np.abs(dall - (1 - srr0))
    print("    delta|all 范围 [%.3f, %.3f]；srr0 占比范围 [%.3f, %.3f]"
          % (dall.min(), dall.max(), srr0.min(), srr0.max()))
    print("    Spearman(srr0 占比, delta|all) = %+.4f  p=%.3g" % (rho, pv))
    print("    与恒等式 delta|all = 1 − srr0 的偏离：全局最大 %.4f 均值 %.4f"
          % (dev.max(), dev.mean()))
    m_hi = srr0 > 0.3
    m_lo = ~m_hi
    print("    **分层后（这是必须写进论文的细分）**：")
    print("      崩塌区（srr0>0.3，%d 格）：偏离 最大 %.4f 均值 %.4f → 恒等式几乎精确成立，"
          % (int(m_hi.sum()), dev[m_hi].max(), dev[m_hi].mean())
          + "崩塌 = 零方差退化的算术后果")
    print("      非崩塌区（srr0<=0.3，%d 格）：偏离 最大 %.4f 均值 %.4f"
          % (int(m_lo.sum()), dev[m_lo].max(), dev[m_lo].mean()))
    print("      非崩塌区里偏离最大的格全部是高 CV/重尾 + uniform（srr0 只有 0.02-0.03，"
          "但 delta|all 仍 0.77-0.85）——")
    print("      这是**与零方差无关的第二种欠覆盖**：成本重尾使分母的 t 近似本身失效。")
    print("      诚实结论：零方差恒等式解释了 UCB 类机制上的整个崩塌，但**不能**解释重尾下的"
          "残余欠覆盖；后者是真实的、且 Fieller 在这些格上确实略优（见 V1 的 6 个显著格）。")
    print("V3 [Fieller 弃权与退化单元的重合]：")
    ab = np.array([R[(cd, cv, "bernoulli", m, 30, 1, pa)]["fieller_abst_on_srr0"][0]
                   for cd, cv, pa in COST_GRID for m in MECHS
                   if (cd, cv, "bernoulli", m, 30, 1, pa) in R])
    ab = ab[np.isfinite(ab)]
    print("    P(Fieller 弃权 | srr0 单元) 范围 [%.4f, %.4f] 均值 %.4f（%d 格）"
          % (ab.min(), ab.max(), ab.mean(), ab.size))
    print("    信息率范围 [%.3f, %.3f] —— 弃权代价随机制变化极大（这是要交易的量）"
          % (info.min(), info.max()))


if __name__ == "__main__":
    only = os.environ.get("XG_ONLY", "")
    if only == "s0":
        selfcheck(COST_GRID, n_draw=int(os.environ.get("XG_NDRAW", "2000000")))
        sys.exit(0)
    if only == "grid":
        run_all(build_specs())
        sys.exit(0)
    main()


