# -*- coding: utf-8 -*-
"""
================================================================================
expD_lib.py · 参数化成本感知调度实验引擎（库，被 import，不可执行实验）
================================================================================
ESEC/FSE 投稿调度实验的参数化引擎。把 expC 的硬编码常量提成函数参数，并入
审稿人基线并把基线实现校准到文献原文（两处硬伤修复，见下）。机制层
（fieller_ci / delta_ci / run_trajectory 核心 / env 构造 / mu_at 语义）逐字复用
expC_cost_sched.py，不做重写。

两处基线硬伤修复（审稿人会直接判"基线错误"的点）：
  1) 原 `kube` 不是 Tran-Thanh 的 KUBE，只是"预算配额均分 + 轮转"的朴素基线。
     已改名为 budget_rr（保留 kube 作 deprecated 别名），并新增 kube_knapsack
     ——真 KUBE（AAAI 2012, arXiv:1204.1909, Algorithm 1, Eq. 6-7）：每步解近似
     无界背包、按密度序贪心分配、以 P(i)=m*_i/Σm* 拉臂。
  2) 原 `ucb_bv` 的探索项公式错误。已校正为 UCB-BV1 / UCB-BV2 两变体
     （Ding et al., AAAI 2013）：分母保正（λ > √(ln(t-1)/n_i) 才有效），
     否则该臂索引 +inf 强制探索；原 ucb_bv 作 ucb_bv1 的 deprecated 别名。

与 expC 的兼容性保证（回归校验的目标）：
  * make_env(...) 在默认参数下产生与 exp_params / ns_env_params 完全一致的 env dict，
    且 base c0/mu0 与 expC 模块级全局 c0/mu0 逐位一致（经 __main__ 回归断言验证）。
  * run(env, strat, B, seed) 对 8 个既有策略的 RNG 消耗顺序与 run_trajectory 完全一致，
    因此 Panel A / C 数值可逐格对齐（budget_rr 的 seed 口径单独处理，见 __main__）。

新基线说明（策略级，均有注释来源）：
  * budget_rr      朴素预算配额轮转基线（原误标为 KUBE；不是 Tran-Thanh 的 KUBE）：
                   每臂等额预算 B/K，轮转拉取直到各自配额耗尽。保留 kube 作 deprecated 别名。
  * kube_knapsack  真 KUBE（Tran-Thanh et al., Knapsack Based Optimal Policies for
                   Budget-Limited Multi-Armed Bandits, AAAI 2012, arXiv:1204.1909,
                   Algorithm 1, Eq. 6-7）：每步密度序贪心解近似无界背包，
                   按 m* 比例拉臂，剩余预算 < min ĉ 即停。成本未知 → 用样本均值 ĉ 估计
                   （必要偏离：原算法假设成本已知，见 _run_kube_knapsack 注释）。
  * ucb_bv1         UCB-BV1（Ding et al., Multi-Armed Bandit with Budget Constraint
                   and Variable Costs, AAAI 2013, Eq. 3）：
                   D_i = r̄_i/ĉ_i + (1+1/λ)·√(ln(t-1)/n_i)/(λ − √(ln(t-1)/n_i))，
                   λ = 成本期望的已知下界（此处取 0.8·min_i c_i，给基线最优参数）。
  * ucb_bv2         UCB-BV2（同上 Eq. 4）：
                   D_i = r̄_i/ĉ_i + (1/λ_t)·(1 + 1/(λ_t − √(ln(t-1)/n_i)))·√(ln(t-1)/n_i)，
                   λ_t = min_i ĉ_{i,t}（数据驱动）。BV1/BV2 分母必须保正，
                   λ ≤ √(ln(t-1)/n_i) 时该臂索引 = +inf（强制探索）。
                   原 ucb_bv 为错误的旧实现，现作 ucb_bv1 的 deprecated 别名。
  * thompson_ratio  Beta(1+sr, 1+n−sr) 抽 μ，成本用样本均值，按抽样 ratio 选臂
                    （2026 fuzz MAB 谱系——Bayes-UCB / Thompson 在比率目标上的对应物）。
  * eps_greedy_ratio ε=0.1 的 ratio 贪心（最朴素对照，审稿人常问）。

RNG 纪律（配对比较的基础）：
  * 同一 (env_seed, traj_seed) 下，不同策略共用同一 traj_seed 调用 run()，
    得到可配对轨迹（差值用于后续配对检验）。
  * arm_stats 返回原始和-式统计（n/sr/sc/sr2/sc2/src），口径与 moment_vars 完全一致
    （MLE 方差 = s2/n − (sum/n)²），后续 CI 重算不会静默错。

运行回归（自检，非实验）：
    cd <this dir> && timeout 3000 python3 expD_lib.py
    可用环境变量 EXP_D_MC 覆盖 MC（如 EXP_D_MC=20 快速回归）。
================================================================================
"""
import os
import numpy as np
from collections import deque
from scipy.stats import t as tdist
from scipy.stats import norm as ndist

# ---------------- 默认常量（与 expC 一致，供 make_env / run 缺省与回归使用） ----------------
BASE_SEED = 20260813        # 基础臂参数生成种子（expC 模块级 rng 同种子）
MU_LO, MU_HI = 0.06, 0.30   # 覆盖增益概率范围
C0_MU, C0_SIG = 5.0, 0.35   # 成本 LogNormal 均值与 σ（σ = 成本方差横轴，可调）
K_DEFAULT = 24
MC_DEFAULT = 60
CHANGE_X = 0.5              # 变点触发时的预算消耗比例
NS_K = 6
NS_MU_BASE = np.array([0.50, 0.42, 0.34, 0.26, 0.18, 0.10])
NS_CHANGE_X = 0.5
EPS_GREEDY_EPS = 0.1        # eps_greedy_ratio 的 ε
UCB_BV_LAM_FLOOR = 1e-9     # ucb_bv1 的 λ 下界（分母保正的兜底；保正逻辑见 choose()）
UCB_BV_LAM_RATIO = 1.0      # ucb_bv1 的 λ = RATIO × min_i c_i（默认 1.0 = 真 c_i 下界；
                            # 保守可取 0.8。算法假设成本下界已知，此处给基线最优参数）


# ====================================================================================
#  环境构造（参数化）
# ====================================================================================
def _base_arms(K, mu_range, cost_sigma, cost_mean, cost_mu_corr, base_seed):
    """基础臂参数（跨 MC 复用，加每-MC 噪声）：返回 (mu0, c0)。

    c0 通过高斯 copula 与 mu0 引入秩相关：q = Φ⁻¹(秩/(K+1)) 为正态分数，
    Zc = ρ·q + sqrt(1−ρ²)·Z，c0 = exp(log(cost_mean) + cost_sigma·Zc)。
    ρ=0 时 Zc = Z → 与 expC 的 rng.lognormal(np.log(cost_mean), cost_sigma, K)
    逐位一致（standard_normal + exp 是 lognormal 的实现方式，已验证）。
    """
    rng = np.random.default_rng(base_seed)
    mu0 = np.sort(rng.uniform(mu_range[0], mu_range[1], K))
    rho = float(np.clip(cost_mu_corr, -1.0, 1.0))
    q = ndist.ppf((np.arange(K) + 0.5) / K)          # 排序秩 → 正态分数（mu0 已排序）
    Z = rng.standard_normal(K)
    Zc = rho * q + np.sqrt(1.0 - rho ** 2) * Z
    c0 = np.exp(np.log(cost_mean) + cost_sigma * Zc)
    return mu0, c0


def make_env(K=K_DEFAULT, mu_range=(MU_LO, MU_HI), cost_sigma=C0_SIG,
             cost_mean=C0_MU, gamma=0.0, nonstat="static", seed=1,
             cost_mu_corr=0.0, ns=False, ns_mu_base=None, decay_n=6,
             rekind_n=4, change_n=6, base_seed=BASE_SEED):
    """构造参数化环境 dict。

    参数：
      K           臂数
      mu_range    (lo, hi) 基础 μ 的均匀区间（仅主环境）
      cost_sigma  成本 LogNormal 的 σ（成本方差横轴，E1 必调项）
      cost_mean   成本 LogNormal 的期望均值
      gamma       成功-成本相关强度（成功轮成本 ×(1+γ)）
      nonstat     static / decay / change / rekindle（语义与 expC 完全一致）
      seed        每-MC 微扰种子（= expC 的 mc_seed / ns_seed）
      cost_mu_corr 成本与 μ 的相关系数（∈[-1,1]，0 = 解耦 = expC 原语义）
      ns          True → K=6 等质成本非平稳隔离环境（Panel C/D 口径，ns_env=True）
      ns_mu_base  ns 模式的 μ 基向量（默认 NS_MU_BASE；len 必须等于 K）
      decay_n / rekind_n / change_n  主环境每场景的臂数（默认 6/4/6）
      base_seed   基础臂参数（mu0/c0）的生成种子

    返回 env dict：
      K, mu, c, gamma, nonstat, decay_arms, rekind_arms, swap_of
      ns_env(仅 ns), cost_sigma, cost_mu_corr, change_x, EC, base_c0, base_mu0
    """
    K_ = int(K)
    r = np.random.default_rng(seed)
    if ns:
        # ---------- 非平稳隔离环境（K=6 等质成本；Panel C/D 口径，逐字复刻 ns_env_params） ----------
        base = np.asarray(NS_MU_BASE if ns_mu_base is None else ns_mu_base, dtype=float)
        if len(base) != K_:
            raise ValueError("ns=True 时 ns_mu_base 长度必须等于 K（默认 NS_MU_BASE 长度 %d）"
                             % len(NS_MU_BASE))
        mu = np.clip(base + 0.02 * r.standard_normal(K_), 0.04, 0.65)
        c = np.full(K_, cost_mean) * (0.95 + 0.1 * r.random(K_))   # 近等质成本（~cost_mean）
        decay_arms = sorted(r.choice(K_, 2, replace=False)) if nonstat == "decay" else []
        rekind_arms = sorted(r.choice(K_, 2, replace=False)) if nonstat == "rekindle" else []
        swap_of = {}
        if nonstat == "change":
            # 排序反转：最优臂 ↔ 最差臂 μ 互换（环境总价值守恒，仅换位）
            i_best, i_worst = int(np.argmax(mu)), int(np.argmin(mu))
            swap_of[i_best] = i_worst
            swap_of[i_worst] = i_best
        return {"K": K_, "mu": mu, "c": c, "gamma": gamma, "ns_env": True,
                "nonstat": nonstat, "decay_arms": decay_arms,
                "rekind_arms": rekind_arms, "swap_of": swap_of,
                "cost_sigma": cost_sigma, "cost_mu_corr": cost_mu_corr,
                "change_x": NS_CHANGE_X,
                "base_c0": c.copy(), "base_mu0": mu.copy(),
                "EC": np.full(K_, cost_mean) * np.exp(cost_sigma ** 2 / 2)}
    # ---------- 主环境（K=24 异质成本；Panel A/B/E 口径，逐字复刻 env_params） ----------
    mu0, c0 = _base_arms(K_, mu_range, cost_sigma, cost_mean, cost_mu_corr, base_seed)
    mu = mu0.copy() * (0.85 + 0.3 * r.random(K_))       # 每 MC 微扰
    c = c0.copy() * (0.9 + 0.2 * r.random(K_))
    decay_arms = sorted(r.choice(K_, decay_n, replace=False)) if nonstat == "decay" else []
    rekind_arms = sorted(r.choice(K_, rekind_n, replace=False)) if nonstat == "rekindle" else []
    swap_of = {}
    if nonstat == "change":
        # 突变排序反转：顶 n ↔ 底 n μ 互换（环境总价值守恒，仅排序反转）
        idx = np.argsort(-mu)
        for i in range(change_n):
            swap_of[int(idx[i])] = int(idx[K_ - 1 - i])
            swap_of[int(idx[K_ - 1 - i])] = int(idx[i])
    return {"K": K_, "mu": mu, "c": c, "gamma": gamma,
            "nonstat": nonstat, "decay_arms": decay_arms,
            "rekind_arms": rekind_arms, "swap_of": swap_of,
            "cost_sigma": cost_sigma, "cost_mu_corr": cost_mu_corr,
            "change_x": CHANGE_X,
            "base_c0": c0, "base_mu0": mu0,
            "EC": cost_mean * np.exp(cost_sigma ** 2 / 2)}


def mu_at(env, a, x):
    """主环境 μ(x)。x = 预算消耗比例 ∈[0,1]。语义与 expC 一致。"""
    mu0a = env["mu"][a]
    ns = env["nonstat"]
    if ns == "static":
        return mu0a
    if ns == "decay" and a in env["decay_arms"]:
        return mu0a * np.exp(-3.0 * x)
    if ns == "rekindle" and a in env["rekind_arms"]:
        # 先降后升 U 形：x<0.4 降到 0.3μ0，x>0.4 回升到 1.4μ0
        if x < 0.4:
            return mu0a * max(0.05, 1.0 - 0.7 * x / 0.4)
        return mu0a * min(1.6, 0.3 + 1.1 * (x - 0.4) / 0.6)
    if ns == "change":
        if x >= env.get("change_x", CHANGE_X) and a in env["swap_of"]:   # 顶6↔底6 排序反转
            return env["mu"][env["swap_of"][a]]
        return env["mu"][a]
    return mu0a


def ns_mu_at(env, a, x):
    """ns 环境 μ(x)（K=6 等质成本隔离环境，Panel C/D 口径）。"""
    mu = env["mu"][a]
    ns = env["nonstat"]
    if ns == "static":
        return mu
    if ns == "decay" and a in env["decay_arms"]:
        return mu * np.exp(-3.0 * x)
    if ns == "rekindle" and a in env["rekind_arms"]:
        # 先降后升 U 形：x<0.4 降到 ~0.05，x>0.4 回升到 1.8μ（新最优）
        if x < 0.4:
            return mu * max(0.05, 1.0 - 0.7 * x / 0.4)
        return mu * min(1.8, 0.3 + 1.5 * (x - 0.4) / 0.6)
    if ns == "change" and x >= env.get("change_x", NS_CHANGE_X) and a in env["swap_of"]:
        return env["mu"][env["swap_of"][a]]
    return mu


def sample_arm(env, a, x, r):
    """拉一次臂：返回 (cost, reward, mu)。逐字复用 expC.sample_arm（σ 改用 env 参数）。"""
    mu = (ns_mu_at if env.get("ns_env") else mu_at)(env, a, x)
    rew = r.random() < mu
    c = r.lognormal(np.log(env["c"][a]), env.get("cost_sigma", C0_SIG))
    if rew:
        c *= (1.0 + env["gamma"])       # 成功轮更贵（分母随分子增长）
    return c, rew, mu


# ====================================================================================
#  Fieller / delta 精确 CI（逐字复用 expC）
# ====================================================================================
def fieller_ci(rhat, chat, srr, scc, src, n, alpha=0.10):
    """Fieller 90% CI (lb, ub)。A≤0 / 判别式<0 / n<2 → (-inf, inf)（诚实无界）。"""
    rhat = np.atleast_1d(np.asarray(rhat, float))
    chat = np.atleast_1d(np.asarray(chat, float))
    srr = np.atleast_1d(np.asarray(srr, float))
    scc = np.atleast_1d(np.asarray(scc, float))
    src = np.atleast_1d(np.asarray(src, float))
    n = np.atleast_1d(np.asarray(n, float))
    bad = n < 2
    ns = np.maximum(n, 2.0)
    tq = tdist.ppf(1 - alpha / 2, ns - 1.0)
    A = chat ** 2 - tq ** 2 * scc / ns
    B = -2.0 * (chat * rhat - tq ** 2 * src / ns)
    C = rhat ** 2 - tq ** 2 * srr / ns
    D = B ** 2 - 4.0 * A * C
    D = np.maximum(D, 0.0)
    root = np.sqrt(D)
    twoA = 2.0 * A
    twoA = np.where(np.abs(twoA) < 1e-12, 1e-12, twoA)     # A=0 → 线性退化，按无界处理
    lb = (-B - root) / twoA
    ub = (-B + root) / twoA
    unb = (A <= 0) | (D <= 0) | bad
    lb[unb] = -np.inf
    ub[unb] = np.inf
    return lb, ub


def delta_ci(rhat, chat, srr, scc, src, n, alpha=0.10):
    """delta 方法 90% CI（近似，忽略非对称性与无界性）。n<2 → (-inf, inf)。"""
    rhat = np.atleast_1d(np.asarray(rhat, float))
    chat = np.atleast_1d(np.asarray(chat, float))
    srr = np.atleast_1d(np.asarray(srr, float))
    scc = np.atleast_1d(np.asarray(scc, float))
    src = np.atleast_1d(np.asarray(src, float))
    n = np.atleast_1d(np.asarray(n, float))
    bad = n < 2
    ns = np.maximum(n, 2.0)
    tq = tdist.ppf(1 - alpha / 2, ns - 1.0)
    chat = np.maximum(chat, 1e-9)
    rhat = np.maximum(rhat, 1e-9)
    var = (rhat / chat) ** 2 * (srr / np.maximum(rhat, 1e-12) ** 2
                                + scc / np.maximum(chat, 1e-12) ** 2
                                - 2.0 * src / np.maximum(rhat * chat, 1e-12)) / ns
    var = np.maximum(var, 0.0)
    lb = rhat / chat - tq * np.sqrt(var)
    ub = rhat / chat + tq * np.sqrt(var)
    lb[bad] = -np.inf
    ub[bad] = np.inf
    return lb, ub


def moment_vars(rh, ch, s2, c2, rc, nn):
    """由和-式统计转样本方差/协方差（MLE 口径）。与 arm_stats 字段完全一致。"""
    return (s2 / np.maximum(nn, 1) - rh ** 2,
            c2 / np.maximum(nn, 1) - ch ** 2,
            rc / np.maximum(nn, 1) - rh * ch)


# ====================================================================================
#  MAB 轨迹（参数化策略）
# ====================================================================================
def run(env, strat, B, seed, W=20, elim_thresh=0.55, elim_every=60,
        probe_every=80, drift_every=15, drift_k=0.8, pulls_log=False):
    """在 env 上以策略 strat 跑一条固定预算 B 的轨迹。

    参数：
      env           make_env(...) 返回的环境 dict
      strat         uniform / greedy_ratio / ucb_delta / ucb_fieller /
                    sw_ucb_fieller / adaptive_sw / elim_fieller / sw_ucb_gain /
                    budget_rr(=kube, deprecated) / kube_knapsack / ucb_bv1(=ucb_bv,
                    deprecated) / ucb_bv2 / thompson_ratio / eps_greedy_ratio
      B             固定 token 预算
      seed          轨迹种子（与 expC 同一口径；不同策略共用同一 seed → 可配对比较）
      W             滑窗长度（expC 默认 20）
      elim_thresh   淘汰阈值：Fieller 上界 < elim_thresh × 榜首 时冻结（expC 0.55）
      elim_every    淘汰检查步间隔（expC 60，且预算消耗 ≥15% 才生效）
      probe_every   可逆淘汰解冻探针步间隔（expC 80，仅 adaptive_sw）
      drift_every   漂移检测步间隔（expC 15，仅 adaptive_sw）
      drift_k       漂移阈值系数：|r_recent − r_hist| > drift_k·max(r_hist,1e-6)+0.006
      pulls_log     True 时返回逐拉日志（(step, arm, cost, rew, x) 元组列表）

    返回 dict：
      reward      累计覆盖增益
      cost        累计成本（budget_rr 可略超 B，见 _run_budget_rr 注释；kube_knapsack
                  在剩余预算 < min ĉ 处停，成本通常 ≤ B）
      frozen      期末冻结布尔数组（K,）
      n_pulls     每臂拉取数（= arm_stats.n，和-式口径；adaptive 探针拉取不计数，与 expC 一致）
      arm_stats   {n, sr, sc, sr2, sc2, src} 和-式统计（K,）——与 moment_vars 口径一致，
                  供后续 CI 重算 / 覆盖率评估
      pulls_log   逐拉日志（pulls_log=False 时为 None）
      scheduler_audit  调度器逐决策审计；对 ucb_fieller 记录 Fieller 非有限时的
                       delta fallback 候选率、被选中率、A/D/n 分支及预算四分位。
    """
    if strat in ("kube", "budget_rr"):
        # budget_rr = 朴素预算配额轮转（原误标为 KUBE）；kube 是它的 deprecated 别名。
        return _run_budget_rr(env, B, seed, pulls_log=pulls_log)
    if strat == "kube_knapsack":
        # 真 KUBE：背包式最优预算分配（Tran-Thanh et al., AAAI 2012）。
        return _run_kube_knapsack(env, B, seed, pulls_log=pulls_log)
    Ke = env["K"]                     # 支持 K=24（主环境）/ K=6（非平稳隔离环境）
    r = np.random.default_rng(seed)
    n = np.zeros(Ke); sr = np.zeros(Ke); sc = np.zeros(Ke)
    sr2 = np.zeros(Ke); sc2 = np.zeros(Ke); src = np.zeros(Ke)
    frozen = np.zeros(Ke, dtype=bool)
    wins = [deque() for _ in range(Ke)]       # (cost, reward) 近窗
    hist_r = np.zeros(Ke); hist_c = np.zeros(Ke); hist_n = np.zeros(Ke)
    total_reward = 0.0; total_cost = 0.0
    use_win = strat in ("sw_ucb_fieller", "sw_ucb_gain", "adaptive_sw", "elim_fieller")
    adaptive = strat == "adaptive_sw"
    elim = strat == "elim_fieller"
    step = 0
    log = [] if pulls_log else None
    scheduler_audit = {
        "policy": "adaptive UCB with delta fallback" if strat == "ucb_fieller" else strat,
        "initialization_pulls": int(Ke),
        "policy_decisions": 0,
        "candidate_evaluations": 0,
        "fallback_candidate_evaluations": 0,
        "selected_fallback_decisions": 0,
        "fallback_n_lt_2": 0,
        "fallback_A_only": 0,
        "fallback_D_only": 0,
        "fallback_A_and_D": 0,
        "selected_n_lt_2": 0,
        "selected_A_only": 0,
        "selected_D_only": 0,
        "selected_A_and_D": 0,
        "quartile_decisions": [0, 0, 0, 0],
        "quartile_fallback_candidates": [0, 0, 0, 0],
        "quartile_candidate_evaluations": [0, 0, 0, 0],
        "quartile_selected_fallback": [0, 0, 0, 0],
    }

    def push(a, c, rew):
        wins[a].append((c, rew))
        if len(wins[a]) > W:
            wins[a].popleft()
        hist_r[a] += rew; hist_c[a] += c; hist_n[a] += 1

    def window_stats():
        nw = np.array([len(w) for w in wins])
        sr_w = np.zeros(Ke); sc_w = np.zeros(Ke); sr2_w = np.zeros(Ke)
        sc2_w = np.zeros(Ke); src_w = np.zeros(Ke)
        for a in range(Ke):
            nn = nw[a]
            if nn == 0:
                continue
            w = np.array(wins[a])
            rw, cw = w[:, 1], w[:, 0]
            sr_w[a] = rw.sum(); sc_w[a] = cw.sum()
            sr2_w[a] = (rw ** 2).sum(); sc2_w[a] = (cw ** 2).sum()
            src_w[a] = (rw * cw).sum()
        return nw, sr_w, sc_w, sr2_w, sc2_w, src_w

    def ucb_hi(rhat, chat, srr, scc, src_, nn):
        """Fieller 上界；inf 处回退 delta（冷启动回退近似），最终 nan→大数。"""
        lo, hi = fieller_ci(rhat, chat, srr, scc, src_, nn)
        _, dhi = delta_ci(rhat, chat, srr, scc, src_, nn)
        nf = ~np.isfinite(hi)
        hi[nf] = dhi[nf]
        return np.nan_to_num(hi, nan=1e9, posinf=1e9, neginf=-1e9)

    def choose():
        if strat == "uniform":
            return int(r.integers(0, Ke))
        if strat == "greedy_ratio":
            safe = sr / np.maximum(sc, 1e-9)
            safe[frozen] = -np.inf
            return int(np.argmax(safe))
        if strat == "eps_greedy_ratio":
            # ε=0.1 探索（均匀随机），否则 ratio 贪心。最朴素基线，审稿人常问。
            if r.random() < EPS_GREEDY_EPS:
                return int(r.integers(0, Ke))
            safe = sr / np.maximum(sc, 1e-9)
            safe[frozen] = -np.inf
            return int(np.argmax(safe))
        if strat == "thompson_ratio":
            # Beta(1+sr, 1+n−sr) 后验抽 μ；成本用样本均值；按抽样 ratio 选臂。
            mu_t = r.beta(1.0 + sr, 1.0 + n - sr)
            cost_t = sc / np.maximum(n, 1)
            idx = mu_t / np.maximum(cost_t, 1e-9)
            idx[frozen] = -np.inf
            return int(np.argmax(idx))
        if strat in ("ucb_bv", "ucb_bv1", "ucb_bv2"):
            # UCB-BV1 / UCB-BV2（Ding et al., "Multi-Armed Bandit with Budget
            # Constraint and Variable Costs", AAAI 2013, Eq. 3-4）。
            #   BV1: D_i = r̄_i/ĉ_i + (1+1/λ)·√(ln(t-1)/n_i) / (λ − √(ln(t-1)/n_i))
            #        λ = 成本期望的已知下界（此处取 0.8·min_i c_i；算法假设下界已知，
            #            此处用环境真值下界的保守 0.8 倍，未用任何臂的身份/最优信息）
            #   BV2: D_i = r̄_i/ĉ_i + (1/λ_t)·(1 + 1/(λ_t − √(ln(t-1)/n_i)))·√(ln(t-1)/n_i)
            #        λ_t = min_i ĉ_{i,t}（数据驱动，随样本更新）
            # 分母必须保正（λ > √(ln(t-1)/n_i) 才有效），否则该臂索引 +inf（强制探索：
            # 均匀随机挑一个未冻结的 +inf 臂，而不是 argmax 的任意第一个）。
            # ucb_bv 是旧实现的错误版本，保留为 ucb_bv1 的 deprecated 别名。
            rhat = sr / np.maximum(n, 1)
            chat = sc / np.maximum(n, 1)
            s = np.sqrt(np.log(max(step, 2.0) - 1.0) / np.maximum(n, 1.0))  # √(ln(t-1)/n)
            if strat == "ucb_bv2":
                lam_t = float(np.min(chat))                        # λ_t = min_i ĉ_i
                bad = lam_t <= s
                denom = np.where(lam_t - s > 0, lam_t - s, 1e-9)
                expl = (1.0 / lam_t) * (1.0 + 1.0 / denom) * s
            else:                                                   # ucb_bv, ucb_bv1
                lam = UCB_BV_LAM_RATIO * float(np.min(env["c"]))   # 已知下界 0.8·min c_i
                bad = lam <= s
                denom = np.where(lam - s > 0, lam - s, 1e-9)
                expl = (1.0 + 1.0 / lam) * s / denom
            idx = rhat / np.maximum(chat, 1e-9) + expl
            idx[bad] = np.inf
            idx[frozen] = -np.inf
            force = np.where(np.isposinf(idx))[0]                   # 分母非正 → 强制探索
            if force.size:
                return int(force[r.integers(0, force.size)])
            return int(np.argmax(idx))
        if strat == "ucb_delta":
            rhat = sr / np.maximum(n, 1); chat = sc / np.maximum(n, 1)
            srr, scc, src_ = moment_vars(rhat, chat, sr2, sc2, src, n)
            lo, hi = delta_ci(rhat, chat, srr, scc, src_, n)
            hi = np.nan_to_num(hi, nan=1e9, posinf=1e9, neginf=-1e9)
            hi[frozen] = -np.inf
            return int(np.argmax(hi))
        if strat == "ucb_fieller":
            rhat = sr / np.maximum(n, 1); chat = sc / np.maximum(n, 1)
            srr, scc, src_ = moment_vars(rhat, chat, sr2, sc2, src, n)
            _, fieller_hi = fieller_ci(rhat, chat, srr, scc, src_, n)
            _, delta_hi = delta_ci(rhat, chat, srr, scc, src_, n)
            fallback = ~np.isfinite(fieller_hi)
            ns = np.maximum(n, 2.0)
            tq = tdist.ppf(0.95, ns - 1.0)
            Aq = chat ** 2 - tq ** 2 * scc / ns
            Bq = -2.0 * (chat * rhat - tq ** 2 * src_ / ns)
            Cq = rhat ** 2 - tq ** 2 * srr / ns
            Dq = Bq ** 2 - 4.0 * Aq * Cq
            nlt = n < 2
            a_only = (~nlt) & (Aq <= 0) & (Dq > 0)
            d_only = (~nlt) & (Aq > 0) & (Dq <= 0)
            both = (~nlt) & (Aq <= 0) & (Dq <= 0)
            hi = fieller_hi.copy()
            hi[fallback] = delta_hi[fallback]
            hi = np.nan_to_num(hi, nan=1e9, posinf=1e9, neginf=-1e9)
            hi[frozen] = -np.inf
            selected = int(np.argmax(hi))

            eligible = ~frozen
            quartile = min(3, int(4.0 * min(total_cost / max(B, 1e-12), 1.0)))
            scheduler_audit["policy_decisions"] += 1
            scheduler_audit["candidate_evaluations"] += int(eligible.sum())
            scheduler_audit["fallback_candidate_evaluations"] += int((fallback & eligible).sum())
            scheduler_audit["fallback_n_lt_2"] += int((nlt & eligible).sum())
            scheduler_audit["fallback_A_only"] += int((a_only & eligible).sum())
            scheduler_audit["fallback_D_only"] += int((d_only & eligible).sum())
            scheduler_audit["fallback_A_and_D"] += int((both & eligible).sum())
            scheduler_audit["quartile_decisions"][quartile] += 1
            scheduler_audit["quartile_candidate_evaluations"][quartile] += int(eligible.sum())
            scheduler_audit["quartile_fallback_candidates"][quartile] += int((fallback & eligible).sum())
            if fallback[selected]:
                scheduler_audit["selected_fallback_decisions"] += 1
                scheduler_audit["quartile_selected_fallback"][quartile] += 1
                if nlt[selected]:
                    scheduler_audit["selected_n_lt_2"] += 1
                elif a_only[selected]:
                    scheduler_audit["selected_A_only"] += 1
                elif d_only[selected]:
                    scheduler_audit["selected_D_only"] += 1
                elif both[selected]:
                    scheduler_audit["selected_A_and_D"] += 1
            return selected
        # 滑窗 / 自适应 / 淘汰
        nw, s_w, c_w, s2_w, c2_w, rc_w = window_stats()
        rhat_w = s_w / np.maximum(nw, 1)
        chat_w = c_w / np.maximum(nw, 1)
        if strat == "sw_ucb_gain":
            g = rhat_w
            sd = np.sqrt(np.maximum(s2_w / np.maximum(nw, 1) - rhat_w ** 2, 0))
            ub = g + tdist.ppf(0.95, np.maximum(nw - 1, 1)) * sd / np.maximum(nw, 0.5) ** 0.5
            ub = np.nan_to_num(ub, nan=1e9, posinf=1e9, neginf=-1e9)
            ub[frozen] = -np.inf
            ub[nw < 2] = 1e9
            return int(np.argmax(ub))
        srr, scc, src_ = moment_vars(rhat_w, chat_w, s2_w, c2_w, rc_w, nw)
        hi = ucb_hi(rhat_w, chat_w, srr, scc, src_, nw)
        hi[frozen] = -np.inf
        hi[nw < 2] = 1e9
        return int(np.argmax(hi))

    # 初始化：每臂各探一次
    for a in range(Ke):
        c, rew, _ = sample_arm(env, a, 0.0, r)
        n[a] += 1; sr[a] += rew; sc[a] += c
        sr2[a] += rew ** 2; sc2[a] += c ** 2; src[a] += rew * c
        if use_win:
            push(a, c, rew)
        total_cost += c; total_reward += rew
        if log is not None:
            log.append((step, a, c, rew, 0.0))
        step += 1

    while total_cost < B:
        x = min(total_cost / B, 1.0)
        a = choose()
        c, rew, _ = sample_arm(env, a, x, r)
        n[a] += 1; sr[a] += rew; sc[a] += c
        sr2[a] += rew ** 2; sc2[a] += c ** 2; src[a] += rew * c
        if use_win:
            push(a, c, rew)
        total_cost += c; total_reward += rew
        if log is not None:
            log.append((step, a, c, rew, x))
        step += 1
        if adaptive:
            # 可逆淘汰的"解冻探针"：每 probe_every 步对每个冻结臂真拉 1 次（计入成本/覆盖），
            # 命中即解冻——冻结臂不会因不再被拉取而永远错过复苏（elim 的致命弱点）。
            if step % probe_every == 0:
                for aa in np.where(frozen)[0]:
                    c, rew, _ = sample_arm(env, aa, x, r)
                    wins[aa].clear()
                    wins[aa].append((c, rew))
                    hist_r[aa] += rew; hist_c[aa] += c; hist_n[aa] += 1
                    if rew == 1:
                        frozen[aa] = False
                    total_cost += c; total_reward += rew
                    if log is not None:
                        log.append((step, aa, c, rew, x))
            # 漂移检测：近窗(末W样本) ratio vs 全历史 ratio 显著背离 → 重置近窗 + 解冻（可逆淘汰）
            if step % drift_every == 0:
                for aa in range(Ke):
                    w = list(wins[aa])
                    if len(w) < 20 or hist_n[aa] < 40:      # 漂移窗 20 / 最小历史 40（expC 原值）
                        continue
                    w = np.array(w)
                    r_recent = w[:, 1].sum() / max(w[:, 0].sum(), 1e-9)
                    r_hist = hist_r[aa] / max(hist_c[aa], 1e-9)
                    if abs(r_recent - r_hist) > drift_k * max(r_hist, 1e-6) + 0.006:
                        wins[aa].clear()
                        if frozen[aa]:
                            frozen[aa] = False
        if elim or adaptive:
            # 置信淘汰（elim=不可逆；adaptive=可逆——被漂移检测块解冻）：
            # 每 elim_every 步且预算消耗≥15% 后，Fieller 上界 < elim_thresh×榜首 的臂冻结。
            if step % elim_every == 0 and total_cost > 0.15 * B:
                nw, s_w, c_w, s2_w, c2_w, rc_w = window_stats()
                rhat = s_w / np.maximum(nw, 1); chat = c_w / np.maximum(nw, 1)
                srr, scc, src_ = moment_vars(rhat, chat, s2_w, c2_w, rc_w, nw)
                lo, hi = fieller_ci(rhat, chat, srr, scc, src_, nw)
                # 非有限区间臂不参与淘汰（安全：不冻信息不足的臂）
                alive = ~frozen & (nw >= max(10, W // 2)) & np.isfinite(hi) & np.isfinite(lo)
                if alive.any():
                    bh = np.max(hi[alive])
                    elim_mask = alive & (hi < elim_thresh * bh)
                    frozen[elim_mask] = True

    return {"reward": total_reward, "cost": total_cost, "frozen": frozen,
            "n_pulls": n.copy(),
            "arm_stats": {"n": n.copy(), "sr": sr.copy(), "sc": sc.copy(),
                          "sr2": sr2.copy(), "sc2": sc2.copy(), "src": src.copy()},
            "pulls_log": log, "scheduler_audit": scheduler_audit}


def _run_budget_rr(env, B, seed, pulls_log=False):
    """budget_rr（原误标为 KUBE）：预算配额均分 + 轮转的朴素基线。

    NOT Tran-Thanh 的 KUBE——只是"每臂等额预算 B/K，轮转拉取直到各自配额耗尽"。
    逐字复刻 expC main() 内联 kube 循环（含 while/break 结构，保证 RNG 消耗顺序
    一致；注意其 seed 口径为 MC 索引，非 10000+i）。额外追踪 arm_stats
    （不改变 RNG 消耗）。`kube` 是它的 deprecated 别名（兼容旧脚本）。"""
    Ke = env["K"]
    r = np.random.default_rng(seed)
    budget_a = B / Ke
    spent = np.zeros(Ke)
    n = np.zeros(Ke); sr = np.zeros(Ke); sc = np.zeros(Ke)
    sr2 = np.zeros(Ke); sc2 = np.zeros(Ke); src = np.zeros(Ke)
    total_reward = 0.0
    kstep = 0
    log = [] if pulls_log else None
    while spent.min() < budget_a:
        for a in range(Ke):
            if spent[a] >= budget_a:
                continue
            c, rw, _ = sample_arm(env, a, 0.0, r)
            spent[a] += c
            n[a] += 1; sr[a] += rw; sc[a] += c
            sr2[a] += rw ** 2; sc2[a] += c ** 2; src[a] += rw * c
            total_reward += rw
            if log is not None:
                log.append((kstep, a, c, rw, 0.0))
            kstep += 1
            if spent.sum() >= B:
                break
    return {"reward": total_reward, "cost": spent.sum(),
            "frozen": np.zeros(Ke, dtype=bool),
            "n_pulls": n.copy(),
            "arm_stats": {"n": n.copy(), "sr": sr.copy(), "sc": sc.copy(),
                          "sr2": sr2.copy(), "sc2": sc2.copy(), "src": src.copy()},
            "pulls_log": log}


def _run_kube_knapsack(env, B, seed, pulls_log=False):
    """真 KUBE（Tran-Thanh et al., "Knapsack Based Optimal Policies for Budget-Limited
    Multi-Armed Bandits", AAAI 2012, arXiv:1204.1909, Algorithm 1 + Eq. 6-7）。

    算法（每步 t > K）：
      1) 解近似无界背包
           max Σ_i m_{i,t}·(μ̂_{i,t} + √(2 ln t / n_{i,t}))
           s.t.  Σ_i m_{i,t}·ĉ_i ≤ B_t（剩余预算）
         用密度序贪心：density_i = (μ̂_i + √(2 ln t / n_i)) / ĉ_i。
         （无界背包单约束下贪心最优：全部预算分配给当前最高密度臂 → m* 为单臂点质量。）
      2) 以概率 P(i) = m*_i / Σ_k m*_k 拉臂 i。
      3) B_{t+1} = B_t − c_{i(t)}；当 B_t < min_i ĉ_i 停止。

    必要偏离（注释，审稿人可见）：原算法假设成本 {c_i} 已知（放入背包约束与密度
    分母），此处成本未知，用样本均值 ĉ_i 估计——这是 fuzzing 场景的必要偏离
    （调度器无法预知单次 fuzz 的 token 成本）。成本若已知则本实现退化为原文。
    """
    Ke = env["K"]
    r = np.random.default_rng(seed)
    n = np.zeros(Ke); sr = np.zeros(Ke); sc = np.zeros(Ke)
    sr2 = np.zeros(Ke); sc2 = np.zeros(Ke); src = np.zeros(Ke)
    total_reward = 0.0; total_cost = 0.0
    step = 0
    log = [] if pulls_log else None
    # 初始化：每臂各探一次（冷启动，与 run() 的 init 循环口径一致）
    for a in range(Ke):
        c, rew, _ = sample_arm(env, a, 0.0, r)
        n[a] += 1; sr[a] += rew; sc[a] += c
        sr2[a] += rew ** 2; sc2[a] += c ** 2; src[a] += rew * c
        total_cost += c; total_reward += rew
        if log is not None:
            log.append((step, a, c, rew, 0.0))
        step += 1
    while True:
        B_t = B - total_cost
        chat = sc / np.maximum(n, 1)
        if B_t < chat.min():                       # 剩余预算买不起任何臂 → 停
            break
        muhat = sr / np.maximum(n, 1)
        ucb = muhat + np.sqrt(2.0 * np.log(max(step, 2)) / np.maximum(n, 1))
        density = ucb / np.maximum(chat, 1e-9)
        m = np.zeros(Ke)
        m[int(np.argmax(density))] = B_t / max(chat[int(np.argmax(density))], 1e-9)
        p = m / m.sum()
        a = int(r.choice(Ke, p=p))
        x = min(total_cost / B, 1.0)
        c, rew, _ = sample_arm(env, a, x, r)
        n[a] += 1; sr[a] += rew; sc[a] += c
        sr2[a] += rew ** 2; sc2[a] += c ** 2; src[a] += rew * c
        total_cost += c; total_reward += rew
        if log is not None:
            log.append((step, a, c, rew, x))
        step += 1
    return {"reward": total_reward, "cost": total_cost,
            "frozen": np.zeros(Ke, dtype=bool),
            "n_pulls": n.copy(),
            "arm_stats": {"n": n.copy(), "sr": sr.copy(), "sc": sc.copy(),
                          "sr2": sr2.copy(), "sc2": sc2.copy(), "src": src.copy()},
            "pulls_log": log}


# ====================================================================================
#  回归校验（复跑 expC Panel A / C，逐格对齐 expC_out.txt）
# ====================================================================================
def _regression(mc=MC_DEFAULT, verbose=True):
    import expC_cost_sched as expc      # 仅用于回归对比基准（模块级只生成 c0/mu0，无副作用）

    # ---- 0) 环境构造逐位校验：make_env(默认) == expC 全局 + env_params ----
    envA = make_env(K=K_DEFAULT, mu_range=(MU_LO, MU_HI), cost_sigma=C0_SIG,
                    cost_mean=C0_MU, gamma=0.0, nonstat="static", seed=1,
                    cost_mu_corr=0.0, base_seed=BASE_SEED)
    checks = []
    checks.append(("make_env.base_c0 == expC.c0", np.allclose(envA["base_c0"], expc.c0)))
    checks.append(("make_env.base_mu0 == expC.mu0", np.allclose(envA["base_mu0"], expc.mu0)))
    checks.append(("make_env(static,seed=1).mu == expC.env_params(static,0,1).mu",
                   np.allclose(envA["mu"], expc.env_params("static", 0.0, 1)["mu"])))
    checks.append(("make_env(static,seed=1).c  == expC.env_params(static,0,1).c",
                   np.allclose(envA["c"], expc.env_params("static", 0.0, 1)["c"])))
    for ns in ("decay", "change", "rekindle"):
        e1 = make_env(K=K_DEFAULT, cost_sigma=C0_SIG, cost_mean=C0_MU, gamma=0.0,
                      nonstat=ns, seed=1, cost_mu_corr=0.0)
        e2 = expc.env_params(ns, 0.0, 1)
        checks.append(("make_env(%s) == expC.env_params(%s,0,1) (mu/c/arms)"
                       % (ns, ns),
                       np.allclose(e1["mu"], e2["mu"]) and np.allclose(e1["c"], e2["c"])
                       and e1["decay_arms"] == e2["decay_arms"]
                       and e1["rekind_arms"] == e2["rekind_arms"]
                       and e1["swap_of"] == e2["swap_of"]))
    for ns in ("static", "decay", "change", "rekindle"):
        e1 = make_env(K=NS_K, cost_sigma=C0_SIG, cost_mean=5.0, gamma=0.0,
                      nonstat=ns, seed=1005, ns=True)
        e2 = expc.ns_env_params(ns, 0.0, 1005)
        checks.append(("make_env(ns=True,%s,seed=1005) == expC.ns_env_params" % ns,
                       np.allclose(e1["mu"], e2["mu"]) and np.allclose(e1["c"], e2["c"])
                       and e1["decay_arms"] == e2["decay_arms"]
                       and e1["rekind_arms"] == e2["rekind_arms"]
                       and e1["swap_of"] == e2["swap_of"]))
    if verbose:
        print("=" * 100)
        print("expD_lib 回归校验 · 0) 环境构造与 expC 逐位一致（%s）" % mc)
        allok = True
        for name, ok in checks:
            allok &= ok
            print("    %-58s %s" % (name, "OK" if ok else "DIFF!!"))
        assert allok, "环境构造与 expC 不一致，回归无效"

    # ---- 1) Panel A：静态 K=24 同质成本 预算 sweep（3 档 × 8 策略 + budget_rr） ----
    REF_A = {"uniform": [44, 133, 264], "greedy_ratio": [79, 252, 515],
             "ucb_delta": [79, 282, 600], "ucb_fieller": [78, 280, 597],
             "sw_ucb_fieller": [70, 228, 475], "elim_fieller": [70, 237, 511],
             "budget_rr": [54, 159, 316], "sw_ucb_gain": [55, 175, 351]}
    B_MULTS = [10, 30, 60]
    if verbose:
        print("-" * 100)
        print("expD_lib 回归校验 · Panel A（static K=24，预算 10/30/60×，MC=%d，"
              "非 budget_rr seed=10000+i，budget_rr seed=i）" % mc)
        print("    %-15s | %-38s | %s" % ("策略", "expC原值", "新引擎(逐格)"))
    A_ok = True
    A_new = {s: [] for s in REF_A}
    for mult in B_MULTS:
        B = mult * np.sum(envA["base_c0"]) * np.exp(envA["cost_sigma"] ** 2 / 2)
        for s in REF_A:
            tot = []
            for i in range(mc):
                seed = i if s == "budget_rr" else 10000 + i
                res = run(envA, s, B, seed=seed)
                tot.append(res["reward"])
            A_new[s].append(float(np.mean(tot)))
    if verbose:
        for s in REF_A:
            cells = []
            for j, ref in enumerate(REF_A[s]):
                v = A_new[s][j]
                rv = int(round(v))
                ok = (rv == ref)
                A_ok &= ok
                cells.append("%s%.0f%s" % ("(" if ok else "DIFF(", v, ")" if ok else ")"))
            print("    %-15s | %-38s | %s" % (s, "  ".join("%d" % x for x in REF_A[s]),
                                              "  ".join(cells)))
        print("    Panel A 逐格对齐: %s" % ("ALL OK" if A_ok else "存在 DIFF（见上）"))

        # ---- 1b) 改名回归：kube(旧名别名) == budget_rr 逐位一致；新基线可运行 ----
        print("-" * 100)
        alias_ok = True
        for mult in (30,):
            B = mult * np.sum(envA["base_c0"]) * np.exp(envA["cost_sigma"] ** 2 / 2)
            for i in range(3):
                ra = run(envA, "kube", B, seed=i)
                rb = run(envA, "budget_rr", B, seed=i)
                bit = (ra["reward"] == rb["reward"] and ra["cost"] == rb["cost"]
                       and np.array_equal(ra["n_pulls"], rb["n_pulls"]))
                alias_ok &= bool(bit)
        print("    kube(旧别名) == budget_rr 逐位一致（3 轨迹×预算30×）: %s"
              % ("OK" if alias_ok else "DIFF"))
        for ns_ in ("kube_knapsack", "ucb_bv1", "ucb_bv2"):
            B = 30 * np.sum(envA["base_c0"]) * np.exp(envA["cost_sigma"] ** 2 / 2)
            v = run(envA, ns_, B, seed=10000 + 7)
            print("    新基线可运行 %-14s reward=%.0f cost=%.0f"
                  % (ns_, v["reward"], v["cost"]))

    # ---- 2) Panel C：K=6 非平稳 4 场景 × 4 策略，预算 100× ----
    REF_C = {"static": {"ucb_fieller": 255, "sw_ucb_fieller": 251,
                        "adaptive_sw": 248, "elim_fieller": 251},
             "decay": {"ucb_fieller": 220, "sw_ucb_fieller": 221,
                       "adaptive_sw": 220, "elim_fieller": 223},
             "change": {"ucb_fieller": 205, "sw_ucb_fieller": 224,
                        "adaptive_sw": 222, "elim_fieller": 222},
             "rekindle": {"ucb_fieller": 244, "sw_ucb_fieller": 241,
                          "adaptive_sw": 244, "elim_fieller": 235}}
    REF_D = {"elim_fieller": 0.308, "adaptive_sw": 0.092}
    NS = ["static", "decay", "change", "rekindle"]
    C_new = {ns: {} for ns in NS}
    D_new = {}
    for ns in NS:
        for s in REF_C[ns]:
            tot = []
            for i in range(mc):
                env = make_env(K=NS_K, cost_sigma=C0_SIG, cost_mean=5.0, gamma=0.0,
                               nonstat=ns, seed=1000 + i, ns=True)
                Bns = 100 * np.sum(env["c"])
                res = run(env, s, Bns, seed=20000 + i)
                tot.append(res["reward"])
            C_new[ns][s] = float(np.mean(tot))
        if ns == "rekindle":
            for s in ("elim_fieller", "adaptive_sw"):
                frz = []
                for i in range(mc):
                    env = make_env(K=NS_K, cost_sigma=C0_SIG, cost_mean=5.0, gamma=0.0,
                                   nonstat=ns, seed=1000 + i, ns=True)
                    Bns = 100 * np.sum(env["c"])
                    res = run(env, s, Bns, seed=20000 + i)
                    rekind = set(env["rekind_arms"])
                    frz.append(len(rekind & set(np.where(res["frozen"])[0]))
                               / max(len(rekind), 1))
                D_new[s] = float(np.mean(frz))
    if verbose:
        print("-" * 100)
        print("expD_lib 回归校验 · Panel C（K=6 非平稳 4 场景 × 4 策略，预算 100×，MC=%d，"
              "env seed 1000+i，traj seed 20000+i）" % mc)
        C_ok = True
        for ns in NS:
            cells = []
            for s in REF_C[ns]:
                v = C_new[ns][s]
                ref = REF_C[ns][s]
                ok = (int(round(v)) == ref)
                C_ok &= ok
                cells.append("%s=%s%.0f%s" % (s, "" if ok else "DIFF(", v,
                                              "" if ok else ")"))
            print("    %-9s | 原值 %-12s | %s" % (ns,
                  " ".join("%s=%d" % (s, REF_C[ns][s]) for s in REF_C[ns]),
                  "  ".join(cells)))
        print("    Panel C 逐格对齐: %s" % ("ALL OK" if C_ok else "存在 DIFF（见上）"))
        print("-" * 100)
        print("expD_lib 回归校验 · Panel D（rekindle 误杀率，可复燃臂期末仍冻结比例）")
        for s in REF_D:
            v = D_new.get(s)
            ok = v is not None and abs(v - REF_D[s]) < 0.005
            print("    %-14s | 原值 %.3f | 新引擎 %.3f  %s" % (s, REF_D[s],
                                                              v if v is not None else float("nan"),
                                                              "OK" if ok else "DIFF"))
    return A_new, A_ok, C_new, C_ok, D_new, checks


if __name__ == "__main__":
    import time
    t_start = time.time()
    mc = int(os.environ.get("EXP_D_MC", str(MC_DEFAULT)))
    A_new, A_ok, C_new, C_ok, D_new, checks = _regression(mc=mc)
    print("=" * 100)
    print("回归完成 · MC=%d · 耗时 %.1fs · Panel A %s · Panel C %s"
          % (mc, time.time() - t_start, "OK" if A_ok else "DIFF",
             "OK" if C_ok else "DIFF"))
