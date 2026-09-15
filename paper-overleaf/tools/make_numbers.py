#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate numbers.tex (prose macros) and the table bodies from data/*.json.

Sources: r2_recomputed.json (frozen holdout, unified convention, degenerate classes),
r4_policy.json (policy ablation, budgets, environments, alpha sweep, fixed-n calibration,
denominator diagnostics, selection), x4_landscape.json, x2_landscape.json, beta_control.json,
realcost_calibration.json, theory.json, anchor_diagnostics.json.  The body text must not
contain typed result values.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")


def J(name):
    return json.load(open(os.path.join(DATA, name)))


r2 = J("r2_recomputed.json")
r4 = J("r4_policy.json")
r5 = J("r5_designs.json")
x4 = J("x4_landscape.json")
x2 = J("x2_landscape.json")
bc = J("beta_control.json")
rc = J("realcost_calibration.json")
th = J("theory.json")
dg = J("anchor_diagnostics.json")
man = J("r2_holdout.json")["manifest"]
r6 = J("r6_reportability.json")  # existence-condition checks and real success rates (ICASSP27-6 reportability.py)

macros = []


def m(name, value):
    macros.append("\\newcommand{\\%s}{%s}" % (name, value))


def f3(x):
    return "%.3f" % x


def f2(x):
    return "%.2f" % x


def ci3(t):
    return "%.3f [%.3f, %.3f]" % (t[0], t[1], t[2])


def ci3s(t):
    return "%.3f~[%.3f,\\,%.3f]" % (t[0], t[1], t[2])


def pct1(x):
    return "%.1f\\%%" % (100 * x)


def sgn3(x):
    return ("+" if x >= 0 else "$-$") + "%.3f" % abs(x)


def pfmt(p):
    if p < 1e-4:
        mant, e = ("%.0e" % p).split("e")
        return "%s\\times 10^{%d}" % (mant, int(e))
    if p < 1e-3:
        return "%.4f" % p
    return "%.3f" % p


METH = (("Fieller", "F"), ("raw delta", "D"), ("delta-JV", "JV"), ("Jeffreys+cost", "JC"), ("Bayes", "By"), ("conf. seq.", "Cs"))
LAMT = {"0.02": "TwoC", "0.05": "FiveC", "0.10": "Ten", "0.20": "Twenty", "0.50": "Half", "1.00": "One"}

# ------------------------------------------------------------------ benchmark constants
m("Karms", str(man["K"])); m("MCtraj", str(man["MC"])); m("SeedLo", str(man["seed_start"])); m("SeedHi", str(man["seed_end"]))
m("BudgetMult", str(man["budget_multiplier"])); m("SigmaCost", "%.2f" % man["sigma"]); m("CVCost", "%.2f" % x2["cells"][0]["cv"])
m("AlphaLevel", "%.2f" % man["alpha"]); m("NominalPct", "%d" % round(100 * (1 - man["alpha"]))); m("ClusterB", str(man["cluster_bootstrap_replicates"]))
m("MuLo", "%.2f" % (math.floor(dg["mu_min"] * 100) / 100)); m("MuHi", "%.2f" % (math.ceil(dg["mu_max"] * 100) / 100))
m("CostLo", "%.1f" % (math.floor(dg["c_min"] * 10) / 10)); m("CostHi", "%.1f" % (math.ceil(dg["c_max"] * 10) / 10))
_az = "%.0e" % dg["all_arms_two_zeros_prob"]; m("AllZeroProb", "$%s\\times10^{%d}$" % (_az.split("e")[0], int(_az.split("e")[1]))); m("RhoLo", "%.4f" % dg["rho_min"]); m("RhoHi", "%.4f" % dg["rho_max"]); m("BudgetVal", "%.0f" % r2["budget"])

# ------------------------------------------------------------------ holdout, unified convention
for tag, blk in (("A", r2["adaptive"]), ("U", r2["uniform"])):
    for meth, sh in METH:
        b = blk["methods"][meth]
        m(tag + "KindPt" + sh, f3(b["kinds"]["point"])); m(tag + "KindLn" + sh, f3(b["kinds"]["line"]))
        m(tag + "KindInt" + sh, f3(b["kinds"]["interval"]))
        m(tag + "Rep" + sh, f3(b["nondeg"]["report"][0])); m(tag + "RepCI" + sh, ci3(b["nondeg"]["report"]))
        m(tag + "Cond" + sh, f3(b["nondeg"]["conditional"][0])); m(tag + "CondCI" + sh, ci3(b["nondeg"]["conditional"]))
        m(tag + "BRep" + sh, f3(b["bounded"]["report"][0])); m(tag + "BCond" + sh, f3(b["bounded"]["conditional"][0]))
        m(tag + "BCondCI" + sh, ci3(b["bounded"]["conditional"]))
        m(tag + "Joint" + sh, f3(b["nondeg"]["joint"][0])); m(tag + "JointCI" + sh, ci3(b["nondeg"]["joint"]))
        m(tag + "ComF" + sh, f3(b["common_F"][0])); m(tag + "ComFCI" + sh, ci3(b["common_F"]))
        m(tag + "SetCov" + sh, f3(b["set_coverage"][0])); m(tag + "SetCovCI" + sh, ci3(b["set_coverage"]))
        m(tag + "Wid" + sh, f3(b["nondeg"]["width_median"][0]))
        m(tag + "ZeroCov" + sh, f3(b["coverage_on_zero_outcome"][0]))
        for lam, lt in LAMT.items():
            m(tag + "SNd" + lt + sh, f3(b["score"]["nondeg"][lam][0]))
            m(tag + "SBd" + lt + sh, f3(b["score"]["bounded"][lam][0]))
A = r2["adaptive"]; U = r2["uniform"]
_hw = 0.0
for _blk in (A, U):
    for _meth in _blk["methods"].values():
        for _t in (_meth["nondeg"]["report"], _meth["nondeg"]["conditional"], _meth["bounded"]["report"], _meth["bounded"]["conditional"],
                   _meth["nondeg"]["joint"], _meth["common_F"], _meth["set_coverage"]):
            _hw = max(_hw, _t[2] - _t[0], _t[0] - _t[1])
m("MaxHalfWidth", "%.3f" % (_hw + 0.0005))
m("DiffCovJCabs", "%.3f" % abs(A["confirmatory_tests"][4]["difference"][0]) if A["confirmatory_tests"][4]["B"] == "Jeffreys+cost" else "")
m("AJVWidRatio", "%.1f" % (A["methods"]["delta-JV"]["nondeg"]["width_median"][0] / A["methods"]["Fieller"]["nondeg"]["width_median"][0]))
m("AJCWidRatio", "%.1f" % (A["methods"]["Jeffreys+cost"]["nondeg"]["width_median"][0] / A["methods"]["Fieller"]["nondeg"]["width_median"][0]))
z = A["zero_outcome"]
m("ZeroFrac", f3(z["fraction"][0])); m("ZeroFracCI", ci3(z["fraction"])); m("ZeroFracPct", pct1(z["fraction"][0]))
m("ZeroPtF", f3(z["fieller_point_given_zero"][0])); m("ZeroLnF", f3(z["fieller_line_given_zero"][0]))
m("ZeroPtD", f3(z["delta_point_given_zero"][0])); m("NondegGivenNonzero", f3(z["fieller_nondeg_given_nonzero"][0]))
m("MedianN", "%d" % z["median_n"]); m("MeanN", "%.1f" % z["mean_n"]); m("NMax", "{:,}".format(int(z["n_max"]))); m("FracNTwo", f3(z["frac_n_eq_2"]))
m("TwoRayShare", f3(z["fieller_two_ray_share"])); m("JcLineNTwo", f3(z["jc_line_share_n2"])); m("JcLineShare", f3(z["jc_line_share"]))
m("UZeroFrac", f3(U["zero_outcome"]["fraction"][0])); m("UMedianN", "%d" % U["zero_outcome"]["median_n"])
m("ArmCondMin", f3(A["per_arm_conditional_min"])); m("ArmCondMax", f3(A["per_arm_conditional_max"]))
m("ArmTopCond", f3(A["per_arm_top3_conditional"][0])); m("ArmTopRep", f3(A["per_arm_top3_report"][0]))
m("ArmSecondCond", f3(A["per_arm_top3_conditional"][1]))
m("MleCondF", f3(r2["adaptive_mle"]["methods"]["Fieller"]["nondeg"]["conditional"][0]))
m("MleRepF", f3(r2["adaptive_mle"]["methods"]["Fieller"]["nondeg"]["report"][0]))
m("MleZeroLn", f3(r2["adaptive_mle"]["zero_outcome"]["fieller_line_given_zero"][0]))
for t in A["confirmatory_tests"]:
    sh = {"raw delta": "D", "delta-JV": "JV", "Jeffreys+cost": "JC"}[t["B"]]
    kind = "Cov" if t["metric"].startswith("common") else "Sc"
    d = t["difference"]
    m("Diff" + kind + sh, "%s [%s, %s]" % (sgn3(d[0]), sgn3(d[1]), sgn3(d[2])))
    m("Holm" + kind + sh, pfmt(t["holm_p"]))
m("NTests", str(len(A["confirmatory_tests"])))
sf = r2["scheduler_fallback"]
m("CandFallback", f3(sf["candidate_fallback_rate"])); m("SelFallback", f3(sf["selected_fallback_rate"]))
m("PolicyDecisions", "{:,}".format(sf["policy_decisions"])); m("SelFallbackCount", "{:,}".format(sf["selected_fallback_decisions"]))
m("SelFallbackQoneCount", "{:,}".format(sf["quartile_selected_fallback"][0]))


def affine(b, acc):
    s1, s2 = b["score"][acc]["0.10"][0], b["score"][acc]["1.00"][0]
    slope = (s2 - s1) / 0.9
    return s1 - slope * 0.1, slope


aF, sF = affine(A["methods"]["Fieller"], "bounded"); aD, sD = affine(A["methods"]["raw delta"], "bounded")
lam_star = (aD - aF) / (sF - sD) if abs(sF - sD) > 1e-12 else float("nan")
m("LamStarBd", "%.2f" % lam_star); m("LamStarBdRho", "%.0f" % (lam_star / r2["rho_max"]))
aJ, sJ = affine(A["methods"]["Jeffreys+cost"], "bounded")
lam_jc = (aJ - aF) / (sF - sJ) if abs(sF - sJ) > 1e-12 else float("nan")
m("LamStarJC", "%.2f" % lam_jc)
m("RhoMaxTwo", "%.3f" % r2["rho_max"])

# ------------------------------------------------------------------ R4
RUNS = r4["runs"]
POL = {"Del": "ucb_fieller_delta", "Jv": "ucb_fieller_jv", "Inf": "ucb_fieller_inf", "Ujv": "ucb_jv", "Uni": "uniform",
       "BvOne": "ucb_bv1", "BvTwo": "ucb_bv2", "Kube": "kube_knapsack", "Ts": "thompson_ratio", "Eps": "eps_greedy_ratio",
       "Gr": "greedy_ratio", "Ud": "ucb_delta"}


def run(tag):
    pol = POL[tag]
    blk = "H" if tag in ("BvOne", "BvTwo", "Kube", "Ts", "Eps", "Gr", "Ud") else "A"
    return RUNS["%s|%s|env1|30x" % (blk, pol)]


for tag in POL:
    r = run(tag)
    m("R" + tag + "Rep", f3(r["methods"]["Fieller"]["report"][0])); m("R" + tag + "Cond", f3(r["methods"]["Fieller"]["conditional"][0]))
    m("R" + tag + "Joint", f3(r["methods"]["Fieller"]["joint"][0])); m("R" + tag + "DJoint", f3(r["methods"]["raw delta"]["joint"][0]))
    m("R" + tag + "JcRep", f3(r["methods"]["Jeffreys+cost"]["report"][0])); m("R" + tag + "JvCond", f3(r["methods"]["delta-JV"]["conditional"][0]))
    m("R" + tag + "Rew", "%.0f" % r["reward_mean"][0]); m("R" + tag + "RewCI", "%.0f [%.0f, %.0f]" % tuple(r["reward_mean"]))
    m("R" + tag + "Nz", f3(r["frac_n2_zero"][0])); m("R" + tag + "Zero", f3(r["zero_outcome_frac"][0]))
m("RPredNz", f3(run("Del")["pred_frac_n2_zero"]))
gaps = [abs(v["frac_n2_zero"][0] - v["pred_frac_n2_zero"]) for k, v in RUNS.items() if "ucb_fieller_delta" in k and not k.startswith("J|")]
m("RNzGapMax", f3(max(gaps))); m("RNzRuns", str(len(gaps)))
rd, rj, ri = run("Del")["reward_mean"][0], run("Jv")["reward_mean"][0], run("Inf")["reward_mean"][0]
m("RRewGainJvPct", "%.1f\\%%" % (100 * (rj - rd) / rd)); m("RRewLossInfPct", "%.1f\\%%" % (100 * (rd - ri) / rd))
m("RRepRatioJv", "%.1f" % (run("Jv")["methods"]["Fieller"]["report"][0] / run("Del")["methods"]["Fieller"]["report"][0]))
m("RRewRatioDelUni", "%.1f" % (rd / run("Uni")["reward_mean"][0]))
for mult, nm in ((5, "Five"), (10, "Ten"), (100, "Hundred")):
    m("RBudRep" + nm, f3(RUNS["B|ucb_fieller_delta|env1|%dx" % mult]["methods"]["Fieller"]["report"][0]))
    m("RUBudRep" + nm, f3(RUNS["B|uniform|env1|%dx" % mult]["methods"]["Fieller"]["report"][0]))
cs = [RUNS["C|ucb_fieller_delta|env%d|30x" % e] for e in (2, 3, 4)]
m("RCNzPairs", ", ".join("%.3f against %.3f" % (c["frac_n2_zero"][0], c["pred_frac_n2_zero"]) for c in cs))
E = r4["env_replication"]
ea, eu = E["ucb_fieller_delta"], E["uniform"]
m("ENenv", str(ea["n_env"])); m("EMC", str(ea["MC_per_env"]))
m("ERepMean", f3(ea["report"]["mean"])); m("ERepLo", f3(ea["report"]["min"])); m("ERepHi", f3(ea["report"]["max"]))
m("ECondMean", f3(ea["conditional"]["mean"])); m("ECondLo", f3(ea["conditional"]["min"])); m("ECondHi", f3(ea["conditional"]["max"]))
m("EJointMean", f3(ea["joint"]["mean"]))
m("ECFGap", "%s [%s, %s]" % (sgn3(ea["common_F_gap"]["mean"]), sgn3(ea["common_F_gap"]["env_boot"][1]), sgn3(ea["common_F_gap"]["env_boot"][2])))
m("EBestCond", f3(ea["truebest_cond"]["mean"])); m("EBestLo", f3(ea["truebest_cond"]["min"])); m("EBestHi", f3(ea["truebest_cond"]["max"]))
m("EBestCondCI", ci3(ea["truebest_cond"]["env_boot"])); m("EBestRep", f3(ea["truebest_report"]["mean"]))
m("EPredGapMax", f3(ea["max_abs_pred_gap"])); m("EZeroMean", f3(ea["zero_outcome_frac"]["mean"]))
m("EUCondMean", f3(eu["conditional"]["mean"])); m("EUCondLo", f3(eu["conditional"]["min"])); m("EUCondHi", f3(eu["conditional"]["max"]))
m("EURepMean", f3(eu["report"]["mean"])); m("EUBestCond", f3(eu["truebest_cond"]["mean"]))
m("ESelCond", f3(ea["sel_cond"]["mean"])); m("EUSelCond", f3(eu["sel_cond"]["mean"]))
AL = r4["alpha_sweep"]
for a, nm in (("0.20", "Twenty"), ("0.10", "Ten"), ("0.05", "Five"), ("0.01", "One")):
    m("AlCond" + nm, f3(AL["ucb_fieller"][a]["Fieller"]["conditional"][0])); m("AlRep" + nm, f3(AL["ucb_fieller"][a]["Fieller"]["report"][0]))
    m("AlDCom" + nm, f3(AL["ucb_fieller"][a]["raw delta"]["common_F"][0])); m("UAlCond" + nm, f3(AL["uniform"][a]["Fieller"]["conditional"][0]))
    m("AlJcRep" + nm, f3(AL["ucb_fieller"][a]["Jeffreys+cost"]["report"][0]))
FX = {rec["n"]: rec for rec in r4["fixed_n_calibration"]}
for n, nm in ((2, "Two"), (3, "Three"), (5, "Five"), (10, "Ten"), (30, "Thirty")):
    m("FixEng" + nm, f3(FX[n]["engine_F"])); m("FixTxt" + nm, f3(FX[n]["textbook_F"])); m("FixRep" + nm, f3(FX[n]["report_F"])); m("FixDel" + nm, f3(FX[n]["engine_D"]))
    m("FixTxtRep" + nm, f3(FX[n]["report_textbook_F"]))
m("FixMaxGap", f3(max(abs(rec["engine_F"] - rec["textbook_F"]) for rec in r4["fixed_n_calibration"])))
DEN = {(rec["cost_dist"], round(rec["cv"], 2), rec["pareto_alpha"]): rec["n"] for rec in r4["denominator_coverage"]}
m("DenLogNLo", f3(DEN[("lognormal", 0.36, None)]["30"]["unbiased"])); m("DenLogNHi", f3(DEN[("lognormal", 2.91, None)]["30"]["unbiased"]))
m("DenGammaHi", f3(DEN[("gamma", 2.91, None)]["30"]["unbiased"])); m("DenTwoHi", f3(DEN[("twopoint", 2.91, None)]["30"]["unbiased"]))
m("DenParInf", f3(DEN[("pareto", 0.36, 1.5)]["30"]["unbiased"])); m("DenTwoHiHundred", f3(DEN[("twopoint", 2.91, None)]["100"]["unbiased"]))
m("DenGammaLo", f3(DEN[("gamma", 0.36, None)]["30"]["unbiased"]))
SA = run("Del")["selection"]; SU = run("Uni")["selection"]
m("SSelIdentPct", pct1(SA["identification"][0])); m("USSelIdentPct", pct1(SU["identification"][0])); m("SBestMedN", "%d" % SA["truebest_n_median"])
SELROWS = (("F", "Fieller|naive"), ("FB", "Fieller|bonferroni"), ("FS", "Fieller|split"), ("JC", "Jeffreys+cost|naive"), ("JCS", "Jeffreys+cost|split"),
           ("BF", "truebest|Fieller|naive"), ("BFB", "truebest|Fieller|bonferroni"), ("BD", "truebest|raw delta|naive"), ("BJC", "truebest|Jeffreys+cost|naive"))
for pre, S_ in (("S", SA), ("US", SU)):
    for tag, key in SELROWS:
        b = S_[key]
        m(pre + tag + "Rep", f3(b["report"][0])); m(pre + tag + "Cond", f3(b["conditional"][0])); m(pre + tag + "CondCI", ci3(b["conditional"]))
        m(pre + tag + "Joint", f3(b["joint"][0])); m(pre + tag + "Wid", f3(b["width_median"]))
m("SBestBonfWidRatio", "%.1f" % (SA["truebest|Fieller|bonferroni"]["width_median"] / SA["truebest|Fieller|naive"]["width_median"]))
m("SSelSplitWidRatio", "%.1f" % (SA["Fieller|split"]["width_median"] / SA["Fieller|naive"]["width_median"]))
m("RMC", str(r4["MC"]))

# ------------------------------------------------------------------ existence conditions (r6)
s1 = r6["s1"]
m("IdCells", str(s1["cells"])); m("IdReps", "{:,}".format(s1["nrep"])); m("IdDraws", "%.0f" % (s1["cells"] * s1["nrep"] / 1e6))
m("IdAgree", "%.6f" % s1["ident_min"]); m("OutMax", "%.4f" % s1["num_max"]); m("FacMax", "%.4f" % s1["factor_max"])
m("FacGamma", "%.3f" % r6["s1_gamma"]["factor_max"]); m("GaussMax", "%.2f" % s1["gauss_max"])
m("CVbiasLoTwo", "%.3f" % s1["cv_bias"]["0.35"][0]); m("CVbiasHiTwo", "%.3f" % s1["cv_bias"]["1.5"][0])
s2 = r6["s2"]
m("RuleCons", str(s2["conservative"])); m("RuleTotal", str(s2["total"])); m("RuleRatioMed", "%.2f" % s2["ratio_med"])
m("RuleRatioLo", "%.2f" % s2["ratio_lo"]); m("RuleRatioHi", "%.2f" % s2["ratio_hi"])
s4 = r6["s4"]; src = r6["s4_src"]
m("RealTotal", str(s4["n_candidates_total"])); m("RealK", str(s4["n_candidates_used"]))
m("RealZeroPct", "%.1f\\%%" % (100 * (src["e7"]["zero"] * src["e7"]["n"] + src["e8"]["zero"] * src["e8"]["n"]) / (src["e7"]["n"] + src["e8"]["n"])))
m("RealNa", str(src["e7"]["n"])); m("RealNb", str(src["e8"]["n"]))
lo_ = s4["sigma0.35"]
m("RealMeas", f3(lo_["meas"])); m("RealPred", f3(lo_["pred"])); m("RealErr", "%.3f" % abs(lo_["err"]))
m("RealFcov", f3(lo_["f_cov"])); m("RealFcovCI", "%.3f [%.3f, %.3f]" % (lo_["f_cov"], lo_["f_lo"], lo_["f_hi"])); m("RealDcov", f3(lo_["d_cov"]))
m("RealNmed", "%d" % lo_["n_med"]); m("RealErrHi", "%.3f" % abs(s4["sigma1.50"]["err"])); m("RealMC", str(lo_["n_units"] // lo_["K"]))
m("RealMedRateB", "%.3f" % src["e8"]["med"])
m("RealNeeded", "%d" % math.ceil(math.log(0.05) / math.log(1 - src["e8"]["med"])))

# ------------------------------------------------------------------ landscapes and controls
m("XfCells", str(x4["n_cells"])); m("XfCondMin", f3(x4["conditional_min"])); m("XfCondMax", f3(x4["conditional_max"])); m("XfCondMean", f3(x4["conditional_mean"]))
m("XfFail", str(x4["cluster_upper_below_nominal"])); m("XfPointBelow", str(x4["point_below_nominal"])); m("XfRepMin", f3(x4["report_min"])); m("XfRepMax", f3(x4["report_max"]))
m("XfDiffMean", sgn3(x4["diff_mean"])); m("XfDOnFMean", f3(x4["delta_on_F_mean"]))
m("XfDiffMax", sgn3(max(c["conditional"][0] - c["delta_on_F"][0] for c in x4["cells"])))
wc = max(x4["cells"], key=lambda c: c["conditional"][0] - c["delta_on_F"][0])
m("XfDiffMaxCell", "%s under %s sampling" % (wc["cost_dist"], wc["mechanism"]))
m("XfDiffMin", sgn3(min(c["conditional"][0] - c["delta_on_F"][0] for c in x4["cells"])))
wr = x4["worst_report_cell"]; m("XfWorstRepCell", "%s with %s" % (wr["cost_dist"], wr["mechanism"].replace("_", "-")))
m("XfHolmFail", str(len(x4["holm_fail_cells"]))); m("XfExpFalse", "%.1f" % x4["expected_false_fails_at_2p5pct"])
hc = sorted(x4["holm_fail_cells"], key=lambda c: c["holm_p_below_nominal"])
m("XfHolmCells", " and ".join("%s under %s sampling (%s, Holm $p=%.3f$)" % (c["cost_dist"], c["mechanism"], ci3(c["conditional"]), c["holm_p_below_nominal"]) for c in hc))
other = [c for c in x4["fail_cells"] if c not in x4["holm_fail_cells"]]
m("XfNonHolmCells", " and ".join("%s under %s sampling (%s, Holm $p=%.2f$)" % (c["cost_dist"], c["mechanism"], f3(c["conditional"][0]), c["holm_p_below_nominal"]) for c in other) if other else "none")
m("XfNDist", str(len(set(c["cost_dist"] for c in x4["cells"])))); m("XfNMech", str(len(set(c["mechanism"] for c in x4["cells"])))); m("XfMC", str(x4.get("MC", 200)))
xa = x2["adaptive"]; xu = x2["uniform"]
m("XtCells", str(x2["n_cells"])); m("XtAdCells", str(len([c for c in x2["cells"] if c["policy"] == "ucb_fieller"])))
m("XtAdCondMin", f3(xa["conditional_min"])); m("XtAdCondMax", f3(xa["conditional_max"])); m("XtAdFail", str(xa["cluster_upper_below_nominal"]))
m("XtAdRepMin", f3(xa["report_min"])); m("XtAdRepMax", f3(xa["report_max"]))
m("XtRepSigLo", f3(xa["report_mean_by_sigma"]["0.35"])); m("XtRepSigMid", f3(xa["report_mean_by_sigma"]["0.9"])); m("XtRepSigHi", f3(xa["report_mean_by_sigma"]["1.5"]))
m("XtUnFail", str(xu["cluster_upper_below_nominal"])); m("XtUnCondMin", f3(xu["conditional_min"]))
wrc = xa["worst_report_cell"]; m("XtWorstRepCell", "$K=%d$, $\\sigma=%.2f$, $\\gamma=%.1f$, budget %d$\\times$" % (wrc["K"], wrc["sigma"], wrc["gamma"], wrc["budget"]))
m("BetaPairs", str(bc["n_pairs"])); m("BetaDAllBern", f3(bc["bern"]["delta_all"])); m("BetaDAllBeta", f3(bc["beta"]["delta_all"]))
m("BetaRepBern", f3(bc["bern"]["report"])); m("BetaRepBeta", f3(bc["beta"]["report"])); m("BetaFCondBern", f3(bc["bern"]["f_cond"])); m("BetaFCondBeta", f3(bc["beta"]["f_cond"]))
m("BetaNu", "%d" % bc["beta_nu"])
row = [r for r in rc["rows"] if r["budget"] == 30][0]
m("RcRep", f3(row["report"][0])); m("RcCondF", f3(row["conditional"][0])); m("RcDOnF", f3(row["delta_on_F"][0])); m("RcDAll", f3(row["delta_all"][0]))
for k, v in th["cv_threshold"].items():
    m("CVthr" + {"2": "Two", "3": "Three", "5": "Five", "10": "Ten", "20": "Twenty", "50": "Fifty"}[k], "%.2f" % v)

# R5: regularised indices, two-stage design, always-existing constructions
for key, tag in (("kl_ucb", "Kl"), ("bernstein_ucb", "Bern"), ("laplace_delta", "Lap")):
    r = r5[key]
    m("R" + tag + "Zero", f3(r["zero_outcome_frac"][0])); m("R" + tag + "Rep", f3(r["methods"]["Fieller"]["report"][0]))
    m("R" + tag + "Rew", "%.0f" % r["reward"][0])
ts = r5["two_stage"]["0.4"]
m("TsRep", f3(ts["methods"]["Fieller"]["report"][0])); m("TsCond", f3(ts["methods"]["Fieller"]["conditional"][0]))
m("TsSet", f3(ts["methods"]["Fieller"]["set_coverage"][0])); m("TsRew", "%.0f" % ts["reward"][0])
m("TsMedN", "%.0f" % ts["median_n"]); m("TsZero", f3(ts["zero_outcome_frac"][0]))
m("TsRepLo", f3(min(r5["two_stage"][k]["methods"]["Fieller"]["report"][0] for k in r5["two_stage"])))
m("TsRepHi", f3(max(r5["two_stage"][k]["methods"]["Fieller"]["report"][0] for k in r5["two_stage"])))
for nm, tag in (("Bayes", "By"), ("conf. seq.", "Cs")):
    b = A["methods"][nm]
    m("Hold" + tag + "Rep", f3(b["nondeg"]["report"][0])); m("Hold" + tag + "Cond", f3(b["nondeg"]["conditional"][0]))
    m("Hold" + tag + "Set", f3(b["set_coverage"][0])); m("Hold" + tag + "Line", f3(b["kinds"]["line"]))
    m("Hold" + tag + "Wid", f3(b["nondeg"]["width_median"][0]))

with open(os.path.join(ROOT, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by tools/make_numbers.py; do not edit\n" + "\n".join(macros) + "\n")
print("numbers.tex: %d macros" % len(macros))

# ------------------------------------------------------------------ tables
lines = ["% generated by tools/make_numbers.py", "\\begin{tabular}{@{}lcccccccccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{degenerate share} & \\multicolumn{2}{c}{non-degenerate acc.} & \\multicolumn{2}{c}{bounded acc.} & & & & \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "method & point & line & report & cond. & report & cond. & joint & set & common-F & width \\\\", "\\midrule"]
for tag, blk, title in (("A", A, "adaptive UCB with delta fallback"),):
    for meth, sh in METH:
        b = blk["methods"][meth]
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            meth, f3(b["kinds"]["point"]), f3(b["kinds"]["line"]), f3(b["nondeg"]["report"][0]), f3(b["nondeg"]["conditional"][0]),
            f3(b["bounded"]["report"][0]), f3(b["bounded"]["conditional"][0]), f3(b["nondeg"]["joint"][0]), f3(b["set_coverage"][0]),
            f3(b["common_F"][0]), f3(b["nondeg"]["width_median"][0])))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "table_main.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

POL_ROWS = (("Uni", "uniform"), ("BvOne", "UCB-BV1"), ("BvTwo", "UCB-BV2"), ("Kube", "KUBE"), ("Ts", "Thompson on ratio"),
            ("Eps", "$\\epsilon$-greedy"), ("Gr", "greedy"), ("Del", "UCB-F, delta fallback"),
            ("Jv", "UCB-F, delta-JV fallback"), ("Inf", "UCB-F, forced exploration"))
R5_ROWS = (("kl_ucb", "KL-UCB"), ("bernstein_ucb", "Bernstein UCB"), ("laplace_delta", "Laplace delta"))
lines = ["% generated by tools/make_numbers.py", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "policy & zero-out. & report & cond. & reward \\\\", "\\midrule"]
for tag, disp in POL_ROWS:
    r = run(tag)
    lines.append("%s & %s & %s & %s & %.0f \\\\" % (
        disp, f3(r["zero_outcome_frac"][0]), f3(r["methods"]["Fieller"]["report"][0]), f3(r["methods"]["Fieller"]["conditional"][0]),
        r["reward_mean"][0]))
for key, disp in R5_ROWS:
    r = r5[key]
    lines.append("%s & %s & %s & %s & %.0f \\\\" % (
        disp, f3(r["zero_outcome_frac"][0]), f3(r["methods"]["Fieller"]["report"][0]),
        f3(r["methods"]["Fieller"]["conditional"][0]), r["reward"][0]))
lines += ["\\bottomrule", "\\end{tabular}"]
NPOL = len(POL_ROWS) + len(R5_ROWS)
open(os.path.join(ROOT, "numbers.tex"), "a", encoding="utf-8").write(
    "\\newcommand{\\NPol}{%s}\n\\newcommand{\\NPolNum}{%d}\n" % (
        {12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}.get(NPOL, str(NPOL)), NPOL))
open(os.path.join(ROOT, "table_policy.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def row_sel(disp, b):
    return "%s & %s & %s & %s & %s \\\\" % (disp, f3(b["report"][0]), ci3s(b["conditional"]), f3(b["joint"][0]), f3(b["width_median"]))


lines = ["% generated by tools/make_numbers.py", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "method & report & conditional & joint & width \\\\", "\\midrule"]
lines.append("\\multicolumn{5}{@{}l@{}}{\\emph{adaptive, arm with the largest empirical ratio (true best in %s)}} \\\\" % pct1(SA["identification"][0]))
for disp, key in (("Fieller", "Fieller|naive"), ("Fieller, $\\alpha/K$", "Fieller|bonferroni"), ("Fieller, split", "Fieller|split")):
    lines.append(row_sel(disp, SA[key]))
lines.append("\\midrule")
lines.append("\\multicolumn{5}{@{}l@{}}{\\emph{adaptive, fixed true-best arm (median %d pulls)}} \\\\" % SA["truebest_n_median"])
for disp, key in (("Fieller", "truebest|Fieller|naive"), ("Fieller, $\\alpha/K$", "truebest|Fieller|bonferroni"), ("raw delta", "truebest|raw delta|naive")):
    lines.append(row_sel(disp, SA[key]))
lines += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(ROOT, "table_select.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("tables written")
