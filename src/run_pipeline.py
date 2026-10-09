"""End-to-end pipeline: audit -> KM/PH -> models -> profit targeting -> sensitivity -> SHAP -> fairness.

Run:  python -I src/run_pipeline.py          (about 5-10 minutes)
Outputs go to results/, figures/ and models/.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sksurv.metrics import brier_score, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from data import DATA_PATH, ROOT, FeatureBuilder, load_clean, split
from economics import ipcw_outcomes, profit_at, profit_curve, random_curve, scores
from models import CONTRACT_COLS, CoxStrat, clf_input, fit_all

warnings.filterwarnings("ignore")
RES, FIG, MOD = ROOT / "results", ROOT / "figures", ROOT / "models"
for d in (RES, FIG, MOD):
    d.mkdir(exist_ok=True)

SEED = 42
BASE = dict(H=12, m=0.40, c=20.0, q=0.25, gamma=0.99)  # paper Table I base values
SENS = dict(H=[6, 12, 18, 24], m=[0.30, 0.40, 0.50], c=[10, 20, 30, 40], q=[0.10, 0.25, 0.40])

# categorical slots 1-5 of the validated reference palette (light surface)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         violet="#4a3aa7", gray="#8a8985", ink="#0b0b0b", ink2="#52514e", surface="#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "text.color": C["ink"], "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e9e8e4", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2.0,
    "figure.dpi": 130,
})


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def log(msg):
    print(msg, flush=True)


# ====================================================================== 1. audit, KM, PH
def step_audit_km_ph(df, audit, Xtr, tr):
    churned = df[df.E == 1]; kept = df[df.E == 0]
    audit.update({
        "churned": int(df.E.sum()), "churn_rate_pct": round(100 * df.E.mean(), 1),
        "tenure_min": int(df["T"].min()), "tenure_max": int(df["T"].max()),
        "tenure_median": float(df["T"].median()),
        "median_tenure_churned": float(churned["T"].median()),
        "median_tenure_retained": float(kept["T"].median()),
        "mean_monthly_churned": round(churned.MonthlyCharges.mean(), 2),
        "mean_monthly_retained": round(kept.MonthlyCharges.mean(), 2),
        "senior_pct": round(100 * df.SeniorCitizen.mean(), 1),
        "churn_by_contract_pct": (100 * df.groupby("Contract").E.mean()).round(1).to_dict(),
        "customers_by_contract": df.Contract.value_counts().to_dict(),
        "customers_by_internet": df.InternetService.value_counts().to_dict(),
    })
    # --- Kaplan-Meier by contract / internet / payment
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
    styles = ["-", "--", "-.", ":"]
    lr = {}
    for ax, var in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
        for i, (lvl, g) in enumerate(df.groupby(var)):
            kmf = KaplanMeierFitter().fit(g["T"], g["E"], label=f"{lvl} (n={len(g):,})")
            kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12, color=cols[i],
                                       linestyle=styles[i])
        lr[var] = float(multivariate_logrank_test(df["T"], df[var], df["E"]).p_value)
        ax.set_title(f"Retention by {var}"); ax.set_xlabel("Tenure (months)")
        ax.axvline(12, color=C["gray"], lw=1, ls=":"); ax.set_ylim(0, 1.02)
        ax.text(12.8, 0.52, "12-month horizon", fontsize=7.5, color=C["ink2"], rotation=90)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("P(still subscribed)")
    savefig(fig, "01_kaplan_meier.png")
    audit["logrank_p"] = lr
    # --- hazard by tenure band
    bands = pd.cut(df["T"], [0, 6, 12, 18, 24, 36, 48, 60, 72])
    haz = df.groupby(bands, observed=True).apply(
        lambda g: pd.Series({"churn_events": int(g.E.sum()), "customers": len(g),
                             "event_rate_pct": 100 * g.E.mean()}))
    haz.index = haz.index.astype(str)
    haz.to_csv(RES / "churn_by_tenure_band.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(haz.index, haz.event_rate_pct, color=C["blue"], width=0.7)
    for x, v in zip(haz.index, haz.event_rate_pct):
        ax.text(x, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8, color=C["ink2"])
    ax.set_title("Share of customers who churned, by tenure at observation")
    ax.set_xlabel("Tenure band (months)"); ax.set_ylabel("% churned"); ax.grid(axis="x", visible=False)
    savefig(fig, "02_churn_by_tenure_band.png")
    # --- proportional-hazards check (before / after stratifying on contract)
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    d0 = Xtr.copy(); d0["T"], d0["E"] = tr["T"].values, tr["E"].values
    c0 = CoxPHFitter(penalizer=0.01).fit(d0, "T", "E")
    p0 = proportional_hazard_test(c0, d0, time_transform="rank").summary["p"]
    cs = CoxStrat(0.05, 0.5).fit(Xtr, tr["T"].values, tr["E"].values)
    d1 = cs._frame(Xtr); d1["T"], d1["E"] = tr["T"].values, tr["E"].values
    p1 = proportional_hazard_test(cs.m, d1, time_transform="rank").summary["p"]
    ph = pd.DataFrame({"p_unstratified": p0, "p_stratified_on_contract": p1})
    ph.to_csv(RES / "proportional_hazards_test.csv")
    audit["ph_violations_p<0.05"] = {"unstratified": int((p0 < 0.05).sum()),
                                     "stratified": int((p1 < 0.05).sum()), "n_covariates": len(p0)}
    audit["ph_contract_p"] = {k: float(p0[k]) for k in CONTRACT_COLS}
    return audit


# ====================================================================== 2. model evaluation
TAU = 60  # test follow-up truncated at 60 months: the censoring distribution is ~0 near 72


def surv_arrays(tr, te):
    """Structured arrays for sksurv metrics; test follow-up administratively truncated at TAU."""
    ytr = Surv.from_arrays(tr.E.astype(bool), tr["T"].astype(float))
    t = np.minimum(te["T"].values, TAU).astype(float)
    e = (te.E.values == 1) & (te["T"].values <= TAU)
    return ytr, Surv.from_arrays(e, t)


def eval_survival_like(S12_risk_pi6, pi12, te, tr, S_full=None):
    ytr, yte = surv_arrays(tr, te)
    out = {"c_index": concordance_index(te["T"], 1 - pi12, te.E),
           "auc_6m": float(cumulative_dynamic_auc(ytr, yte, S12_risk_pi6, [6])[0][0]),
           "auc_12m": float(cumulative_dynamic_auc(ytr, yte, pi12, [12])[0][0])}
    if S_full is not None:
        out["ibs_1_59"] = float(integrated_brier_score(ytr, yte, S_full[:, :59], np.arange(1, 60)))
        out["brier_12m"] = float(brier_score(ytr, yte, S_full[:, [11]], [12])[1][0])
    return out


def calibration(pi, te, nb=10):
    q = pd.qcut(pd.Series(pi).rank(method="first"), nb, labels=False)
    rows = []
    for b in range(nb):
        m = (q == b).values
        kmf = KaplanMeierFitter().fit(te["T"][m], te.E[m])
        obs = 1 - float(kmf.survival_function_at_times(12).iloc[0])
        rows.append({"bin": b + 1, "n": int(m.sum()), "pred_pi12": float(pi[m].mean()), "obs_pi12": obs})
    d = pd.DataFrame(rows)
    d["abs_err"] = (d.pred_pi12 - d.obs_pi12).abs()
    return d


def step_models(tr, te, Xtr, Xte, surv, clf, hp):
    rows = {}; Spred = {}
    for k, m in surv.items():
        S = m.surv(Xte); Spred[k] = S
        rows[m.name] = eval_survival_like(1 - S[:, 5], 1 - S[:, 11], te, tr, S)
    probs = {k: m.proba(clf_input(k, Xte, te["T"].values)) for k, m in clf.items()}
    for k, m in clf.items():
        p = probs[k]
        r = eval_survival_like(p, p, te, tr)
        r.update({"roc_auc_flag": roc_auc_score(te.E, p),
                  "precision@0.5": precision_score(te.E, p > 0.5),
                  "recall@0.5": recall_score(te.E, p > 0.5),
                  "f1@0.5": f1_score(te.E, p > 0.5)})
        rows[m.name] = r
    ytr_, yte_ = surv_arrays(tr, te)
    rows["Tenure alone (leakage check)"] = {  # rank by short tenure only; no other information
        "c_index": np.nan,
        "auc_6m": float(cumulative_dynamic_auc(ytr_, yte_, -te["T"].values.astype(float), [6])[0][0]),
        "auc_12m": float(cumulative_dynamic_auc(ytr_, yte_, -te["T"].values.astype(float), [12])[0][0])}
    tbl = pd.DataFrame(rows).T
    tbl.to_csv(RES / "model_comparison.csv")
    return tbl, Spred, probs


# ====================================================================== 3. profit analysis
FAIR = ["Logistic prob.", "XGBoost prob."]                       # same information as the survival models
TEN = ["Logistic + tenure prob.", "XGBoost + tenure prob."]     # tenure-at-observation used as a feature
PROB_RULES = FAIR + TEN
RULES = ["Random"] + PROB_RULES + ["Survival risk (pi_12)",
         "Expected profit (paper, G)", "Expected profit (refined, G2)"]


def rule_scores(Ssel, probs, MC, p):
    sc = scores(Ssel, MC, p["H"], p["m"], p["q"], p["c"], p["gamma"])
    return sc, {"Logistic prob.": probs["logit"], "XGBoost prob.": probs["xgb"],
                "Logistic + tenure prob.": probs["logit_t"], "XGBoost + tenure prob.": probs["xgb_t"],
                "Survival risk (pi_12)": sc["pi"], "Expected profit (paper, G)": sc["G"],
                "Expected profit (refined, G2)": sc["G2"]}


def profit_table(te, Ssel, probs, p, per=1000.0):
    """Max-over-curve profit for ranking rules, threshold profit for G rules (per `per` customers)."""
    MC = te.MonthlyCharges.values
    w, lost = ipcw_outcomes(te["T"].values, te.E.values, MC, p["H"], p["m"], p["gamma"])
    sc, rs = rule_scores(Ssel, probs, MC, p)
    n = len(te); f = per / n; out = {}
    out["Random"] = {"max_profit": max(0.0, (random_curve(w, lost, p["q"], p["c"]) * f).max()),
                     "at_share": 0.0, "profit_if_all": random_curve(w, lost, p["q"], p["c"])[-1] * f}
    curves = {"Random": random_curve(w, lost, p["q"], p["c"])}
    for k, s in rs.items():
        cur = profit_curve(s, w, lost, p["q"], p["c"]); curves[k] = cur
        out[k] = {"max_profit": cur.max() * f, "at_share": cur.argmax() / n, "profit_if_all": cur[-1] * f}
    for k, key in [("Expected profit (paper, G)", "G"), ("Expected profit (refined, G2)", "G2")]:
        pr, cnt = profit_at(sc[key], 0.0, w, lost, p["q"], p["c"])
        out[k].update({"profit_at_G>0": pr * f, "share_contacted_G>0": cnt / n})
    oracle = np.maximum(0, p["q"] * w * lost - p["c"]).sum() * f
    return out, curves, (w, lost, sc, rs), oracle


def bootstrap_profit(te, Ssel, probs, p, B=300, seed=7):
    """Bootstrap customers (fixed IPCW weights) -> CI for each rule's max / operating-point profit."""
    rng = np.random.default_rng(seed); n = len(te)
    MC = te.MonthlyCharges.values
    w, lost = ipcw_outcomes(te["T"].values, te.E.values, MC, p["H"], p["m"], p["gamma"])
    sc, rs = rule_scores(Ssel, probs, MC, p)
    res = {k: [] for k in rs}; res["G2 @ threshold"] = []; res["G @ threshold"] = []
    res["Random"] = []
    for k_ in ("diff_G2_thr_minus_best_fair_max", "diff_G2_thr_minus_best_tenure_max",
               "diff_survrisk_max_minus_best_fair_max", "diff_G2_thr_minus_survrisk_max"):
        res[k_] = []
    for _ in range(B):
        i = rng.integers(0, n, n); wi, li = w[i], lost[i]
        mx = {}
        for k, s in rs.items():
            mx[k] = profit_curve(s[i], wi, li, p["q"], p["c"]).max() * 1000 / n
            res[k].append(mx[k])
        res["G2 @ threshold"].append(profit_at(sc["G2"][i], 0, wi, li, p["q"], p["c"])[0] * 1000 / n)
        res["G @ threshold"].append(profit_at(sc["G"][i], 0, wi, li, p["q"], p["c"])[0] * 1000 / n)
        res["Random"].append(random_curve(wi, li, p["q"], p["c"]).max() * 1000 / n)
        g2 = res["G2 @ threshold"][-1]
        res["diff_G2_thr_minus_best_fair_max"].append(g2 - max(mx[r] for r in FAIR))
        res["diff_G2_thr_minus_best_tenure_max"].append(g2 - max(mx[r] for r in TEN))
        res["diff_survrisk_max_minus_best_fair_max"].append(
            mx["Survival risk (pi_12)"] - max(mx[r] for r in FAIR))
        res["diff_G2_thr_minus_survrisk_max"].append(g2 - mx["Survival risk (pi_12)"])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}
    return ci


def step_profit(te, Ssel, probs):
    out, curves, aux, oracle = profit_table(te, Ssel, probs, BASE)
    ci = bootstrap_profit(te, Ssel, probs, BASE)
    n = len(te)
    rows = []
    for r in RULES:
        o = out[r]
        rows.append({"rule": r, "max_profit_per_1000": o["max_profit"], "best_share_contacted": o["at_share"],
                     "profit_if_contact_all_per_1000": o["profit_if_all"],
                     "max_ci_lo": ci.get(r, [np.nan] * 2)[0], "max_ci_hi": ci.get(r, [np.nan] * 2)[1],
                     "profit_at_G>0_per_1000": o.get("profit_at_G>0", np.nan),
                     "share_contacted_G>0": o.get("share_contacted_G>0", np.nan)})
    tbl = pd.DataFrame(rows); tbl.to_csv(RES / "profit_comparison.csv", index=False)
    json.dump({"oracle_per_1000": oracle, "bootstrap_ci": ci, "n_test": n, "params": BASE},
              open(RES / "profit_bootstrap.json", "w"), indent=2)
    # --- profit curve figure
    sty = {"Random": (C["gray"], "--"), "XGBoost + tenure prob.": (C["orange"], "-."),
           "XGBoost prob.": (C["blue"], "-"), "Survival risk (pi_12)": (C["aqua"], ":"),
           "Expected profit (paper, G)": (C["magenta"], "--"),
           "Expected profit (refined, G2)": (C["violet"], "-")}
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); x = np.linspace(0, 100, n + 1); f = 1000 / n
    for r in sty:
        col, ls = sty[r]
        ax.plot(x, curves[r] * f, color=col, ls=ls, label=r, lw=2.4 if "refined" in r else 1.8)
    for r, key in [("Expected profit (paper, G)", "G"), ("Expected profit (refined, G2)", "G2")]:
        o = out[r]
        ax.scatter([100 * o["share_contacted_G>0"]], [o["profit_at_G>0"]], s=60, color=sty[r][0],
                   edgecolor=C["surface"], linewidth=2, zorder=5)
    ax.axhline(0, color=C["ink2"], lw=0.8)
    ax.set_xlabel("Share of customers contacted (%)"); ax.set_ylabel("Net profit per 1,000 customers ($)")
    ax.set_title("Net campaign profit by targeting rule (H=12, margin 40%, offer $20, success 25%)")
    ax.legend(fontsize=8, loc="upper right"); ax.set_xlim(0, 60); ax.set_ylim(-5500, 3200)
    ax.annotate("dots = where the profit rule stops (score > 0); x-axis cut at 60%", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", fontsize=8, color=C["ink2"])
    savefig(fig, "04_profit_curves.png")
    return tbl, out, ci, oracle, aux


# ====================================================================== 4. sensitivity
def step_sensitivity(te, Ssel, probs):
    rows = []
    for param, vals in SENS.items():
        for v in vals:
            p = dict(BASE); p[param] = v
            out, _, _, _ = profit_table(te, Ssel, probs, p)
            best_fair = max(out[r]["max_profit"] for r in FAIR)
            best_ten = max(out[r]["max_profit"] for r in TEN)
            rows.append({"param": param, "value": v,
                         "refined_G2>0": out["Expected profit (refined, G2)"]["profit_at_G>0"],
                         "share_G2>0": out["Expected profit (refined, G2)"]["share_contacted_G>0"],
                         "paper_G>0": out["Expected profit (paper, G)"]["profit_at_G>0"],
                         "share_G>0": out["Expected profit (paper, G)"]["share_contacted_G>0"],
                         "best_fair_prob_ranking_max": best_fair,
                         "best_tenure_prob_ranking_max": best_ten,
                         "survival_risk_max": out["Survival risk (pi_12)"]["max_profit"],
                         "contact_all": out["XGBoost prob."]["profit_if_all"]})
    oat = pd.DataFrame(rows); oat.to_csv(RES / "sensitivity_oat.csv", index=False)
    # 2-D grid q x c on the refined rule
    qs, cs = [0.10, 0.175, 0.25, 0.325, 0.40], [10, 20, 30, 40]
    A = np.zeros((len(cs), len(qs))); D = np.zeros_like(A); SH = np.zeros_like(A)
    for i, c in enumerate(cs):
        for j, q in enumerate(qs):
            p = dict(BASE); p["q"], p["c"] = q, c
            out, _, _, _ = profit_table(te, Ssel, probs, p)
            A[i, j] = out["Expected profit (refined, G2)"]["profit_at_G>0"]
            SH[i, j] = out["Expected profit (refined, G2)"]["share_contacted_G>0"]
            D[i, j] = A[i, j] - max(out[r]["max_profit"] for r in FAIR)
    grid = pd.DataFrame([{"q": q, "c": c, "refined_profit": A[i, j], "share_contacted": SH[i, j],
                          "advantage_vs_best_prob_max": D[i, j]}
                         for i, c in enumerate(cs) for j, q in enumerate(qs)])
    grid.to_csv(RES / "sensitivity_q_by_c.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])
    div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    for ax, M, cmap, ttl, norm in [
        (axes[0], A, seq, "Refined rule: profit per 1,000 customers ($)", None),
        (axes[1], D, div, "Refined rule minus best like-for-like classifier's peak ($)",
         TwoSlopeNorm(0, min(D.min(), -1), max(D.max(), 1)))]:
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{q * 100:g}%" for q in qs])
        ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"${c}" for c in cs])
        ax.set_xlabel("Offer success rate q"); ax.set_title(ttl, fontsize=10)
        ax.set_ylabel("Offer cost c" if ax is axes[0] else "")
        ax.grid(False)
        for i in range(len(cs)):
            for j in range(len(qs)):
                v = M[i, j]
                dark = (norm is None and v > M.max() * 0.55) or (norm is not None and abs(v) > 0.7 * np.abs(M).max())
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=9,
                        color="white" if dark else C["ink"])
        fig.colorbar(im, ax=ax, fraction=0.04)
    savefig(fig, "05_sensitivity_q_by_c.png")
    return oat, grid


# ====================================================================== 5. SHAP
def step_shap(sel_key, surv, clf, Xtr, Xte, te, sc):
    sel = surv[sel_key]
    cols = list(Xte.columns)
    rng = np.random.default_rng(0)
    top = int(np.argmax(sc["G2"]))  # most profitable customer, forced into the explained sample
    idx = np.unique(np.concatenate([[top], rng.choice(len(Xte), 249, replace=False)]))
    Xs = Xte.iloc[idx]
    bg = Xtr.sample(100, random_state=0)
    f = lambda a: 1 - sel.surv(pd.DataFrame(a, columns=cols, index=range(len(a))))[:, 11]
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(bg.values, max_samples=100), seed=0)
    sv_surv = expl(Xs.values, max_evals=2 * len(cols) + 1)
    sv_surv.feature_names = cols
    xgbm = clf["xgb"].m
    sv_x = shap.TreeExplainer(xgbm)(Xte)
    imp = pd.DataFrame({"survival_pi12": np.abs(sv_surv.values).mean(0),
                        "xgboost_logodds": np.abs(sv_x.values).mean(0)}, index=cols)
    imp.sort_values("survival_pi12", ascending=False).to_csv(RES / "shap_importance.csv")
    rho = float(spearmanr(imp.survival_pi12, imp.xgboost_logodds)[0])
    # --- global bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ttl, colr in [(axes[0], "survival_pi12", f"{sel.name.split(' (')[0]} survival model: 12-month churn risk", C["blue"]),
                               (axes[1], "xgboost_logodds", "XGBoost classifier: churn log-odds", C["orange"])]:
        s = imp[col].sort_values().tail(12)
        ax.barh(s.index, s.values, color=colr, height=0.65); ax.set_title(ttl, fontsize=9.5)
        ax.grid(axis="y", visible=False); ax.set_xlabel("mean |SHAP value|")
    fig.subplots_adjust(wspace=0.45)
    savefig(fig, "06_shap_global.png")
    # --- local explanation for the top-profit customer
    pos = int(np.where(idx == top)[0][0])
    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(sv_surv[pos], max_display=10, show=False)
    plt.title("Why this customer is the top retention target (change in 12-month churn risk)",
              fontsize=10, loc="left")
    savefig(plt.gcf(), "07_shap_top_customer.png")
    return imp, rho


# ====================================================================== 6. group check
def step_groups(te, sc, probs, w_lost):
    w, lost = w_lost
    sel_G2 = sc["G2"] > 0; sel_G = sc["G"] > 0
    k = int(sel_G2.sum())
    sel_xgb = np.zeros(len(te), bool); sel_xgb[np.argsort(-probs["xgb_t"])[:k]] = True
    rows = []
    for gname, mask in [("Senior citizen", te.SeniorCitizen == 1), ("Partner", te.Partner == "Yes"),
                        ("Dependents", te.Dependents == "Yes")]:
        mask = mask.values
        for rname, sel in [("Refined rule (G2>0)", sel_G2), ("Paper rule (G>0)", sel_G),
                           (f"XGBoost+tenure top-{k} (same size)", sel_xgb)]:
            n_sel = int(sel.sum())
            ob = lambda m_: 100 * (1 - float(KaplanMeierFitter().fit(te["T"][m_], te.E[m_])
                                              .survival_function_at_times(12).iloc[0]))
            rows.append({"group": gname, "rule": rname, "n_selected": n_sel,
                         "obs_churn12_in_group_%": ob(mask), "obs_churn12_outside_%": ob(~mask),
                         "share_of_base_%": 100 * mask.mean(),
                         "share_of_selected_%": 100 * (sel & mask).sum() / max(n_sel, 1),
                         "selection_rate_in_group_%": 100 * (sel & mask).sum() / mask.sum(),
                         "selection_rate_outside_%": 100 * (sel & ~mask).sum() / (~mask).sum()})
    g = pd.DataFrame(rows); g["rate_ratio"] = g["selection_rate_in_group_%"] / g["selection_rate_outside_%"].replace(0, np.nan)
    g.to_csv(RES / "group_selection.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, gname in zip(axes, g.group.unique()):
        d = g[g.group == gname]
        names = ["Base"] + list(d.rule)
        vals = [d["share_of_base_%"].iloc[0]] + list(d["share_of_selected_%"])
        ax.bar(range(len(vals)), vals, color=[C["gray"], C["violet"], C["magenta"], C["blue"]][:len(vals)], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=C["ink2"])
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(["Base", "Refined", "Paper", "XGB"], fontsize=8)
        ax.set_title(f"{gname}: share of group", fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of customers"); savefig(fig, "08_group_shares.png")
    return g


# ====================================================================== 7. extra figures
def fig_calibration(te, Spred, probs, sel_key, surv):
    cals = {sel_key: calibration(1 - Spred[sel_key][:, 11], te),
            "xgb": calibration(probs["xgb"], te), "xgb_t": calibration(probs["xgb_t"], te)}
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot([0, 0.8], [0, 0.8], color=C["gray"], ls=":", lw=1.2, label="perfect")
    sty = {sel_key: (C["violet"], "-", "o", surv[sel_key].name), "xgb": (C["blue"], "--", "s", "XGBoost"),
           "xgb_t": (C["orange"], "-.", "^", "XGBoost + tenure")}
    for k, d in cals.items():
        col, ls, mk, lab = sty[k]
        ax.plot(d.pred_pi12, d.obs_pi12, color=col, ls=ls, marker=mk, ms=6, label=lab,
                markeredgecolor=C["surface"], markeredgewidth=1.5)
    ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed 12-month churn (Kaplan-Meier)")
    ax.set_title("Calibration at the 12-month horizon (test set)"); ax.legend(fontsize=8)
    savefig(fig, "03_calibration_12m.png")
    s = {k: {"mean_abs_error": float(d.abs_err.mean()), "max_abs_error": float(d.abs_err.max())} for k, d in cals.items()}
    pd.concat({k: d for k, d in cals.items()}).to_csv(RES / "calibration_deciles.csv")
    return s


def fig_individual_curves(te, Ssel, sc, sel_name):
    pi = sc["pi"]; order = np.argsort(pi)
    picks = {"Low risk": order[int(0.1 * len(order))], "Median risk": order[len(order) // 2],
             "High risk": order[int(0.95 * len(order))], "Top retention target": int(np.argmax(sc["G2"]))}
    cols = [C["blue"], C["orange"], C["aqua"], C["violet"]]; ls = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4)); t = np.arange(0, 73)
    for (lab, i), c_, l_ in zip(picks.items(), cols, ls):
        S = np.concatenate([[1.0], Ssel[i]])
        r = te.iloc[i]
        ax.plot(t, S, color=c_, ls=l_, label=f"{lab}: {r.Contract}, ${r.MonthlyCharges:.0f}/mo")
    ax.axvline(12, color=C["gray"], lw=1, ls=":"); ax.set_xlim(0, 36)
    ax.set_xlabel("Months since sign-up (tenure)"); ax.set_ylabel("P(still subscribed)")
    ax.set_title("Customer-specific survival curves (selected survival model)"); ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.02); savefig(fig, "09_customer_survival_curves.png")


# ====================================================================== 8. seed robustness
def step_seeds(df, hp, sel_key, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for sd in seeds:
        tr, te = split(df, sd); fb = FeatureBuilder().fit(tr)
        Xtr, Xte = fb.transform(tr), fb.transform(te)
        surv, clf, _ = fit_all(Xtr, tr["T"].values, tr.E.values, hp=hp, seed=sd)
        Ssel = surv[sel_key].surv(Xte)
        probs = {k: m.proba(clf_input(k, Xte, te["T"].values)) for k, m in clf.items()}
        out, _, _, _ = profit_table(te, Ssel, probs, BASE)
        pi12 = 1 - Ssel[:, 11]
        ytr, yte = surv_arrays(tr, te)
        rows.append({"seed": sd,
                     "cindex_survival": concordance_index(te["T"], 1 - pi12, te.E),
                     "cindex_xgb": concordance_index(te["T"], 1 - probs["xgb"], te.E),
                     "auc12_survival": float(cumulative_dynamic_auc(ytr, yte, pi12, [12])[0][0]),
                     "auc12_xgb": float(cumulative_dynamic_auc(ytr, yte, probs["xgb"], [12])[0][0]),
                     **{f"max|{r}": out[r]["max_profit"] for r in RULES[1:]},
                     "refined_at_G2>0": out["Expected profit (refined, G2)"]["profit_at_G>0"],
                     "paper_at_G>0": out["Expected profit (paper, G)"]["profit_at_G>0"],
                     "share_G2>0": out["Expected profit (refined, G2)"]["share_contacted_G>0"]})
        log(f"  seed {sd} done")
    d = pd.DataFrame(rows); d.to_csv(RES / "seed_robustness.csv", index=False)
    return d


# ====================================================================== main
def main():
    log("[1/8] loading + audit")
    df, audit = load_clean()
    tr, te = split(df, SEED)
    fb = FeatureBuilder().fit(tr); Xtr, Xte = fb.transform(tr), fb.transform(te)
    audit["train_n"], audit["test_n"] = len(tr), len(te)
    audit = step_audit_km_ph(df, audit, Xtr, tr)
    log("[2/8] tuning + fitting models (CV on training data only)")
    surv, clf, hp = fit_all(Xtr, tr["T"].values, tr.E.values, seed=SEED)
    sel_key = max(hp["cv_cindex"], key=hp["cv_cindex"].get)  # chosen on CV, never on the test set
    audit["selected_survival_model"] = surv[sel_key].name
    log(f"  selected survival model by CV C-index: {surv[sel_key].name}")
    log("[3/8] test-set evaluation")
    tbl, Spred, probs = step_models(tr, te, Xtr, Xte, surv, clf, hp)
    print(tbl.round(3).to_string())
    cal = fig_calibration(te, Spred, probs, sel_key, surv)
    Ssel = Spred[sel_key]
    log("[4/8] profit analysis (+bootstrap)")
    ptbl, pout, ci, oracle, aux = step_profit(te, Ssel, probs)
    print(ptbl.round(2).to_string())
    w, lost, sc, rs = aux
    fig_individual_curves(te, Ssel, sc, surv[sel_key].name)
    # extra: profit of each survival model's own refined rule (robustness to model choice)
    alt = {}
    for k, S in Spred.items():
        o, _, _, _ = profit_table(te, S, probs, BASE)
        alt[surv[k].name] = {"refined_at_G2>0": o["Expected profit (refined, G2)"]["profit_at_G>0"],
                             "paper_at_G>0": o["Expected profit (paper, G)"]["profit_at_G>0"],
                             "survival_risk_max": o["Survival risk (pi_12)"]["max_profit"]}
    pd.DataFrame(alt).T.to_csv(RES / "profit_by_survival_model.csv")
    log("[5/8] sensitivity")
    oat, grid = step_sensitivity(te, Ssel, probs)
    log("[6/8] SHAP")
    imp, rho = step_shap(sel_key, surv, clf, Xtr, Xte, te, sc)
    log("[7/8] group check + customer scores")
    groups = step_groups(te, sc, probs, (w, lost))
    out = te[["customerID", "Contract", "InternetService", "MonthlyCharges", "tenure", "Churn"]].copy()
    for t in (3, 6, 12, 24):
        out[f"churn_prob_{t}m"] = 1 - Ssel[:, t - 1]
    out["V_paper"], out["L_expected_loss"] = sc["V"], sc["L"]
    out["G_paper"], out["G2_refined"] = sc["G"], sc["G2"]
    out["recommend_contact"] = (sc["G2"] > 0).astype(int)
    out["priority_rank"] = (-sc["G2"]).argsort().argsort() + 1
    out.sort_values("priority_rank").to_csv(RES / "customer_targeting_table_test.csv", index=False)
    log("[8/8] seed robustness")
    seeds = step_seeds(df, hp, sel_key)
    # persist models + config
    with open(MOD / "models.pkl", "wb") as fh:
        pickle.dump({"feature_builder": fb, "survival": surv, "classifiers": clf,
                     "selected": sel_key, "hyperparameters": hp, "base_params": BASE}, fh)
    summary = {"audit": audit, "hyperparameters": hp, "calibration": cal, "shap_spearman": rho,
               "oracle_profit_per_1000": oracle, "alt_survival_models": alt,
               "seed_means": seeds.drop(columns="seed").mean().to_dict(),
               "seed_sds": seeds.drop(columns="seed").std().to_dict()}
    json.dump(summary, open(RES / "summary.json", "w"), indent=2, default=float)
    log("done")


if __name__ == "__main__":
    main()
