"""Survival models and churn-classifier baselines (paper Sec. III-D).

Every survival wrapper exposes .surv(X) -> ndarray (n, 72): S(t | x) for t = 1..72 months.
Every classifier wrapper exposes .proba(X) -> ndarray (n,): P(churn flag = 1).
"""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from xgboost import XGBClassifier

from data import MAX_T

warnings.filterwarnings("ignore")
TIMES = np.arange(1, MAX_T + 1)
CONTRACT_COLS = ["Contract_OneYear", "Contract_TwoYear"]


# ---------------------------------------------------------------- survival wrappers
class CoxStrat:
    """Elastic-net Cox model stratified on contract type (PH fails for contract)."""
    name = "Cox (elastic net, strat. contract)"

    def __init__(self, penalizer=0.05, l1_ratio=0.5):
        self.penalizer, self.l1_ratio = penalizer, l1_ratio

    def _frame(self, X):
        d = X.drop(columns=CONTRACT_COLS).copy()
        d["strata"] = np.select([X.Contract_TwoYear == 1, X.Contract_OneYear == 1], [2, 1], 0)
        return d

    def fit(self, X, T, E):
        d = self._frame(X); d["T"], d["E"] = np.asarray(T), np.asarray(E)
        self.m = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        self.m.fit(d, "T", "E", strata=["strata"])
        return self

    def surv(self, X):
        # lifelines returns stratified predictions grouped by stratum, not in input order:
        # re-select columns by the input index so rows line up with customers.
        sf = self.m.predict_survival_function(self._frame(X), times=TIMES)
        return sf[list(X.index)].T.values


class WeibullAFT:
    name = "Weibull AFT"

    def __init__(self, penalizer=0.01):
        self.penalizer = penalizer

    def fit(self, X, T, E):
        d = X.copy(); d["T"], d["E"] = np.asarray(T), np.asarray(E)
        self.m = WeibullAFTFitter(penalizer=self.penalizer).fit(d, "T", "E")
        return self

    def surv(self, X):
        return self.m.predict_survival_function(X, times=TIMES).T.values


class RSF:
    name = "Random survival forest"

    def __init__(self, min_samples_leaf=15, n_estimators=300, seed=0):
        self.msl, self.n, self.seed = min_samples_leaf, n_estimators, seed

    def fit(self, X, T, E):
        y = Surv.from_arrays(np.asarray(E).astype(bool), np.asarray(T).astype(float))
        self.m = RandomSurvivalForest(n_estimators=self.n, min_samples_leaf=self.msl,
                                      max_features="sqrt", n_jobs=-1, random_state=self.seed)
        self.m.fit(X.values, y)
        return self

    def surv(self, X):
        S = self.m.predict_survival_function(X.values, return_array=True)
        ut = self.m.unique_times_
        idx = np.searchsorted(ut, TIMES, side="right") - 1  # step function at integer t
        out = np.where(idx[None, :] >= 0, S[:, np.clip(idx, 0, None)], 1.0)
        return out


# ---------------------------------------------------------------- classifier wrappers
class Logit:
    def __init__(self, name="Logistic regression"):
        self.name = name

    def fit(self, X, E):
        self.m = LogisticRegressionCV(Cs=10, cv=3, scoring="roc_auc", max_iter=2000).fit(X, E)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)[:, 1]


class XGB:
    def __init__(self, max_depth=3, n_estimators=250, seed=0, name="XGBoost"):
        self.name = name
        self.kw = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=seed,
                       eval_metric="logloss", n_jobs=-1)

    def fit(self, X, E):
        self.m = XGBClassifier(**self.kw).fit(X, E)
        return self

    def proba(self, X):
        return self.m.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- tuning / fitting
def with_tenure(X, T):
    """Add tenure (in years) as an ordinary feature for the '+ tenure' classifier baselines."""
    return X.assign(tenure=np.asarray(T, dtype=float) / 12.0)


def clf_input(key, X, T):
    """Classifier input for a given key: the '_t' variants get the tenure column."""
    return with_tenure(X, T) if key.endswith("_t") else X


def _cv_cindex(make, X, T, E, folds=3, seed=0):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = []
    for a, b in skf.split(X, E):
        m = make().fit(X.iloc[a], T[a], E[a])
        s12 = m.surv(X.iloc[b])[:, 11]  # S(12): higher = safer
        scores.append(concordance_index(T[b], s12, E[b]))
    return float(np.mean(scores))


def _cv_auc(make, X, E, folds=3, seed=0):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(np.mean([roc_auc_score(E[b], make().fit(X.iloc[a], E[a]).proba(X.iloc[b]))
                          for a, b in skf.split(X, E)]))


def fit_all(Xtr, Ttr, Etr, hp=None, seed=0, verbose=True):
    """Fit all models. hp=None -> tune by 3-fold CV on the training data only."""
    Ttr, Etr = np.asarray(Ttr), np.asarray(Etr)
    cv = {}
    if hp is None:
        hp = {}
        grids = {
            "cox": ("penalizer", [0.01, 0.05, 0.1, 0.3], lambda v: CoxStrat(v, 0.5)),
            "aft": ("penalizer", [0.001, 0.01, 0.1], lambda v: WeibullAFT(v)),
            "rsf": ("min_samples_leaf", [15, 40], lambda v: RSF(v, n_estimators=120, seed=seed)),
        }
        for key, (pname, vals, mk) in grids.items():
            res = {v: _cv_cindex(lambda v=v: mk(v), Xtr, Ttr, Etr, seed=seed) for v in vals}
            best = max(res, key=res.get)
            hp[key] = best; cv[key] = res[best]
            if verbose: print(f"  CV {key}: {res} -> {best}")
        res = {d: _cv_auc(lambda d=d: XGB(d, seed=seed), Xtr, Etr, seed=seed) for d in [2, 3, 4]}
        hp["xgb_depth"] = max(res, key=res.get)
        if verbose: print(f"  CV xgb depth: {res} -> {hp['xgb_depth']}")
        Xt_ = with_tenure(Xtr, Ttr)
        res = {d: _cv_auc(lambda d=d: XGB(d, seed=seed), Xt_, Etr, seed=seed) for d in [2, 3, 4]}
        hp["xgb_t_depth"] = max(res, key=res.get)
        if verbose: print(f"  CV xgb+tenure depth: {res} -> {hp['xgb_t_depth']}")
        hp["cv_cindex"] = cv
    surv = {
        "cox": CoxStrat(hp["cox"], 0.5).fit(Xtr, Ttr, Etr),
        "aft": WeibullAFT(hp["aft"]).fit(Xtr, Ttr, Etr),
        "rsf": RSF(hp["rsf"], 300, seed).fit(Xtr, Ttr, Etr),
    }
    Xt = with_tenure(Xtr, Ttr)
    clf = {"logit": Logit().fit(Xtr, Etr),
           "xgb": XGB(hp["xgb_depth"], seed=seed).fit(Xtr, Etr),
           # conventional strong baselines: tenure at observation used as an ordinary feature
           "logit_t": Logit("Logistic regression + tenure").fit(Xt, Etr),
           "xgb_t": XGB(hp["xgb_t_depth"], seed=seed, name="XGBoost + tenure").fit(Xt, Etr)}
    return surv, clf, hp
