"""Data loading, cleaning, feature engineering and splitting (paper Sec. III-A, III-C)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC_SCALED = ["MonthlyCharges", "n_addons", "price_ratio"]
MAX_T = 72  # tenure is capped at 72 months in the data


def load_clean(path=DATA_PATH):
    """Return (clean_df, audit_dict). Repairs TotalCharges and drops tenure-0 rows."""
    raw = pd.read_csv(path)
    audit = {"rows": len(raw), "cols": raw.shape[1],
             "duplicates": int(raw.duplicated().sum()),
             "unique_ids": bool(raw.customerID.is_unique)}
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    audit["missing_total_charges"] = int(raw.TotalCharges.isna().sum())
    audit["missing_all_tenure0"] = bool((raw.loc[raw.TotalCharges.isna(), "tenure"] == 0).all())
    df = raw[raw.tenure > 0].reset_index(drop=True)  # zero-duration rows carry no survival info
    audit["rows_after_drop"] = len(df)
    df["T"] = df["tenure"].astype(int)
    df["E"] = (df["Churn"] == "Yes").astype(int)
    # TotalCharges ~ tenure * MonthlyCharges -> check before admitting as a feature
    audit["corr_totalcharges_vs_tenure_x_monthly"] = float(
        np.corrcoef(df.TotalCharges, df.tenure * df.MonthlyCharges)[0, 1])
    audit["corr_totalcharges_vs_tenure"] = float(np.corrcoef(df.TotalCharges, df.tenure)[0, 1])
    return df, audit


def split(df, seed=42, test_size=0.3):
    """Stratified 70/30 split on the joint (event, contract) label."""
    strata = df["E"].astype(str) + "_" + df["Contract"]
    tr, te = train_test_split(df.index, test_size=test_size, random_state=seed, stratify=strata)
    return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)


class FeatureBuilder:
    """Fit on the training portion only; transform any frame.

    Output columns are numeric. TotalCharges and tenure are NOT features
    (tenure is the duration; TotalCharges is ~ tenure x MonthlyCharges).
    """

    def fit(self, train):
        self.plan_mean_ = train.groupby("InternetService")["MonthlyCharges"].mean().to_dict()
        X = self._raw(train)
        self.mu_ = X[NUMERIC_SCALED].mean()
        self.sd_ = X[NUMERIC_SCALED].std(ddof=0).replace(0, 1.0)
        self.columns_ = list(X.columns)
        return self

    def _raw(self, d):
        X = pd.DataFrame(index=d.index)
        X["gender_Male"] = (d.gender == "Male").astype(int)
        X["SeniorCitizen"] = d.SeniorCitizen.astype(int)
        X["Partner"] = (d.Partner == "Yes").astype(int)
        X["Dependents"] = (d.Dependents == "Yes").astype(int)
        X["PhoneService"] = (d.PhoneService == "Yes").astype(int)
        X["MultipleLines"] = (d.MultipleLines == "Yes").astype(int)
        X["Internet_DSL"] = (d.InternetService == "DSL").astype(int)
        X["Internet_Fiber"] = (d.InternetService == "Fiber optic").astype(int)
        for a in ADDONS:
            X[a] = (d[a] == "Yes").astype(int)  # 'No internet service' -> 0
        X["n_addons"] = X[ADDONS].sum(axis=1)
        X["Contract_OneYear"] = (d.Contract == "One year").astype(int)
        X["Contract_TwoYear"] = (d.Contract == "Two year").astype(int)
        X["PaperlessBilling"] = (d.PaperlessBilling == "Yes").astype(int)
        X["Pay_CreditCard"] = (d.PaymentMethod == "Credit card (automatic)").astype(int)
        X["Pay_ElectronicCheck"] = (d.PaymentMethod == "Electronic check").astype(int)
        X["Pay_MailedCheck"] = (d.PaymentMethod == "Mailed check").astype(int)
        X["MonthlyCharges"] = d.MonthlyCharges.astype(float)
        X["price_ratio"] = d.MonthlyCharges / d.InternetService.map(self.plan_mean_)
        return X

    def transform(self, d):
        X = self._raw(d)
        X[NUMERIC_SCALED] = (X[NUMERIC_SCALED] - self.mu_) / self.sd_
        return X[self.columns_].astype(float)
