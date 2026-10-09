"""Score customers in the Telco CSV format with the saved models.

    python -I src/score_new.py path/to/customers.csv [out.csv]

Columns required: the 21 Telco columns (Churn / TotalCharges / tenure are not used as features).
IMPORTANT: scores are 12-month churn risk *from sign-up* (the paper's formulation, t = tenure).
For customers who are already months into their contract the right quantity is conditional survival
S(t0+H)/S(t0); that extension is not implemented here.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data import ROOT
from economics import scores


def main(src, dst="scored_customers.csv"):
    art = pickle.load(open(ROOT / "models" / "models.pkl", "rb"))
    df = pd.read_csv(src)
    X = art["feature_builder"].transform(df)
    model = art["survival"][art["selected"]]
    S = model.surv(X)
    p = art["base_params"]
    sc = scores(S, df.MonthlyCharges.values, p["H"], p["m"], p["q"], p["c"], p["gamma"])
    out = pd.DataFrame({"customerID": df.get("customerID", pd.Series(range(len(df)))),
                        "churn_prob_6m": 1 - S[:, 5], "churn_prob_12m": sc["pi"],
                        "expected_margin_lost_12m": sc["L"], "net_gain_if_contacted": sc["G2"],
                        "recommend_contact": (sc["G2"] > 0).astype(int)})
    out.sort_values("net_gain_if_contacted", ascending=False).to_csv(dst, index=False)
    print(f"{len(out)} customers scored, {int(out.recommend_contact.sum())} recommended -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
