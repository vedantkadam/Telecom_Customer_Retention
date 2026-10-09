# Survival-Based, Profit-Driven Customer Retention Targeting

Implementation of the framework proposed in the first-review paper *"Survival-Based, Profit-Driven Customer
Retention Targeting in Telecom Subscription Services"* (Goa Institute of Management, Advanced Machine Learning
for Business Transformation), run on the public IBM Telco customer-churn sample (7,043 customers, 21 fields).

The paper stopped at a validated problem definition and data audit. This repository carries out the plan it
set for the next phase: Kaplan-Meier curves, survival models, a churn-classifier baseline, an expected-profit
targeting rule, profit curves, sensitivity checks, SHAP explanations and a group-level selection check.

**Start with `results/RESULTS.md`** (all tables and figures, generated from the output files).

## What was found (base case: H = 12 months, margin 40%, offer cost $20, success rate 25% *assumed*)

- The paper's data audit reproduces (churn ~26.5%, contract effects, tenure medians, charges), apart from last-digit differences caused by dropping the 11 tenure-0 rows.
- Survival models rank 12-month churn risk better than like-for-like classifiers (AUC@12m 0.914 vs 0.857) and are
  far better calibrated at the horizon (mean decile error 0.024 vs 0.100). Cox, Weibull AFT and the survival forest are close.
- Profit: survival-risk ranking beat the best like-for-like classifier's peak in 5/5 further random splits, but the margin is
  modest relative to bootstrap noise on the main split (95% interval for the difference: -$66 to +$519 per 1,000 customers).
- **The paper's literal rule (Eq. 3, G = q·π·V − c) underperformed.** V is the survival-weighted value a customer *generates*, not the
  margin lost if they leave. A refined score (`G2 = q·L − c`, L = expected margin lost inside the horizon) earned $1,256 vs $701 per
  1,000 customers at its own stopping point on the main split, and beat the paper's rule on all 5 other splits. Note G2 matches the
  evaluation's value definition by construction, so this comparison is not independent evidence.
- **Classifiers that use tenure as a feature beat everything, but that is leakage here.** Ranking by short tenure alone scores AUC 1.000
  against the 12-month outcome, because tenure is the duration that defines it. Under the paper's formulation (risk from sign-up)
  tenure is not known at decision time. These rows are reported separately and labelled as a leakage reference.
- Profit is highly sensitive to the success rate q, which this dataset cannot identify; at low q, high offer cost or a 6-month horizon
  the profit rule correctly recommends no campaign.

## Layout

```
data/            put WA_Fn-UseC_-Telco-Customer-Churn.csv here (Kaggle; not redistributed)
src/data.py        cleaning, tenure-0 handling, features, stratified 70/30 split
src/models.py      Cox (elastic net, stratified), Weibull AFT, random survival forest, logistic, XGBoost
src/economics.py   horizon risk, value and profit scores, IPCW realised-profit evaluation
src/run_pipeline.py  the whole analysis -> results/, figures/, models/
src/make_report.py   builds results/RESULTS.md from the outputs
src/score_new.py     score new customers with the saved model
results/           CSV/JSON outputs, RESULTS.md, customer_targeting_table_test.csv
figures/           9 figures
models/models.pkl  fitted models + preprocessing + hyper-parameters
```

## Run it

```bash
pip install -r requirements.txt
python -I src/run_pipeline.py     # ~1 minute; seed 42, 5 extra seeds for robustness
python -I src/make_report.py
python -I src/score_new.py data/WA_Fn-UseC_-Telco-Customer-Churn.csv scored.csv
```

## How it maps to the paper

| Paper | Here |
|---|---|
| III-A data, 11 blank TotalCharges (tenure 0) dropped | `data.load_clean`; n = 7,032 |
| III-B duration/event encoding, π (Eq. 1), V (Eq. 2), G (Eq. 3) | `economics.scores` (`pi`, `V`, `G`) |
| III-C leakage control; stratified 70/30 on event × contract; fit on train only | `data.split`, `FeatureBuilder` |
| III-D KM, elastic-net Cox with PH check, Weibull AFT, random survival forest; logistic + XGBoost; SHAP | `run_pipeline.py` steps 1-6 |
| III-E C-index, integrated Brier, AUC@6/12, calibration; classifier ROC-AUC/P/R/F1; profit vs share contacted; group shares | `results/model_comparison.csv`, `profit_comparison.csv`, `group_selection.csv`, figures 03-08 |
| III-F Table I and sensitivity | base values and ranges in `run_pipeline.py` (`BASE`, `SENS`); `sensitivity_*.csv` |

## Decisions and deviations - read before citing numbers

1. **Realised-profit evaluation.** There are no past offers, so profit uses the assumed q: contacting a customer earns
   `q × (margin lost inside the horizon if they actually churned) − c`. Customers censored before the horizon have unknown status and are
   handled by inverse-probability-of-censoring weights. Baselines' peaks use the best cut-off picked with test labels (favourable to them);
   the profit rules stop on their own score.
2. **Added `G2` (refined rule)**, alongside the paper's `G`. Both are reported.
3. **Added tenure-augmented classifiers and a tenure-only check** (not in the paper) to expose the leakage above.
4. **Model selection** among survival models by 3-fold CV C-index on the training data only (Cox chosen). Hyper-parameters tuned by CV on training data.
   Weibull AFT gives noticeably higher profit than Cox at the refined rule's stopping point ($1,522 vs $1,256), so model choice matters about as much as rule choice.
5. **Time-dependent metrics** truncate test follow-up at 60 months (the censoring distribution is ~0 near 72); integrated Brier covers months 1-59.
6. **Stratified Cox** on contract (the paper's stated fallback); elastic-net mixing fixed at 0.5. PH is still questionable for a few covariates (see `results/proportional_hazards_test.csv`).
7. **Risk is measured from sign-up**, as the paper defines it. It is not conditional on months already survived, so applying scores to existing customers
   (`score_new.py`) is a simplification; conditional survival S(t0+H)/S(t0) is the natural next step.
8. `TotalCharges` is excluded (correlation 0.9996 with tenure × MonthlyCharges); `tenure` is the duration, not a feature, in the survival and like-for-like models.

## Limitations (also in the paper)

No intervention data (q and c are assumptions, no uplift claims); cross-sectional snapshot with no dates; service flags may reflect the plan at or after departure;
tenure capped at 72; teaching sample from an unnamed operator; SHAP describes the models, not why customers leave. The group check shows seniors selected
~1.8× as often as non-seniors and partners/dependents far less often, in line with observed churn gaps; any real use would need a policy decision on that.

## Environment

`autograd-gamma` (a lifelines dependency) failed to build in this environment. It is only used by lifelines' generalized-gamma fitter, which this project does not use,
so a small stub module was placed in site-packages and `autograd` installed separately. lifelines also declares `pandas<3` while pandas 3.0.5 was used; the fitters used here
ran without errors. On a normal environment `pip install lifelines` should work and pandas 2.x is the safer choice (not re-run with it).

The Cell2Cell files that accompanied the task were not used: the holdout file has no churn labels, and its churn flag is a short-horizon outcome with tenure measured at
observation rather than time to churn, so the paper's duration/event formulation does not apply to it.
