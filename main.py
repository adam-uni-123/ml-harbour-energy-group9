"""
FI4008 Group Assessment (Group 9) - Harbour Energy plc (LSE: HBR), BUY or SELL?

This runs the whole analysis from start to finish:
  1. checks the raw data and builds the monthly table
  2. the financial and market evidence (share record, results, comparables)
  3. the machine-learning model: can we predict if Harbour beats the FTSE 250 next month?
  4. the valuation: DCF, comparables, precedent deals and LBO, blended into a target price

Run it from this folder (it needs data/raw/):  python main.py
Takes about 1 minute and rewrites everything in tables/ and figures/.
"""

import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
import matplotlib

import data
import market
import models
import evaluation
import dcf
import valuation
import plots
from data import SEED, FEATURES, PRICE_PENCE, save_table

np.random.seed(SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(f"Python {platform.python_version()} | numpy {np.__version__} | pandas {pd.__version__} | "
      f"scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")

# 1. Check the raw files have not changed, then build the monthly table for the model
print("\n--- 1. Data ---")
data.check_raw_files()
data.build_model_table()
plots.prices()

# 2. Financial and market evidence
print("\n--- 2. Market evidence ---")
total_returns = market.share_record()
market.financial_summary()
market.comparables()
plots.relative_performance(total_returns)

# 3. Machine learning: tune on 2011-2021, test once on 2022-2026
print("\n--- 3. Model ---")
all_months, train, test, live = models.split_data()
X_train, y_train = train[FEATURES], train["y"].astype(int)
X_test, y_test = test[FEATURES], test["y"].astype(int)

search = models.tune(train)
model = search.best_estimator_
p_test = model.predict_proba(X_test)[:, 1]
res, results = evaluation.test_results(y_train, y_test, p_test)
ci = evaluation.bootstrap_auc(y_test, p_test)
placebo_aucs = evaluation.placebo(model, X_train, y_train, X_test, y_test)
evaluation.coefficients(model, all_months)
p_walk = models.walk_forward(model, all_months, test)
auc_walk = roc_auc_score(y_test, p_walk.loc[y_test.index])
evaluation.money_test(test, p_test)
p_live = models.live_signal(model, all_months, live)

summary = pd.Series({
    "train_months": len(train), "test_months": len(test),
    "test_outperform_share": y_test.mean(), "chosen_C": search.best_params_["model__C"],
    "cv_auc": search.best_score_, "test_auc": res["roc_auc"],
    "test_auc_ci_low": ci[0], "test_auc_ci_high": ci[1],
    "test_accuracy": res["accuracy"], "naive_accuracy": results.loc[1, "accuracy"],
    "test_balanced_accuracy": res["balanced_accuracy"],
    "placebo_auc_mean": placebo_aucs.mean(),
    "placebo_p_value": (placebo_aucs >= res["roc_auc"]).mean(),   # share of placebos at least as good
    "walk_forward_auc": auc_walk,
    "live_month": live.index[0], "live_prob_outperform_next_month": p_live,
})
save_table(summary, "T10_model_summary", header=["value"])
predictions = pd.DataFrame({"prob": p_test, "y": y_test.values, "excess_next": test["excess_next"].values},
                           index=test.index)
save_table(predictions, "T11_test_predictions")
print(summary.to_string())
plots.test_results()
plots.coefficients()
plots.placebo()

# 4. Valuation: four methods, blended with the team's weights
print("\n--- 4. Valuation ---")
cash_flows, dcf_base, dcf_low, dcf_high = dcf.run()
value_summary = valuation.run(cash_flows, dcf_base, dcf_low, dcf_high)
plots.football_field(value_summary)

# 5. Summary of the checks. Each one is an assert earlier in the run, so getting
#    this far means they all passed
print("\n--- 5. Checks ---")
for item in ["Raw files match their recorded SHA-256 fingerprints",
             "Target alignment: y on row t equals whether Harbour beat the FTSE 250 in month t+1",
             "Chronology: every training month comes before every test month",
             "Walk-forward: each yearly refit only uses months before the year it predicts"]:
    print("  ✓", item)
target = value_summary.loc["Blended (team weights)", "base_p"]
print(f"\nTarget price {target:.0f}p vs share price {PRICE_PENCE}p "
      f"({100 * (target / PRICE_PENCE - 1):+.1f}%) -> "
      f"{'SELL' if target < PRICE_PENCE else 'BUY'}")
