"""
models.py - the machine-learning model: logistic regression

Question: will Harbour Energy beat the FTSE 250 next month? (1 = yes, 0 = no)

How we stop the model "seeing the future" (data leakage):
  * Chronological split: train 2011-2021, test 2022-2026, no shuffling.
  * Capping extreme values and scaling sit INSIDE the Pipeline, so they are
    learned from training data only and then applied unchanged to new data.
  * The one setting we tune (C, how strongly the model is held back) is chosen
    by time-series cross-validation inside the training years only.
  * Fixed random seed (42) so every number repeats exactly.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import SEED, FEATURES, TRAIN_END, PROCESSED, save_table

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]        # small C = simpler, more cautious model


class Winsorizer(BaseEstimator, TransformerMixin):
    # caps each predictor at its 1st and 99th percentile in the TRAINING data
    # (Premier Oil's 2016 and 2020 crises produced huge one-month moves)
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.quantile(X, self.q, axis=0)
        self.hi_ = np.quantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_pipeline():
    return Pipeline([
        ("winsor", Winsorizer(0.01)),           # cap outliers
        ("scale", StandardScaler()),            # put every predictor on the same scale
        ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def split_data():
    data = pd.read_csv(PROCESSED / "monthly_dataset.csv", index_col="month")
    live = pd.read_csv(PROCESSED / "live_row.csv", index_col="month")
    train = data[data.index <= TRAIN_END]
    test = data[data.index > TRAIN_END]
    assert train.index.max() < test.index.min()  # every training month comes before every test month
    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} months)")
    print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} months)")
    return data, train, test, live


def tune(train):
    # try each C with 5-fold time-series cross-validation (1-month gap between folds)
    X, y = train[FEATURES], train["y"].astype(int)
    search = GridSearchCV(make_pipeline(), {"model__C": C_VALUES},
                          cv=TimeSeriesSplit(n_splits=5, gap=1), scoring="roc_auc")
    search.fit(X, y)
    cv = pd.DataFrame(search.cv_results_)[["param_model__C", "mean_test_score", "std_test_score"]]
    cv.columns = ["C", "cv_auc_mean", "cv_auc_sd"]
    save_table(cv, "T04_cv_tuning", index=False)
    print(f"Chosen C = {search.best_params_['model__C']} (cross-validated AUC {search.best_score_:.3f})")
    return search                               # search.best_estimator_ is refitted on all training data


def walk_forward(model, data, test):
    # robustness: refit every January on all earlier months and predict that year only
    predictions = []
    for year in sorted({m[:4] for m in test.index}):
        fit_rows = data[data.index < f"{year}-01"]
        pred_rows = test[test.index.str.startswith(year)]
        assert fit_rows.index.max() < pred_rows.index.min()
        m = clone(model).fit(fit_rows[FEATURES], fit_rows["y"].astype(int))
        predictions.append(pd.Series(m.predict_proba(pred_rows[FEATURES])[:, 1], index=pred_rows.index))
    return pd.concat(predictions)


def live_signal(model, data, live):
    # refit on every labelled month, then predict the month after the newest data
    final = clone(model).fit(data[FEATURES], data["y"].astype(int))
    return final.predict_proba(live[FEATURES])[:, 1][0]
