"""
evaluation.py - how good is the model, and could the result just be luck?

  * Test-period accuracy and ROC-AUC against a naive "always the same answer" forecast
  * A bootstrap range for the test AUC
  * Placebo test: shuffle the training labels 200 times (a model with no real
    information) and see where the real model sits
  * Coefficients turned into odds ratios, and the same model refitted on
    Harbour-only data from April 2021 as a robustness check
  * Money test: hold Harbour when the model says "outperform", otherwise the FTSE 250
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

from data import SEED, FEATURES, save_table

N_PLACEBO = 200
N_BOOTSTRAP = 2000


def scores(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "roc_auc": roc_auc_score(y_true, prob),
            "share_predicted_outperform": pred.mean()}


def test_results(y_train, y_test, p_test):
    res = scores(y_test, p_test)
    majority = int(y_train.mean() >= 0.5)       # the more common answer IN TRAINING
    naive_pred = np.full(len(y_test), majority)
    if majority:
        naive_name = "Naive: always outperform"
    else:
        naive_name = "Naive: always underperform"
    results = pd.DataFrame([
        {"model": "Logistic regression", **res},
        {"model": naive_name, "accuracy": accuracy_score(y_test, naive_pred),
         "balanced_accuracy": balanced_accuracy_score(y_test, naive_pred),
         "roc_auc": 0.5, "share_predicted_outperform": float(majority)},
    ])
    save_table(results, "T05_test_results", index=False)
    cm = pd.DataFrame(confusion_matrix(y_test, (p_test >= 0.5).astype(int)),
                      index=["actual: under", "actual: out"],
                      columns=["predicted: under", "predicted: out"])
    save_table(cm, "T06_confusion_matrix")
    print(results.round(3).to_string(index=False))
    print(cm)
    return res, results


def bootstrap_auc(y_test, p_test):
    # resample the test months with replacement 2,000 times: how much could the AUC move?
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y_test), len(y_test))
        if y_test.iloc[i].nunique() == 2:       # AUC needs both answers in the sample
            aucs.append(roc_auc_score(y_test.iloc[i], p_test[i]))
    return np.percentile(aucs, [2.5, 97.5])


def placebo(model, X_train, y_train, X_test, y_test):
    aucs = []
    for k in range(N_PLACEBO):
        y_fake = y_train.sample(frac=1, random_state=SEED + k).values   # shuffled labels
        fake = clone(model).fit(X_train, y_fake)
        aucs.append(roc_auc_score(y_test, fake.predict_proba(X_test)[:, 1]))
    aucs = np.array(aucs)
    save_table(pd.DataFrame({"placebo_auc": aucs}), "T08_placebo", index=False)
    return aucs


def coefficients(model, data):
    coefs = model.named_steps["model"].coef_[0]
    coef = pd.DataFrame({"feature": FEATURES, "coef_per_1sd": coefs,
                         "odds_ratio_per_1sd": np.exp(coefs)})
    # robustness: the same model fitted on the genuine-Harbour months only
    post = data[data.index >= "2021-04"]
    m_post = clone(model).fit(post[FEATURES], post["y"].astype(int))
    coef["coef_harbour_only_2021_2026"] = m_post.named_steps["model"].coef_[0]
    coef["same_sign"] = np.sign(coef["coef_per_1sd"]) == np.sign(coef["coef_harbour_only_2021_2026"])
    save_table(coef, "T07_coefficients", index=False)
    print(coef.round(3).to_string(index=False))


def money_test(test, p_test):
    hold_hbr = p_test >= 0.5
    strategy = np.where(hold_hbr, test["r_hbr_next"], test["r_ftse_next"])
    money = pd.DataFrame({
        "strategy": ["Model switch (HBR or FTSE 250)", "Buy and hold HBR", "Buy and hold FTSE 250"],
        "cumulative_return": [np.expm1(strategy.sum()), np.expm1(test["r_hbr_next"].sum()),
                              np.expm1(test["r_ftse_next"].sum())]})
    save_table(money, "T09_money_test", index=False)
