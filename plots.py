"""
plots.py - all six figures, drawn from the saved tables

F01  Harbour vs FTSE 250 since 2011, and the Brent oil price
F02  Harbour vs FTSE 250 and peers since April 2021
F03  Model test results: ROC curve and confusion matrix
F04  What the model learned: odds ratios
F05  Placebo check
F06  Valuation "football field"
"""

import matplotlib
matplotlib.use("Agg")                           # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
from sklearn.metrics import roc_curve

from data import FIG_DIR, TAB_DIR, PRICE_PENCE, bloomberg_monthly, yahoo_monthly

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#f0efec"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "figure.dpi": 200, "savefig.bbox": "tight",
})
TEST_START = pd.Period("2022-01", "M")


def shade_test(ax, index):
    # grey band over the test period
    ax.axvspan(TEST_START.to_timestamp(), index[-1].to_timestamp(), color=SHADE, zorder=0)
    ax.text(TEST_START.to_timestamp(), ax.get_ylim()[1], "  test period", va="top", color=INK2, fontsize=8)


def prices():
    hbr = bloomberg_monthly("HBR_monthly")["total_return"]
    ftse = bloomberg_monthly("FTSE250_monthly")["total_return"]
    brent = yahoo_monthly("BRENT_FUT")
    start = pd.Period("2011-01", "M")
    hbr, ftse, brent = hbr[hbr.index >= start], ftse[ftse.index >= start], brent[brent.index >= start]

    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a.plot(hbr.index.to_timestamp(), 100 * hbr / hbr.iloc[0], color=BLUE,
           label="Harbour Energy (Premier Oil before Apr 2021)")
    a.plot(ftse.index.to_timestamp(), 100 * ftse / ftse.iloc[0], color=ORANGE, label="FTSE 250")
    a.set_yscale("log")
    a.set_ylabel("Total return, Jan 2011 = 100 (log scale)")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    a.axvline(pd.Timestamp("2021-04-01"), color=INK2, lw=0.8, ls=":")
    a.text(pd.Timestamp("2021-03-01"), 25, "Premier becomes\nHarbour", color=INK2, fontsize=8, ha="right")
    shade_test(a, hbr.index)
    b.plot(brent.index.to_timestamp(), brent, color=AQUA)
    b.set_ylabel("Brent, US$/bbl")
    shade_test(b, brent.index)
    fig.suptitle("Harbour has lagged the FTSE 250 and moves with oil", x=0.02, y=1.02, ha="left",
                 color=INK, fontsize=11, fontweight="bold")
    fig.savefig(FIG_DIR / "F01_prices.png")
    plt.close(fig)


def relative_performance(tr):
    start = pd.Period("2021-04", "M")
    x = tr[tr.index >= start]
    x = 100 * x / x.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for col in ["EnQuest", "Aker BP", "DNO", "Shell", "BP"]:       # peers in grey
        if col in x:
            ax.plot(x.index.to_timestamp(), x[col], color="#c3c2b7", lw=1.2)
            ax.text(x.index[-1].to_timestamp(), x[col].iloc[-1], f" {col}", color="#8a8984", fontsize=7, va="center")
    for col, colour in [("Harbour Energy", BLUE), ("FTSE 250 (benchmark)", ORANGE)]:
        ax.plot(x.index.to_timestamp(), x[col], color=colour, lw=2.4)
        ax.text(x.index[-1].to_timestamp(), x[col].iloc[-1], f" {col.split(' (')[0]}", color=colour,
                fontsize=8, va="center", fontweight="bold")
    ax.axhline(100, color="#52514e", lw=0.8)
    ax.set_ylabel("Total return, Apr 2021 = 100")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.set_title("Since becoming Harbour, the shares trail the FTSE 250 and most peers", loc="left",
                 fontsize=11, fontweight="bold")
    fig.savefig(FIG_DIR / "F02_relative_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def test_results():
    pred = pd.read_csv(TAB_DIR / "T11_test_predictions.csv", index_col=0)
    summary = pd.read_csv(TAB_DIR / "T10_model_summary.csv", index_col=0)["value"]
    cm = pd.read_csv(TAB_DIR / "T06_confusion_matrix.csv", index_col=0).values
    fpr, tpr, _ = roc_curve(pred["y"], pred["prob"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    a.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Coin flip (AUC 0.50)")
    a.plot(fpr, tpr, color=BLUE, label=f"Model (AUC {float(summary['test_auc']):.2f})")
    a.set_xlabel("False-alarm rate")
    a.set_ylabel("Hit rate")
    a.set_title("ROC curve", color=INK, loc="left")
    a.legend(frameon=False, loc="lower right")
    a.set_aspect("equal")
    b.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2):
            b.text(j, i, cm[i, j], ha="center", va="center", color=INK, fontsize=12)
    b.set_xticks([0, 1], ["Predicted\nunderperform", "Predicted\noutperform"])
    b.set_yticks([0, 1], ["Actual\nunderperform", "Actual\noutperform"])
    b.grid(False)
    b.set_title("Confusion matrix (57 months)", color=INK, loc="left")
    for spine in b.spines.values():
        spine.set_visible(False)
    fig.suptitle("Out of sample, the model does no better than chance", x=0.02, ha="left",
                 color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "F03_test_results.png")
    plt.close(fig)


def coefficients():
    c = pd.read_csv(TAB_DIR / "T07_coefficients.csv")
    names = {"brent_1m": "Brent, 1-month change", "brent_3m": "Brent, 3-month change",
             "gas_1m": "European gas, 1-month change (lagged)", "hbr_mom_12_1": "HBR 12-1 month momentum",
             "hbr_vol_6m": "HBR 6-month volatility", "ftse_1m": "FTSE 250, last month",
             "gbpusd_1m": "GBP/USD, 1-month change"}
    c["label"] = c["feature"].map(names)
    c = c.sort_values("odds_ratio_per_1sd")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(c["label"], c["odds_ratio_per_1sd"] - 1, left=1, color=BLUE, height=0.55)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xlim(min(0.65, c["odds_ratio_per_1sd"].min() - 0.06), max(1.35, c["odds_ratio_per_1sd"].max() + 0.06))
    for y, v in enumerate(c["odds_ratio_per_1sd"]):
        if v >= 1:
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", ha="left", color=INK2, fontsize=8)
        else:
            ax.text(v - 0.01, y, f"{v:.2f}", va="center", ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("Odds ratio of outperforming next month, per 1 s.d. rise (1 = no effect)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What the model learned in training (2011-2021)", loc="left", color=INK, fontsize=11,
                 fontweight="bold")
    fig.savefig(FIG_DIR / "F04_coefficients.png")
    plt.close(fig)


def placebo():
    p = pd.read_csv(TAB_DIR / "T08_placebo.csv")["placebo_auc"]
    real = float(pd.read_csv(TAB_DIR / "T10_model_summary.csv", index_col=0).loc["test_auc", "value"])
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(p, bins=25, color=BLUE, edgecolor="white", linewidth=1)
    ax.axvline(real, color=ORANGE, lw=2)
    ax.text(real, ax.get_ylim()[1] * 0.95, f"  real model\n  AUC {real:.2f}", color=INK, va="top")
    ax.set_xlabel("Test AUC when the training labels are randomly shuffled (200 runs)")
    ax.set_ylabel("Runs")
    ax.grid(axis="x", visible=False)
    ax.set_title("Placebo check: the real model sits inside the luck range", loc="left",
                 color=INK, fontsize=11, fontweight="bold")
    fig.savefig(FIG_DIR / "F05_placebo.png")
    plt.close(fig)


def football_field(summary):
    order = summary.index.tolist()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, name in enumerate(order):
        low, high, base = summary.loc[name, ["low_p", "high_p", "base_p"]]
        if name == "Blended (team weights)":
            colour, alpha = "#0b0b0b", 0.25
        else:
            colour, alpha = "#2a78d6", 0.35
        ax.barh(i, high - low, left=low, height=0.5, color=colour, alpha=alpha)
        ax.plot(base, i, "|", color=colour, ms=16, mew=2.5)
        ax.text(high + 8, i, f"{base:.0f}p", va="center", fontsize=8, color="#0b0b0b")
    ax.axvline(PRICE_PENCE, color="#eb6834", lw=1.6)
    ax.text(PRICE_PENCE + 4, -0.55, f"Share price {PRICE_PENCE:.0f}p", color="#eb6834", fontsize=8, va="bottom")
    ax.set_xlim(0, summary["high_p"].max() + 60)
    ax.set_ylim(len(order) - 0.5, -0.8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("Value per share (pence)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", lw=0.6)
    methods = summary["base_p"].iloc[:-1]
    n_below = int((methods < PRICE_PENCE).sum())
    if n_below > len(methods) / 2:
        title = f"{n_below} of {len(methods)} methods value Harbour below today's share price"
    else:
        title = f"{len(methods) - n_below} of {len(methods)} methods value Harbour above today's share price"
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.savefig(FIG_DIR / "F06_football_field.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
