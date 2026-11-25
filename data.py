"""
data.py - settings, loading the raw files, and the monthly table for the model

Reads the Bloomberg, Yahoo Finance and FRED files saved in data/raw/, checks
they are exactly what we downloaded, and builds one monthly table:
  * 7 predictors that only use information public at the end of month t
  * the target: 1 if Harbour beats the FTSE 250 in month t+1, otherwise 0
Every other file gets its data from the small "read" functions in here.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# settings shared by all the other files
SEED = 42
ROOT = Path(__file__).resolve().parent
RAW, MANUAL, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "manual", ROOT / "data" / "processed"
FIG_DIR, TAB_DIR = ROOT / "figures", ROOT / "tables"
for folder in (PROCESSED, FIG_DIR, TAB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

BLOOMBERG_FILE = RAW / "bloomberg_data_filled.xlsx"
TEAM_FILE = RAW / "team_data_filled.xlsx"       # the Bloomberg data pack we made for the team
FEATURES = ["brent_1m", "brent_3m", "gas_1m", "hbr_mom_12_1", "hbr_vol_6m", "ftse_1m", "gbpusd_1m"]
TRAIN_END = "2021-12"                           # train 2011-2021, test 2022 onwards

# market data on the valuation date, 20 November 2026 (Bloomberg)
PRICE_PENCE = 250.4                             # Harbour share price
GBPUSD = 1.3084                                 # US$ per GBP 1
NOKUSD = 0.1024                                 # US$ per NOK 1
NON_VOTING_SHARES = 251.5                       # million unlisted non-voting shares (LetterOne)
NON_VOTING_PREMIUM = 1.13                       # each carries 13% more earnings and dividends (H1-26 EPS note)


def save_table(df, name, **kwargs):
    df.to_csv(TAB_DIR / f"{name}.csv", **kwargs)


# ---------------------------------------------------------------------------
# 1. Check the raw files and list where they came from
# ---------------------------------------------------------------------------
def check_raw_files():
    # make sure every raw file is exactly what we downloaded (SHA-256 hash in metadata.json)
    meta = json.loads((RAW / "metadata.json").read_text(encoding="utf-8"))
    for rec in meta["files"]:
        digest = hashlib.sha256((RAW / rec["file"]).read_bytes()).hexdigest()
        assert digest == rec["sha256"], f"{rec['file']} differs from the recorded download"
    print(f"All {len(meta['files'])} raw files verified (downloaded {meta['downloaded']})")
    sources = pd.DataFrame(meta["files"])[["file", "description", "source"]]
    save_table(sources, "T01_data_sources", index=False)


# ---------------------------------------------------------------------------
# 2. Small functions that each read one raw file
# ---------------------------------------------------------------------------
def bloomberg_monthly(sheet):
    # one monthly sheet of the Bloomberg file: date, price, total-return index
    df = pd.read_excel(BLOOMBERG_FILE, sheet_name=sheet, header=None, skiprows=4, usecols=[0, 1, 2])
    df.columns = ["date", "price", "total_return"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index = df.index.to_period("M")          # label each row by its month
    return df.astype(float)


def yahoo_monthly(name):
    # last closing price of each month from a Yahoo Finance file
    df = pd.read_csv(RAW / f"yahoo_{name}.csv", index_col="Date", parse_dates=True)
    s = df["Close"].dropna().resample("ME").last()
    s.index = s.index.to_period("M")
    return s


def european_gas():
    # IMF European gas price, monthly average, US$ per mmbtu (from FRED)
    gas = pd.read_csv(RAW / "fred_PNGASEUUSDM.csv", index_col=0, parse_dates=True).iloc[:, 0]
    gas.index = gas.index.to_period("M")
    return gas


def peer_total_returns():
    # monthly total-return indices for Harbour, the FTSE 250 and peers (team pack tab 6)
    raw = pd.read_excel(TEAM_FILE, sheet_name="6 Share performance", header=None)
    series = {}
    for col in range(0, raw.shape[1], 3):       # each company takes 3 columns
        name = raw.iat[5, col]
        if not isinstance(name, str):
            continue
        block = raw.iloc[7:, [col, col + 1]].dropna()
        block.columns = ["date", "tr"]
        months = pd.to_datetime(block["date"]).dt.to_period("M")
        series[name] = pd.Series(block["tr"].astype(float).values, index=months)
    return pd.DataFrame(series)


def financial_history():
    # Harbour's reported results FY2019-FY2025 (team pack tab 1), US$ million
    hist = pd.read_excel(TEAM_FILE, sheet_name="1 Financial history", header=None,
                         skiprows=5, usecols=range(8), nrows=15).dropna(how="all")
    hist = hist.set_index(0)
    hist.columns = hist.iloc[0]
    return hist.iloc[1:]


def consensus_forecasts():
    # analysts' average forecasts FY2026-FY2028 (team pack tab 2), US$ million
    fc = pd.read_excel(TEAM_FILE, sheet_name="2 Forecasts", header=None,
                       skiprows=5, usecols=range(4), nrows=11).set_index(0)
    fc.columns = fc.iloc[0]
    fc = fc.iloc[1:]
    fc.columns = [f"{c}e" for c in fc.columns]  # "e" = estimate
    return fc


def discount_rate_inputs():
    # CAPM and cost-of-capital inputs on 20 Nov 2026 (team pack tab 3), looked up by their label
    tab = pd.read_excel(TEAM_FILE, sheet_name="3 Discount rate", header=None, usecols=[0, 1]).dropna()
    values = dict(zip(tab[0].astype(str).str.strip(), tab[1]))
    pick = {"gilt_10y_%": "UK 10-year government bond yield (%)",
            "beta_adjusted": "Beta vs FTSE All-Share, adjusted (2y weekly)",
            "equity_premium_%": "Equity risk premium (%) - ASSUMPTION",
            "bbg_cost_of_equity_%": "Bloomberg cost of equity (%)"}
    return {key: float(values[label]) for key, label in pick.items()}


def peer_table():
    # peer company data (team pack tab 5), one row per company
    peers = pd.read_excel(TEAM_FILE, sheet_name="5 Comparables", header=5, nrows=14)
    peers.columns = [str(c).strip() for c in peers.columns]
    return peers


def futures_curve():
    # Futures prices on 20 Nov 2026 (Bloomberg), one row per DELIVERY month:
    # Brent (US$/bbl) and UK NBP gas (pence per therm).
    # The two strips start in different months: the front Brent contract is January 2027
    # (COF7) but the front NBP contract is December 2026 (FNZ6). So each price is placed on
    # the month its contract code names, not on its row number.
    fut = pd.read_excel(TEAM_FILE, sheet_name="4 Oil & gas prices", header=None)
    rows = fut.iloc[6:42, [1, 2, 3, 4]]
    rows.columns = ["brent_code", "brent", "nbp_code", "nbp_pence"]
    brent = pd.Series(rows["brent"].astype(float).values, index=[contract_month(c) for c in rows["brent_code"]])
    nbp = pd.Series(rows["nbp_pence"].astype(float).values, index=[contract_month(c) for c in rows["nbp_code"]])
    curve = pd.concat([brent.rename("brent"), nbp.rename("nbp_pence")], axis=1).sort_index()
    return curve


def contract_month(code):
    # Bloomberg futures codes end in a month letter and the year, e.g. COF7 = Jan 2027,
    # COG29 = Feb 2029, FNZ6 = Dec 2026
    letters = "FGHJKMNQUVXZ"
    code = str(code).strip()
    tail = code[2:]
    for pos in range(len(tail) - 1, -1, -1):
        if tail[pos] in letters and tail[pos + 1:].isdigit():
            month = letters.index(tail[pos]) + 1
            digits = tail[pos + 1:]
            year = 2020 + int(digits) if len(digits) == 1 else 2000 + int(digits)
            return pd.Period(year=year, month=month, freq="M")
    raise ValueError(f"cannot read the delivery month of futures contract {code}")


def precedent_deals():
    # North Sea takeovers, typed in from the press releases (links in the file)
    return pd.read_csv(MANUAL / "precedent_deals.csv")


# ---------------------------------------------------------------------------
# 3. The monthly table for the model
# ---------------------------------------------------------------------------
def check_premier_join(hbr):
    # Before April 2021 the Bloomberg series is Premier Oil, Harbour's legal predecessor.
    # Check it matches Yahoo's Harbour prices after the merger (no jump, no gap).
    yahoo = yahoo_monthly("HBR")
    bbg = bloomberg_monthly("HBR_monthly")["price"]
    both = pd.concat([bbg, yahoo], axis=1, keys=["bloomberg", "yahoo"]).dropna()
    both = both[both.index >= "2021-05"]
    gap = (both["bloomberg"] / both["yahoo"] - 1).abs().max()
    print(f"Bloomberg vs Yahoo Harbour prices since May 2021: largest gap {gap:.2%}")


def build_model_table():
    # 1. Raw series. Total-return indices include reinvested dividends.
    hbr = bloomberg_monthly("HBR_monthly")["total_return"]
    ftse = bloomberg_monthly("FTSE250_monthly")["total_return"]
    brent = yahoo_monthly("BRENT_FUT")          # US$ per barrel
    gbpusd = yahoo_monthly("GBPUSD")            # US$ per GBP
    gas = european_gas().reindex(hbr.index)     # line up on the same months
    check_premier_join(hbr)

    # 2. Monthly returns (log returns add up neatly over time)
    r_hbr = np.log(hbr).diff()
    r_ftse = np.log(ftse).diff()

    # 3. Predictors, all known at the end of month t
    X = pd.DataFrame(index=hbr.index)
    X["brent_1m"] = np.log(brent).diff(1)       # oil price change over the last month
    X["brent_3m"] = np.log(brent).diff(3)       # oil price change over the last 3 months
    # the gas price is a monthly average published after the month ends,
    # so we use LAST month's change to avoid peeking at the future
    X["gas_1m"] = np.log(gas).diff(1).shift(1)
    # momentum: Harbour's return from 12 months ago to 1 month ago (Jegadeesh and Titman, 1993)
    X["hbr_mom_12_1"] = np.log(hbr).shift(1) - np.log(hbr).shift(12)
    X["hbr_vol_6m"] = r_hbr.rolling(6).std()    # how jumpy the shares were over 6 months
    X["ftse_1m"] = r_ftse                       # market mood this month
    X["gbpusd_1m"] = np.log(gbpusd).diff(1)     # Harbour earns dollars, shares are priced in pence

    # 4. Target: did Harbour beat the FTSE 250 NEXT month?
    excess_next = (r_hbr - r_ftse).shift(-1)    # month t+1's result, stored on row t
    table = X.copy()
    table["excess_next"] = excess_next
    table["y"] = (excess_next > 0).astype(float)
    table.loc[excess_next.isna(), "y"] = np.nan # newest month: the future is not known yet
    table["r_hbr_next"] = r_hbr.shift(-1)
    table["r_ftse_next"] = r_ftse.shift(-1)

    # 5. Keep months with every predictor; keep the newest month aside for the live signal
    table = table.dropna(subset=FEATURES)
    labelled = table.dropna(subset=["y"])
    live = table[table["y"].isna()].tail(1)

    # target check: y on row t really is month t+1's result
    assert (labelled["y"] == (labelled["r_hbr_next"] > labelled["r_ftse_next"])).all()

    labelled.to_csv(PROCESSED / "monthly_dataset.csv", index_label="month")
    live.to_csv(PROCESSED / "live_row.csv", index_label="month")
    print(f"Model table: {len(labelled)} labelled months, {labelled.index.min()} to "
          f"{labelled.index.max()}; Harbour beat the FTSE 250 in {labelled['y'].mean():.1%} of them")
    print(f"Live row: {live.index[-1]} (used to predict the following month)")
