"""
download_data.py  -  FI4008 Group Assessment: Harbour Energy plc (LSE: HBR)

WHAT THIS DOES (plain English)
    Downloads every free dataset the project needs and saves each one as a CSV
    file in the folder  data/raw/  next to this script. After this, all analysis
    reads only these saved files, so the results can be reproduced exactly even
    if the websites change their data later.

HOW TO RUN (on your own laptop, with internet)
    1. Open a terminal (Anaconda Prompt on Windows) in the folder holding this file.
    2. Install the two packages once:
           python -m pip install --upgrade yfinance pandas
    3. Run:
           python download_data.py
    4. It takes 2-5 minutes. Upload everything in  data/raw/  to the project
       (CSV and JSON files, plus download_log.csv).

    If a single ticker fails, the script prints a warning and carries on.
    That is expected for a few (for example PMO.L, the old Premier Oil ticker).

SOURCES
    Yahoo Finance via the yfinance package (share prices, index, peers, Brent
    futures, TTF gas futures, GBP/USD, company financial statements).
    FRED, Federal Reserve Bank of St. Louis (Brent spot, European gas price,
    GBP/USD, UK interest rates). FRED CSV downloads need no API key.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
START = "2010-01-01"          # ML sample starts 2014; earlier data allows 12-month warm-up and charts
END = None                    # None = up to today
OUT = Path(__file__).resolve().parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

# Yahoo Finance tickers: name -> (ticker, why we need it)
YAHOO = {
    # The company and its predecessor
    "HBR":    ("HBR.L",     "Harbour Energy plc, the company we value (pence)"),
    "PMO":    ("PMO.L",     "Premier Oil plc, legal predecessor before the 2021 reverse takeover (may be empty)"),
    # Benchmarks
    "FTSE250": ("^FTMC",    "FTSE 250 index, the benchmark for the ML target"),
    "FTSE100": ("^FTSE",    "FTSE 100 index, context"),
    "FTSEALL": ("^FTAS",    "FTSE All-Share index, for beta in the discount rate"),
    # UK-listed North Sea peers
    "ENQ":    ("ENQ.L",     "EnQuest, UK North Sea producer"),
    "SQZ":    ("SQZ.L",     "Serica Energy, UK North Sea producer"),
    "ITH":    ("ITH.L",     "Ithaca Energy, UK North Sea producer"),
    "DEC":    ("DEC.L",     "Diversified Energy, gas producer"),
    "TLW":    ("TLW.L",     "Tullow Oil, UK-listed E&P"),
    # Norwegian peers (Harbour has large Norway assets after Wintershall Dea)
    "AKRBP":  ("AKRBP.OL",  "Aker BP, Norwegian producer"),
    "VAR":    ("VAR.OL",    "Var Energi, Norwegian producer"),
    "DNO":    ("DNO.OL",    "DNO, Norwegian-listed E&P"),
    # Large integrated majors for context
    "SHEL":   ("SHEL.L",    "Shell plc, context"),
    "BP":     ("BP.L",      "BP plc, context"),
    # Commodities and currency
    "BRENT_FUT": ("BZ=F",   "Brent crude front-month future (USD/bbl), cross-check for FRED"),
    "TTF_FUT":   ("TTF=F",  "Dutch TTF gas future (EUR/MWh), proxy for European/NBP gas"),
    "GBPUSD":    ("GBPUSD=X", "GBP/USD exchange rate, cross-check for FRED"),
}

# Companies whose financial statements and profile we also save (for comps)
FUNDAMENTALS = ["HBR.L", "ENQ.L", "SQZ.L", "ITH.L", "DEC.L", "TLW.L",
                "AKRBP.OL", "VAR.OL", "DNO.OL", "SHEL.L", "BP.L"]

# FRED series: id -> description
FRED = {
    "DCOILBRENTEU":     "Brent crude spot price, USD per barrel, daily (EIA)",
    "POILBREUSDM":      "Brent crude price, USD per barrel, monthly average (IMF)",
    "PNGASEUUSDM":      "European natural gas price, USD per MMBtu, monthly (IMF)",
    "DEXUSUK":          "US dollars per British pound, daily (Federal Reserve H.10)",
    "IRLTLT01GBM156N":  "UK 10-year government bond yield, monthly (OECD)",
    "IR3TIB01GBM156N":  "UK 3-month interbank rate, monthly (OECD)",
    "IRSTCB01GBM156N":  "UK central bank (Bank Rate) policy rate, monthly (OECD)",
    "GBRCPIALLMINMEI":  "UK consumer price index, monthly (OECD)",
}

log_rows = []   # one row per file saved, written to download_log.csv at the end
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def record(file_name, source, code, df, note=""):
    """Add one line to the download log describing what was saved."""
    if isinstance(df, pd.DataFrame) and len(df) and isinstance(df.index, pd.DatetimeIndex):
        first, last = df.index.min().date(), df.index.max().date()
    else:
        first = last = ""
    log_rows.append({"file": file_name, "source": source, "ticker_or_series": code,
                     "first_date": first, "last_date": last,
                     "rows": len(df) if df is not None else 0,
                     "downloaded_at": STAMP, "note": note})


# ---------------------------------------------------------------------------
# 1. Yahoo Finance daily prices (raw close AND dividend-adjusted close)
# ---------------------------------------------------------------------------
print("\n=== 1. Yahoo Finance daily prices ===")
for name, (ticker, why) in YAHOO.items():
    try:
        # auto_adjust=False keeps both 'Close' (actual price) and 'Adj Close'
        # (price adjusted for dividends and splits, used for total returns).
        df = yf.download(ticker, start=START, end=END, auto_adjust=False,
                         actions=True, progress=False, multi_level_index=False)
        if df is None or df.empty:
            print(f"  WARNING {ticker}: no data returned")
            record(f"yahoo_{name}.csv", "Yahoo Finance", ticker, None, "EMPTY - " + why)
            continue
        df.index.name = "Date"
        df.to_csv(OUT / f"yahoo_{name}.csv")
        record(f"yahoo_{name}.csv", "Yahoo Finance", ticker, df, why)
        print(f"  OK  {ticker:10s} {len(df):5d} rows  {df.index.min().date()} to {df.index.max().date()}")
    except Exception as e:                      # keep going if one ticker fails
        print(f"  WARNING {ticker}: {e}")
        record(f"yahoo_{name}.csv", "Yahoo Finance", ticker, None, f"FAILED: {e}")
    time.sleep(1)                               # be polite to Yahoo's servers

# ---------------------------------------------------------------------------
# 2. Dividends and share splits/consolidations for Harbour and Premier
#    (Premier did a 1-for-20 share consolidation at the 2021 merger, so we
#     need to see exactly how Yahoo has adjusted the history.)
# ---------------------------------------------------------------------------
print("\n=== 2. Dividends and splits ===")
for ticker in ["HBR.L", "PMO.L"]:
    try:
        t = yf.Ticker(ticker)
        acts = t.actions                         # columns: Dividends, Stock Splits
        acts.index.name = "Date"
        fname = f"yahoo_actions_{ticker.replace('.', '_')}.csv"
        acts.to_csv(OUT / fname)
        record(fname, "Yahoo Finance", ticker, acts, "dividends and splits")
        print(f"  OK  {ticker}: {len(acts)} dividend/split events")
    except Exception as e:
        print(f"  WARNING {ticker} actions: {e}")

# ---------------------------------------------------------------------------
# 3. Company financial statements and profiles (for comps and cross-checks)
#    Yahoo keeps only about 4 years of annual statements. We still take the
#    main numbers from the annual reports; this is a cross-check.
# ---------------------------------------------------------------------------
print("\n=== 3. Financial statements and profiles ===")
for ticker in FUNDAMENTALS:
    tag = ticker.replace(".", "_")
    t = yf.Ticker(ticker)
    tables = {"income": t.income_stmt, "balance": t.balance_sheet, "cashflow": t.cashflow}
    for label, table in tables.items():
        try:
            if table is not None and not table.empty:
                fname = f"fin_{tag}_{label}.csv"
                table.to_csv(OUT / fname)
                record(fname, "Yahoo Finance", ticker, table, f"annual {label} statement")
        except Exception as e:
            print(f"  WARNING {ticker} {label}: {e}")
    try:
        info = t.info                            # market cap, shares, EV, beta, etc.
        with open(OUT / f"info_{tag}.json", "w") as f:
            json.dump(info, f, indent=2, default=str)
        record(f"info_{tag}.json", "Yahoo Finance", ticker, None, "company profile snapshot")
    except Exception as e:
        print(f"  WARNING {ticker} info: {e}")
    print(f"  done {ticker}")
    time.sleep(1)

# Analyst price targets for Harbour (context for our own target price)
try:
    targets = yf.Ticker("HBR.L").analyst_price_targets
    with open(OUT / "info_HBR_analyst_targets.json", "w") as f:
        json.dump(targets, f, indent=2, default=str)
    record("info_HBR_analyst_targets.json", "Yahoo Finance", "HBR.L", None, "analyst price targets")
except Exception as e:
    print(f"  WARNING analyst targets: {e}")

# ---------------------------------------------------------------------------
# 4. FRED macro series (no API key needed for the CSV download)
# ---------------------------------------------------------------------------
print("\n=== 4. FRED macro series ===")
for series, desc in FRED.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    try:
        df = pd.read_csv(url, storage_options={"User-Agent": "Mozilla/5.0"})
        df.columns = ["Date", series]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        # FRED marks missing days with "." -> convert to numbers, blanks become NaN
        df[series] = pd.to_numeric(df[series], errors="coerce")
        df = df[df.index >= START]
        df.to_csv(OUT / f"fred_{series}.csv")
        record(f"fred_{series}.csv", "FRED (Federal Reserve Bank of St. Louis)", series, df, desc)
        print(f"  OK  {series:18s} {len(df):5d} rows  {df.index.min().date()} to {df.index.max().date()}")
    except Exception as e:
        print(f"  WARNING {series}: {e}")
        record(f"fred_{series}.csv", "FRED", series, None, f"FAILED: {e}")

# ---------------------------------------------------------------------------
# 5. Write the download log and a quick health check
# ---------------------------------------------------------------------------
log = pd.DataFrame(log_rows)
log.to_csv(OUT / "download_log.csv", index=False)

with open(OUT / "environment.txt", "w") as f:
    f.write(f"downloaded_at: {STAMP}\n")
    f.write(f"python: {sys.version.split()[0]}\n")
    f.write(f"pandas: {pd.__version__}\n")
    f.write(f"yfinance: {yf.__version__}\n")

print("\n=== 5. Health check: Harbour price history ===")
hbr_file = OUT / "yahoo_HBR.csv"
if hbr_file.exists():
    hbr = pd.read_csv(hbr_file, index_col="Date", parse_dates=True)
    print(f"  HBR.L history starts {hbr.index.min().date()} and ends {hbr.index.max().date()}")
    # The five biggest daily moves: a jump near March/April 2021 would show
    # the Premier Oil to Harbour join is not adjusted properly.
    moves = hbr["Adj Close"].pct_change().abs().sort_values(ascending=False).head(5)
    print("  Five largest daily moves (check none is a data error):")
    for d, m in moves.items():
        print(f"    {d.date()}  {m:.1%}")

failed = log[log["note"].astype(str).str.startswith(("EMPTY", "FAILED"))]
print(f"\nSaved {len(log) - len(failed)} files to {OUT}")
if len(failed):
    print("These did not download (fine to report, I will work around them):")
    print(failed[["ticker_or_series", "note"]].to_string(index=False))
print("\nNext: upload everything in data/raw/ to the project.")
