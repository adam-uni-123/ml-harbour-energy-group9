"""
download_fred.py  -  FRED macro series only, with a time limit.

Use this if step 4 of download_data.py hangs. Each request gives up after
30 seconds and is retried twice, so the whole script finishes in a few
minutes at worst. Files are saved to data/raw/ just like the main script.

Run:   python download_fred.py
"""

import io
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

START = "2010-01-01"
OUT = Path(__file__).resolve().parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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

TIMEOUT = 30     # seconds before giving up on one request
TRIES = 3        # first attempt + 2 retries

log_rows = []
for series, desc in FRED.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    for attempt in range(1, TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                text = r.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(text))
            df.columns = ["Date", series]
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df[series] = pd.to_numeric(df[series], errors="coerce")   # "." -> blank
            df = df[df.index >= START]
            df.to_csv(OUT / f"fred_{series}.csv")
            log_rows.append({"file": f"fred_{series}.csv", "source": "FRED",
                             "ticker_or_series": series,
                             "first_date": df.index.min().date(),
                             "last_date": df.index.max().date(), "rows": len(df),
                             "downloaded_at": STAMP, "note": desc})
            print(f"  OK  {series:18s} {len(df):5d} rows")
            break
        except Exception as e:
            print(f"  try {attempt}/{TRIES} failed for {series}: {e}")
            if attempt == TRIES:
                log_rows.append({"file": f"fred_{series}.csv", "source": "FRED",
                                 "ticker_or_series": series, "rows": 0,
                                 "downloaded_at": STAMP, "note": f"FAILED: {e}"})
            time.sleep(3)

pd.DataFrame(log_rows).to_csv(OUT / "fred_log.csv", index=False)
ok = sum(1 for r in log_rows if r["rows"])
print(f"\nSaved {ok} of {len(FRED)} FRED series to {OUT}")
if ok < len(FRED):
    print("Some failed. Try opening this link in your browser:")
    print("  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU")
    print("If the browser can't open it either, your network is blocking FRED.")

# ---------------------------------------------------------------------------
# Section 5 of download_data.py (it did not run if you stopped that script):
# rebuild the download log from every file in data/raw/, record package
# versions, and run the Harbour price health check.
# ---------------------------------------------------------------------------
import sys
import yfinance as yf

print("\n=== Section 5: download log and health check ===")
rows = []
for f in sorted(OUT.glob("*.csv")):
    if f.name in ("download_log.csv", "fred_log.csv"):
        continue
    try:
        d = pd.read_csv(f, index_col=0)
        d.index = pd.to_datetime(d.index, errors="coerce", utc=True)
        dates = d.index.dropna()
        first = dates.min().date() if len(dates) else ""
        last = dates.max().date() if len(dates) else ""
        source = "FRED" if f.name.startswith("fred_") else "Yahoo Finance"
        rows.append({"file": f.name, "source": source, "first_date": first,
                     "last_date": last, "rows": len(d),
                     "file_saved_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    except Exception as e:
        rows.append({"file": f.name, "note": f"could not read: {e}"})
pd.DataFrame(rows).to_csv(OUT / "download_log.csv", index=False)
print(f"  download_log.csv lists {len(rows)} files")

with open(OUT / "environment.txt", "w") as fh:
    fh.write(f"log_built_at: {STAMP}\n")
    fh.write(f"python: {sys.version.split()[0]}\n")
    fh.write(f"pandas: {pd.__version__}\n")
    fh.write(f"yfinance: {yf.__version__}\n")

hbr_file = OUT / "yahoo_HBR.csv"
if hbr_file.exists():
    hbr = pd.read_csv(hbr_file, index_col="Date", parse_dates=True)
    print(f"  HBR.L history starts {hbr.index.min().date()} and ends {hbr.index.max().date()}")
    moves = hbr["Adj Close"].pct_change().abs().sort_values(ascending=False).head(5)
    print("  Five largest daily moves (check none is a data error):")
    for d, m in moves.items():
        print(f"    {d.date()}  {m:.1%}")
else:
    print("  yahoo_HBR.csv not found - check the first script saved it")
print("\nNext: upload everything in data/raw/ to the project.")
