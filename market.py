"""
market.py - the financial and market evidence

  * Share-price record: return, volatility, worst fall and beta for Harbour,
    the FTSE 250 and peers over 1, 3 and 5 years
  * Harbour's results FY2019-FY2025 plus analysts' forecasts FY2026-FY2028
  * Trading comparables: what peers are worth as a multiple of their earnings
"""

import numpy as np
import pandas as pd

from data import GBPUSD, NOKUSD, PRICE_PENCE, NON_VOTING_SHARES, NON_VOTING_PREMIUM, save_table, peer_total_returns, financial_history, \
    consensus_forecasts, peer_table

# close North Sea peers. Diversified (US gas) and Tullow (West Africa) are left out;
# Shell, BP and Equinor appear only for context.
PEERS = ["EnQuest", "Serica Energy", "Ithaca Energy", "Aker BP", "Var Energi", "DNO", "OKEA", "BlueNord"]


def risk_return(tr, months):
    # annual return, annual volatility, worst fall and beta vs the FTSE 250 over the last N months
    r = np.log(tr).diff().iloc[-months:]
    bench = r["FTSE 250 (benchmark)"]
    rows = []
    for name in tr.columns:
        x = r[name].dropna()
        if len(x) < months * 0.9:               # skip companies without enough history
            continue
        level = np.exp(x.cumsum())
        rows.append({"company": name, "window_years": months // 12,
                     "ann_return_%": 100 * np.expm1(x.mean() * 12),
                     "ann_volatility_%": 100 * x.std() * np.sqrt(12),
                     "max_drawdown_%": 100 * (level / level.cummax() - 1).min(),
                     "beta_vs_FTSE250": x.cov(bench.loc[x.index]) / bench.loc[x.index].var()})
    return pd.DataFrame(rows)


def share_record():
    tr = peer_total_returns()
    table = pd.concat([risk_return(tr, 12), risk_return(tr, 36), risk_return(tr, 60)])
    save_table(table.round(2), "T02_risk_return", index=False)
    harbour_and_index = table[table["company"].isin(["Harbour Energy", "FTSE 250 (benchmark)"])]
    print(harbour_and_index.round(2).to_string(index=False))
    return tr


def financial_summary():
    hist = financial_history()
    fc = consensus_forecasts().rename(index={"Net debt (year end)": "Net debt"})
    keep = ["Revenue", "EBITDA", "Operating profit (EBIT)", "Net profit",
            "Capital spending (capex)", "Free cash flow", "Net debt"]
    out = pd.concat([hist.loc[keep], fc.reindex(keep)], axis=1).astype(float)
    out.loc["EBITDA margin %"] = 100 * out.loc["EBITDA"] / out.loc["Revenue"]
    out.loc["Net debt / EBITDA (x)"] = out.loc["Net debt"] / out.loc["EBITDA"]
    save_table(out.round(1), "T03_financial_summary")


def comparables():
    c = peer_table()
    # Bloomberg gave market values in local currency, so convert them to US$
    fx = c["Currency of share price"].map({"GBp": GBPUSD, "NOK": NOKUSD})

    def num(col):
        return pd.to_numeric(c[col], errors="coerce")

    out = pd.DataFrame({"company": c["Company"]})
    out["mkt_cap_usd_m"] = num("Market value of equity (US$ m)") * fx
    out["net_debt_usd_m"] = num("Net debt (US$ m)")
    out["ev_usd_m"] = num("Enterprise value EV (US$ m)")
    # Bloomberg counts only Harbour's 1,492m listed voting shares. Add the 251.5m unlisted
    # non-voting shares (worth 1.13 voting shares each) so Harbour's own multiples use the same
    # share count as our valuation.
    hbr = out["company"] == "Harbour Energy"
    extra = NON_VOTING_SHARES * NON_VOTING_PREMIUM * PRICE_PENCE / 100 * GBPUSD
    out.loc[hbr, "mkt_cap_usd_m"] += extra
    out.loc[hbr, "ev_usd_m"] += extra
    out["ev_ebitda_ltm"] = out["ev_usd_m"] / num("EBITDA last year (US$ m)")
    out["ev_ebitda_2026e"] = out["ev_usd_m"] / num("EBITDA 2026 forecast (US$ m)")
    out["ev_ebitda_2027e"] = out["ev_usd_m"] / num("EBITDA 2027 forecast (US$ m)")
    out["p_cf_ltm"] = out["mkt_cap_usd_m"] / num("Operating cash flow last year (US$ m)")
    out["net_debt_ebitda"] = out["net_debt_usd_m"] / num("EBITDA last year (US$ m)")
    out["div_yield_%"] = num("Dividend yield %")
    out["target_upside_%"] = 100 * (num("Analyst target price (local)") / num("Share price (local)") - 1)

    median = out[out["company"].isin(PEERS)].median(numeric_only=True)
    median["company"] = "Median, North Sea peers"
    out = pd.concat([out, median.to_frame().T], ignore_index=True)
    save_table(out.round(2), "T12_comparables", index=False)
    return out
