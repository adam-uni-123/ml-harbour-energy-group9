"""
dcf.py - discounted cash flow (DCF / NAV) valuation of Harbour Energy

The idea: estimate the cash Harbour's oil and gas fields will produce each year
from 2027 to 2045, discount it back to today at 9% a year, then subtract what
the company owes (debt, hybrid bonds, leases, clean-up costs). What is left,
divided by the number of shares, is the value of one share.

All money is US$ million unless the name says otherwise. Every assumption is in
the block below with its source:
  AR25 = Annual Report 2025, H126 = half-year results 2026,
  TU = trading update 5 Nov 2026, BBG = Bloomberg 20 Nov 2026
"""

import pandas as pd

from data import (RAW, GBPUSD, PRICE_PENCE, NON_VOTING_SHARES, NON_VOTING_PREMIUM, futures_curve,
                  discount_rate_inputs, save_table)

# --- shares and what the company owes ----------------------------------------
VOTING_SHARES = 1491.6          # million, BBG (after the 2026 buybacks)
SHARES = VOTING_SHARES + NON_VOTING_SHARES * NON_VOTING_PREMIUM   # c.1,776m voting-share equivalents
NET_DEBT = 5100                 # TU: net debt at 30 Sep 2026
HYBRIDS = 2044                  # H126: hybrid (subordinated) notes, treated like debt
LEASES = 569 + 286              # H126: lease liabilities
DECOM_PROVISION = 7423          # H126: cost of removing old platforms, in today's money
DECOM_TAX_RELIEF = 0.50         # UK relief c.40%, Norway 78%, so about half overall
DECOM_BOOK_RATE = 0.04          # AR25: Harbour discounts this bill at 3.1-4.8% ...
DECOM_YEARS = 15                # ... and "around half" is spent 10-20 years from now

# --- production -------------------------------------------------------------
PLATEAU_KBOEPD = 500            # 500-505 kboe/d in 2026, "sustained through the end of
PLATEAU_END = 2030              #  the decade" (AR25 p27, TU)
DECLINE = 0.12                  # production then falls 12% a year
LAST_YEAR = 2045
MIX = {"liquids": 0.40, "euro_gas": 0.40, "other_gas": 0.20}   # TU: c.40/40/20 split

# --- prices Harbour received in Jan-Sep 2026, before hedging (TU) ----------
REAL_OIL_9M = 92.0              # US$ per barrel of oil
LIQ_TO_OIL = 81 / 85            # oil + NGLs sell for about 95% of the oil price
REAL_GAS_9M = 17.1              # European gas, US$ per thousand cubic feet (mscf)
OTHER_GAS = 3.8                 # Argentina, North Africa etc., US$ per mscf
MSCF_PER_BOE = 6.0              # 6 mscf of gas = 1 barrel of oil equivalent (boe)

# --- costs, investment and tax ------------------------------------------------
CASH_COST_PER_BOE = 21.9        # TU 9M 2026: (revenue 9.8bn - EBITDAX 6.8bn) / 136.8 mmboe
CAPEX_PER_BOE = 11.0            # about $2.0bn a year (BBG consensus) / 182.5 mmboe
INFLATION = 0.02                # costs and long-run prices rise 2% a year
TAX_TO_2029 = 0.62              # cash tax on (EBITDAX - capex) while the Energy Profits Levy applies
TAX_FROM_2030 = 0.52            # after the Levy ends in 2030 (AR25)

# --- hedges already agreed (TU appendix): year: (volume, fixed price) --------
GAS_HEDGES = {2027: (21, 14.9), 2028: (8, 11.0)}   # mmboe, US$ per mscf
OIL_HEDGES = {2027: (18, 81.0), 2028: (8, 75.0)}   # million barrels, US$ per barrel

# --- discounting ---------------------------------------------------------------
WACC = 0.09                     # our choice; discount_rate_check() prints the evidence for it
CREDIT_SPREAD = 0.015           # assumption: spread over gilts for a BBB- (investment-grade) borrower
DEBT_TAX_RELIEF = 0.25          # interest gets only partial relief (none against the Energy Profits Levy);
                                # the same 25% is used for the LBO in valuation.py
VALUATION_YEAR = 2026.9         # 20 Nov 2026; each year's cash arrives mid-year
HARBOUR_LONG_TERM = (75.0, 11.0)  # H126 impairment note: Harbour's own long-term Brent ($/bbl) and
HARBOUR_LT_FROM = 2029            # European gas ($/mmbtu) from 2029, then +2.5% a year.
HARBOUR_LT_INFLATION = 0.025      # Used in one scenario only.


def price_deck(long_run="futures"):
    # Brent (US$/bbl) and UK gas (US$/mmbtu) for each calendar year:
    # the average futures price of that year's delivery months for 2027-2029, then +2% a year.
    # Gas is seasonal (dear in winter), so every year must hold a full 12 months. The NBP
    # strip ends with November 2029, so 2029 gas uses the last 12 contracts (Dec-28 to Nov-29).
    # long_run="harbour": Harbour's own long-term prices from 2029 instead (H126).
    curve = futures_curve()
    yearly = curve.groupby(curve.index.year).mean()
    gas_2029 = curve["nbp_pence"].dropna().iloc[-12:].mean()
    years = range(2027, LAST_YEAR + 1)
    deck = pd.DataFrame(index=years, columns=["brent", "gas_mmbtu"], dtype=float)
    for year in [2027, 2028, 2029]:
        nbp = gas_2029 if year == 2029 else yearly.loc[year, "nbp_pence"]
        deck.loc[year, "brent"] = yearly.loc[year, "brent"]
        deck.loc[year, "gas_mmbtu"] = nbp / 100 * GBPUSD * 10   # pence per therm -> US$ per mmbtu (10 therms)
    for year in range(2030, LAST_YEAR + 1):
        deck.loc[year] = deck.loc[year - 1] * (1 + INFLATION)
    if long_run == "harbour":
        for year in range(HARBOUR_LT_FROM, LAST_YEAR + 1):
            deck.loc[year] = [p * (1 + HARBOUR_LT_INFLATION) ** (year - HARBOUR_LT_FROM)
                              for p in HARBOUR_LONG_TERM]
    return deck


def realisation_ratios():
    # how Harbour's realised prices compared with market prices in Jan-Sep 2026
    brent = pd.read_csv(RAW / "yahoo_BRENT_FUT.csv", parse_dates=["Date"]).set_index("Date")["Close"]
    gas = pd.read_csv(RAW / "fred_PNGASEUUSDM.csv", parse_dates=[0], index_col=0).iloc[:, 0]
    brent_9m = brent["2026-01":"2026-09"].mean()
    gas_9m = gas["2026-01":"2026-09"].mean()
    return REAL_OIL_9M * LIQ_TO_OIL / brent_9m, REAL_GAS_9M / gas_9m, brent_9m, gas_9m


def cash_flows(wacc=WACC, price_shift=0.0, long_run="futures", plateau_end=PLATEAU_END, decline=DECLINE):
    # year-by-year free cash flow of the whole company (before interest), 2027-2045
    deck = price_deck(long_run) * (1 + price_shift)
    r_liq, r_gas, _, _ = realisation_ratios()
    rows = []
    for year in deck.index:
        if year <= plateau_end:
            kboepd = PLATEAU_KBOEPD
        else:
            kboepd = PLATEAU_KBOEPD * (1 - decline) ** (year - plateau_end)
        mmboe = kboepd * 365 / 1000                          # million barrels in the year
        inflation = (1 + INFLATION) ** (year - 2026)
        brent = deck.loc[year, "brent"]
        gas = deck.loc[year, "gas_mmbtu"]

        revenue_per_boe = (MIX["liquids"] * brent * r_liq
                           + MIX["euro_gas"] * gas * r_gas * MSCF_PER_BOE
                           + MIX["other_gas"] * OTHER_GAS * inflation * MSCF_PER_BOE)
        revenue = mmboe * revenue_per_boe

        # hedges: on hedged volumes Harbour gets the fixed price instead of the market price
        gas_vol, gas_price = GAS_HEDGES.get(year, (0, 0))
        oil_vol, oil_price = OIL_HEDGES.get(year, (0, 0))
        hedging = (gas_vol * MSCF_PER_BOE * (gas_price - gas * r_gas)
                   + oil_vol * (oil_price - brent * r_liq / LIQ_TO_OIL))

        ebitdax = revenue + hedging - mmboe * CASH_COST_PER_BOE * inflation
        capex = mmboe * CAPEX_PER_BOE * inflation
        if year < 2030:
            tax_rate = TAX_TO_2029
        else:
            tax_rate = TAX_FROM_2030
        tax = max(0.0, tax_rate * (ebitdax - capex))
        fcf = ebitdax - capex - tax

        years_away = year + 0.5 - VALUATION_YEAR              # 2027's cash is ~0.6 years away
        discount_factor = 1 / (1 + wacc) ** years_away
        rows.append({"year": year, "production_kboepd": kboepd, "brent": brent, "gas_usd_mmbtu": gas,
                     "revenue": revenue, "hedging": hedging, "ebitdax": ebitdax, "capex": capex,
                     "tax": tax, "fcf": fcf, "discount_factor": discount_factor,
                     "pv_fcf": fcf * discount_factor})
    return pd.DataFrame(rows).set_index("year")


def discount_rate_check():
    # Why 9%? Compare a CAPM-based WACC with Bloomberg's own cost of equity for Harbour
    k = discount_rate_inputs()
    cost_of_equity = (k["gilt_10y_%"] + k["beta_adjusted"] * k["equity_premium_%"]) / 100
    cost_of_debt = (k["gilt_10y_%"] / 100 + CREDIT_SPREAD) * (1 - DEBT_TAX_RELIEF)
    equity = PRICE_PENCE / 100 * SHARES * GBPUSD         # market value of the shares, US$m
    debt = NET_DEBT + HYBRIDS
    wacc = (equity * cost_of_equity + debt * cost_of_debt) / (equity + debt)
    print(f"Discount rate: CAPM cost of equity {cost_of_equity:.2%} (gilt {k['gilt_10y_%']:.2f}% + "
          f"beta {k['beta_adjusted']:.2f} x {k['equity_premium_%']:.0f}% premium); after-tax cost of debt "
          f"{cost_of_debt:.2%}; equity {equity / (equity + debt):.0%} of capital -> WACC {wacc:.1%}. "
          f"Bloomberg cost of equity {k['bbg_cost_of_equity_%']:.1f}%. We use {WACC:.0%}.")
    return wacc


def decom_after_tax(wacc=WACC):
    # the clean-up bill, re-discounted at our rate and after tax relief
    rediscount = ((1 + DECOM_BOOK_RATE) / (1 + wacc)) ** DECOM_YEARS
    return DECOM_PROVISION * rediscount * (1 - DECOM_TAX_RELIEF)


def pence_per_share(enterprise_value, wacc=WACC):
    # enterprise value minus everything owed = equity; divide by shares, convert to pence
    owed = NET_DEBT + HYBRIDS + LEASES + decom_after_tax(wacc)
    equity = enterprise_value - owed
    return 100 * equity / SHARES / GBPUSD


def value(wacc=WACC, **scenario):
    # value per share in pence for one set of assumptions
    return pence_per_share(cash_flows(wacc, **scenario)["pv_fcf"].sum(), wacc)


def run():
    cf = cash_flows()
    save_table(cf.round(1), "T14_dcf_cash_flows")
    ev = cf["pv_fcf"].sum()
    base, low, high = value(), value(0.11), value(0.07)
    discount_rate_check()
    r_liq, r_gas, b9, g9 = realisation_ratios()
    print(f"Realised vs market prices, Jan-Sep 2026: liquids {r_liq:.2f} x Brent (avg ${b9:.1f}), "
          f"gas {r_gas:.2f} x EU gas (avg ${g9:.1f})")
    print(f"After-tax decommissioning at our rate: ${decom_after_tax():,.0f}m")
    print(f"DCF enterprise value ${ev:,.0f}m -> {base:.0f}p a share (7%: {high:.0f}p, 11%: {low:.0f}p)")

    # sensitivity: value per share for different discount rates and price levels
    waccs = [0.07, 0.08, 0.09, 0.10, 0.11]
    shifts = [-0.2, -0.1, 0.0, 0.1, 0.2]
    grid = pd.DataFrame(index=[f"{w:.0%}" for w in waccs], columns=[f"{s:+.0%}" for s in shifts], dtype=float)
    for w in waccs:
        for s in shifts:
            # shareholders cannot lose more than they own, so a negative value is shown as 0
            grid.loc[f"{w:.0%}", f"{s:+.0%}"] = max(0.0, value(w, price_shift=s))
    grid.index.name = "WACC \\ oil and gas prices vs futures"
    save_table(grid.round(0), "T15_dcf_sensitivity")
    print(grid.round(0).to_string())

    # scenarios: what has to be true for the shares to be worth more (or less)?
    scenarios = {
        "Base case (futures prices, plateau to 2030, -12% a year after)": {},
        "Harbour's own long-term prices from 2029 ($75 Brent, $11 gas)": {"long_run": "harbour"},
        "Growth case: LLOG + Mexico hold plateau to 2032, -10% after": {"plateau_end": 2032, "decline": 0.10},
        "Both of the above": {"long_run": "harbour", "plateau_end": 2032, "decline": 0.10},
        "Prices 10% below futures": {"price_shift": -0.10},
    }
    rows = []
    for name, settings in scenarios.items():
        c = cash_flows(**settings)
        rows.append({"scenario": name,
                     "cum_production_mmboe": round((c["production_kboepd"] * 0.365).sum()),
                     "ev_usd_m": round(c["pv_fcf"].sum()),
                     "pence": round(pence_per_share(c["pv_fcf"].sum()))})
    table = pd.DataFrame(rows)
    table["upside_%"] = (100 * (table["pence"] / PRICE_PENCE - 1)).round(1)
    save_table(table, "T16_dcf_scenarios", index=False)
    print(table.to_string(index=False))

    # how far above the futures curve prices must be for the DCF to reach today's share price
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value(price_shift=mid) < PRICE_PENCE else (lo, mid)
    print(f"Break-even: the DCF equals {PRICE_PENCE}p when prices are {100 * lo:.1f}% above the futures curve")
    return cf, base, low, high
