"""
valuation.py - the other three valuation methods, and the blend of all four

  1. DCF / NAV        (dcf.py)                                             weight 40%
  2. Trading comps    peers' EV/EBITDA multiple x Harbour's EBITDA          weight 25%
  3. Precedents       price paid per barrel of reserves in North Sea deals  weight 20%
  4. LBO              most a private-equity buyer could pay for 20% a year  weight 15%

Each method gives a low, base and high value in pence per share. The base values
are blended with the team's weights to give our target price.
"""

import pandas as pd

import dcf
from data import TAB_DIR, PRICE_PENCE, GBPUSD, precedent_deals, save_table

WEIGHTS = {"DCF / NAV": 0.40, "Trading comparables": 0.25, "Precedent transactions": 0.20, "LBO": 0.15}

# the peers we compare with (as in market.py, Serica has no Bloomberg EV so it drops out)
PEERS = ["EnQuest", "Ithaca Energy", "Aker BP", "Var Energi", "DNO", "OKEA", "BlueNord"]
HBR_EBITDA = {2026: 9696.5, 2027: 8574.8}   # Bloomberg consensus, US$ million
PRODUCTION_KBOEPD = 505                     # 2026 guidance 500-505 kboe/d (trading update)
RESERVES_2P = 1210                          # mmboe, our estimate: 1,121 at end-2025 (annual report)
                                            # plus Harbour's target of >=150% reserve replacement in 2026

# LBO assumptions
LBO_LEVERAGE = 1.5              # new debt = 1.5 x "normal" EBITDAX (2029, after the price spike)
LBO_RATE = 0.08                 # interest rate on that debt
LBO_TAX_SHIELD = 0.25           # interest is partly tax-deductible
LBO_EXIT_MULTIPLE = 2.0         # sell after five years at 2.0 x that year's EBITDAX: between the
                                # peers' forward (1.42x) and trailing (3.03x) medians in T12
LBO_YEARS = 5
LEASE_PAYMENTS = 190            # a year (half-year results: $95m in six months)
DECOM_SPEND = 280               # a year (half-year results: $139m in six months)


def to_pence(enterprise_value, leases=False):
    # multiples give enterprise value; subtract what ranks ahead of shareholders,
    # divide by shares, convert to pence.
    # Bloomberg's net debt INCLUDES leases (Harbour FY25: 4,939 = 4,305 net debt + 634 leases),
    # so a peer EV/EBITDA multiple values a business with its leases: deduct them too (leases=True).
    # Deal EVs in press releases are equity plus borrowings, so precedents deduct net debt and hybrids.
    equity = enterprise_value - dcf.NET_DEBT - dcf.HYBRIDS - (dcf.LEASES if leases else 0)
    return 100 * equity / dcf.SHARES / GBPUSD


def comparables():
    # 2027 is the year ahead (2026 is almost over), so the base case uses 2027 multiples
    peers = pd.read_csv(TAB_DIR / "T12_comparables.csv").set_index("company").loc[PEERS]
    values = {}
    for year in (2026, 2027):
        multiples = peers[f"ev_ebitda_{year}e"]
        for label, q in (("p25", 0.25), ("median", 0.5), ("p75", 0.75)):
            m = multiples.quantile(q)
            values[(year, label)] = (m, to_pence(m * HBR_EBITDA[year], leases=True))
            print(f"Comparables {year}e {label}: {m:.2f}x EBITDA -> {values[(year, label)][1]:.0f}p")
    return values


def precedents():
    deals = precedent_deals()
    deals["ev_per_2p_boe"] = deals["ev_usd_m"] / deals["reserves_mmboe"]
    deals["ev_per_flowing_boe_k"] = deals["ev_usd_m"] / deals["production_kboepd"]
    save_table(deals.round(2), "T13_precedent_deals", index=False)

    core = deals[deals["use"] == "Core"]
    per_barrel = core["ev_per_2p_boe"].dropna()
    values = {}
    for label, multiple in (("low", per_barrel.min()), ("median", per_barrel.median()),
                            ("high", per_barrel.max())):
        values[label] = (multiple, to_pence(multiple * RESERVES_2P))
        print(f"Precedents {label}: ${multiple:.1f} per 2P barrel -> {values[label][1]:.0f}p")
    # cross-check only: price per flowing barrel ignores Harbour's short reserve life
    flowing = core["ev_per_flowing_boe_k"].median()
    print(f"Cross-check: ${flowing:.1f}k per flowing boe/d -> {to_pence(flowing * PRODUCTION_KBOEPD):.0f}p")
    return values


def lbo(cf, irr=0.20, exit_multiple=LBO_EXIT_MULTIPLE):
    # The buyer borrows 1.5x normal EBITDAX, adds its own money, buys the shares and repays
    # Harbour's bonds and hybrids. Every spare dollar then repays the new debt, and the
    # company is sold after five years at 2.0x EBITDAX. Work backwards from the sale to
    # find the most the buyer could pay today and still earn `irr` a year.
    debt_at_start = LBO_LEVERAGE * cf.loc[2029, "ebitdax"]
    debt = debt_at_start
    years = range(2027, 2027 + LBO_YEARS)
    for year in years:
        interest_after_tax = debt * LBO_RATE * (1 - LBO_TAX_SHIELD)
        spare_cash = (cf.loc[year, "fcf"] - interest_after_tax - LEASE_PAYMENTS
                      - DECOM_SPEND * (1 - dcf.DECOM_TAX_RELIEF))
        debt = max(0.0, debt - spare_cash)
    sale_price = exit_multiple * cf.loc[years[-1], "ebitdax"]
    equity_at_sale = sale_price - debt
    buyer_cheque = equity_at_sale / (1 + irr) ** LBO_YEARS
    for_the_shares = buyer_cheque + debt_at_start - dcf.NET_DEBT - dcf.HYBRIDS
    return 100 * for_the_shares / dcf.SHARES / GBPUSD


def run(cf, dcf_base, dcf_low, dcf_high):
    comps = comparables()
    deals = precedents()
    lbo_base, lbo_low, lbo_high = lbo(cf, 0.20), lbo(cf, 0.25), lbo(cf, 0.15)
    print(f"LBO: {lbo_base:.0f}p at a 20% return (25%: {lbo_low:.0f}p, 15%: {lbo_high:.0f}p)")
    for exit_multiple in (1.42, 3.03):          # sensitivity to the exit multiple
        print(f"  LBO at a {exit_multiple:.2f}x exit: {lbo(cf, 0.20, exit_multiple):.0f}p")

    summary = pd.DataFrame({
        "base_p": [dcf_base, comps[(2027, "median")][1], deals["median"][1], lbo_base],
        "low_p": [dcf_low, comps[(2027, "p25")][1], deals["low"][1], lbo_low],
        "high_p": [dcf_high, comps[(2027, "p75")][1], deals["high"][1], lbo_high],
        "weight": list(WEIGHTS.values())}, index=list(WEIGHTS.keys()))
    blend = []
    for col in ["base_p", "low_p", "high_p"]:
        blend.append((summary[col] * summary["weight"]).sum())
    summary.loc["Blended (team weights)"] = blend + [1.0]
    summary["upside_%"] = 100 * (summary["base_p"] / PRICE_PENCE - 1)
    save_table(summary.round({"base_p": 1, "low_p": 1, "high_p": 1, "weight": 2, "upside_%": 1}),
               "T17_valuation_summary")
    equal = summary.iloc[:4]["base_p"].mean()
    print(f"Equal weights would give {equal:.0f}p")
    print(summary.round({"base_p": 1, "low_p": 1, "high_p": 1, "weight": 2, "upside_%": 1}).to_string())
    return summary
