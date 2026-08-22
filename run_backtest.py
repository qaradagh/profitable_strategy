#!/usr/bin/env python3
"""Full backtest report for the Asia-Session Gold setup.

    python3 run_backtest.py              # headline report
    python3 run_backtest.py --json out.json

Everything printed here is net of spread, slippage, commission and overnight
financing, with every signal filled on the bar *after* it is generated.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from strategy import costs as costs_mod
from strategy import data, engine, metrics, portfolio
from strategy.risk_backtest import RiskConfig, run_trades, summarise
from strategy.strategies.session import gold_session, session_open_mask

pd.set_option("display.width", 200)

LIVE = portfolio.Config(w_gold=0.85, w_idx_mr=0.075, w_tom=0.075)
START = "2017-05-01"          # first date on which every sleeve has data


def header(t: str) -> None:
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the headline numbers to this file")
    ap.add_argument("--leverage", type=float, default=3.0)
    args = ap.parse_args()

    sleeves, comb = portfolio.build(LIVE)
    total = comb["total"]
    book = total[total.index >= START]

    header("1. SLEEVES  (net of all costs, at their weight in the book)")
    rows = [metrics.summary_row(c, comb[c][comb[c].ne(0).idxmax():]) for c in comb.columns[:-1]]
    print(metrics.fmt(rows)[["strategy", "start", "end", "years", "cagr", "vol",
                             "sharpe", "maxdd"]].to_string(index=False))

    header("2. THE BOOK, UNLEVERED (1x)")
    print(metrics.fmt([metrics.summary_row("portfolio", book)])[
        ["strategy", "start", "end", "years", "cagr", "vol", "sharpe", "sortino",
         "maxdd", "calmar", "profit_factor"]].to_string(index=False))

    header("3. GOLD SLEEVE ALONE, LONG HISTORY (2013-2026, H4)")
    g = data.load("xauusd", "240")
    gr = engine.to_daily(engine.simulate(g, gold_session(g, hold_bars=1), "xauusd").net)
    print(metrics.fmt([metrics.summary_row("gold session 1x", gr)])[
        ["strategy", "start", "end", "years", "cagr", "vol", "sharpe", "maxdd", "calmar"]].to_string(index=False))
    yr = metrics.yearly(gr)
    print("\nby calendar year (%):")
    print(pd.DataFrame({"year": yr.index.year, "ret": yr.values.round(2)}).to_string(index=False))

    header("4. YEAR BY YEAR, WHOLE BOOK (%)")
    yb = metrics.yearly(book)
    print(pd.DataFrame({"year": yb.index.year, "ret": yb.values.round(2)}).to_string(index=False))

    header("5. LEVERAGE FRONTIER  — pick your risk level")
    fr = portfolio.frontier(book, [1, 2, 3, 4, 5, 6, 8, 10])
    show = fr.copy()
    for c in ["cagr", "vol", "maxdd", "monthly_mean", "monthly_median",
              "worst_month", "pct_months_pos"]:
        show[c] = (show[c] * 100).round(2)
    print(show.round(2).to_string(index=False))

    header(f"6. TRADE FREQUENCY  (since {START})")
    tot = 0
    years = (book.index[-1] - book.index[0]).days / 365.25
    for k, v in sleeves.items():
        n = int((v.trades.entry >= START).sum())
        tot += n
        print(f"  {k:16s} {n:5d} trades  ({n / years:5.1f}/yr)")
    print(f"  {'TOTAL':16s} {tot:5d} trades  ({tot / years:5.1f}/yr = {tot / years / 52:.1f} per week)")

    header(f"7. RISK-SIZED VIEW  — 1% of equity risked per gold trade, {args.leverage:.0f}x book")
    d1 = data.load("xauusd", "60")
    ent = session_open_mask(d1.index)
    sig = np.zeros(len(d1), dtype=bool)
    sig[:-1] = ent[1:]
    cfg = RiskConfig(risk_pct=0.01, stop_atr=4.0, hold_bars=3, entry_spread_mult=2.0)
    blot, _ = run_trades(d1, sig, "xauusd", cfg)
    s = summarise(blot)
    for k in ["trades", "years", "trades_per_week", "cagr", "avg_R", "win_rate",
              "stop_rate", "avg_leverage", "max_dd", "monthly_mean",
              "monthly_median", "worst_month", "pct_months_positive"]:
        v = s[k]
        print(f"  {k:20s} {v:>10.4f}" if isinstance(v, float) else f"  {k:20s} {v:>10}")

    header("8. COST SENSITIVITY  — the single biggest dependency")
    base = dict(costs_mod._BASE)
    rows = []
    for f in [0.5, 1.0, 1.5, 2.0, 3.0]:
        costs_mod.COSTS.update({k: costs_mod.scaled(k, f) for k in base})
        _, c2 = portfolio.build(LIVE)
        t = c2["total"]
        t = t[t.index >= START]
        st = metrics.stats(t)
        rows.append(dict(cost_x=f, gold_spread=costs_mod.COSTS["xauusd"].spread,
                         cagr=100 * st["cagr"], sharpe=st["sharpe"], maxdd=100 * st["maxdd"]))
    costs_mod.COSTS.update(base)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    header("9. LOOK-AHEAD AUDIT  — trading the wrong bar must destroy the result")
    rows = []
    for lag in [0, 1, 2]:
        t = gold_session(g, hold_bars=1).shift(lag).fillna(0.0)
        st = metrics.stats(engine.to_daily(engine.simulate(g, t, "xauusd").net))
        rows.append(dict(extra_lag=lag, cagr=100 * st["cagr"], sharpe=st["sharpe"]))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.json:
        st = metrics.stats(book)
        out = dict(book={k: (str(v) if not isinstance(v, (int, float)) else v)
                         for k, v in st.items()},
                   frontier=fr.to_dict(orient="records"),
                   yearly={int(y): float(r) for y, r in zip(yb.index.year, yb.values)},
                   gold_yearly={int(y): float(r) for y, r in zip(yr.index.year, yr.values)},
                   equity=[[str(i.date()), float(v)] for i, v in
                           (1 + book).cumprod().resample("W").last().dropna().items()],
                   trades_per_week=tot / years / 52)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
