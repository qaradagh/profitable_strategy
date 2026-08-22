"""Trade-level simulator with stop-loss and fixed-fractional risk sizing.

The vectorised engine in `engine.py` answers "what does this exposure profile
return".  This module answers the question a discretionary trader actually
asks: *if I risk 1% of the account per trade, what do I make per month, and
what is the worst run?*

Sizing is fixed-fractional against the stop: the number of units is chosen so
that being stopped out costs exactly `risk_pct` of current equity.  Because
the stop distance is measured in ATR, position size automatically shrinks when
gold is volatile and grows when it is calm.

Intrabar handling is deliberately pessimistic:
* if a bar's low breaches the stop, the trade is closed at the stop price plus
  slippage - never at a better price;
* if both the stop and the profit target sit inside the same bar, the stop is
  assumed to have been hit first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import costs as costs_mod
from .indicators import atr


@dataclass
class RiskConfig:
    risk_pct: float = 0.01        # fraction of equity lost if the stop is hit
    stop_atr: float = 1.0         # stop distance in ATR units
    target_atr: float = 0.0       # 0 disables the profit target
    hold_bars: int = 3            # time exit
    atr_len: int = 20
    max_leverage: float = 20.0    # cap on notional / equity
    entry_spread_mult: float = 1.0  # widen the entry spread (session-open cost)


def run_trades(df: pd.DataFrame, entries: np.ndarray, symbol: str, cfg: RiskConfig,
               cost_model: costs_mod.CostModel | None = None,
               start_equity: float = 10_000.0) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate long trades opened on the bar *after* each True in `entries`.

    Returns a trade blotter and the equity curve stamped at each exit.
    """
    cm = cost_model or costs_mod.get(symbol)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    a = atr(df, cfg.atr_len).to_numpy(float)
    idx = df.index
    n = len(df)

    long_rate, _ = costs_mod.financing_rates(symbol, idx, cm)
    hours = np.zeros(n)
    hours[:-1] = np.diff(idx.values).astype("timedelta64[s]").astype(float) / 3600.0

    equity = start_equity
    rows = []
    for i in np.flatnonzero(entries):
        j = i + 1                                  # fill on the next bar's open
        if j >= n - 1 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry_px = o[j]
        stop_dist = cfg.stop_atr * a[i]
        stop_px = entry_px - stop_dist
        target_px = entry_px + cfg.target_atr * a[i] if cfg.target_atr > 0 else np.inf

        # Fixed-fractional sizing: a stop-out costs exactly risk_pct of equity.
        units = (cfg.risk_pct * equity) / stop_dist
        notional = units * entry_px
        if notional > cfg.max_leverage * equity:
            units = cfg.max_leverage * equity / entry_px
            notional = units * entry_px

        entry_cost = units * (cm.spread / 2.0 * cfg.entry_spread_mult + cm.slippage)
        entry_cost += notional * cm.commission_bps / 2.0 / 1e4

        end = min(j + cfg.hold_bars, n - 1)
        exit_px, reason, k = o[end], "time", end
        for k in range(j, end + 1):
            if lo[k] <= stop_px:                    # stop first if both touched
                exit_px, reason = stop_px - cm.slippage, "stop"
                break
            if h[k] >= target_px:
                exit_px, reason = target_px - cm.slippage, "target"
                break
        else:
            k = end

        exit_cost = units * (cm.spread / 2.0 + cm.slippage)
        exit_cost += units * exit_px * cm.commission_bps / 2.0 / 1e4
        held_h = hours[j:k].sum() if k > j else hours[j]
        carry = notional * long_rate[j] / 360.0 * (held_h / 24.0)

        pnl = units * (exit_px - entry_px) - entry_cost - exit_cost - carry
        r_mult = pnl / (cfg.risk_pct * equity)
        equity += pnl
        rows.append(dict(entry_time=idx[j], exit_time=idx[k], reason=reason,
                         entry=entry_px, exit=exit_px, units=units,
                         notional=notional, leverage=notional / (equity - pnl),
                         pnl=pnl, ret=pnl / (equity - pnl), R=r_mult, equity=equity))

    blotter = pd.DataFrame(rows)
    if blotter.empty:
        return blotter, pd.Series(dtype=float)
    curve = blotter.set_index("exit_time")["equity"]
    return blotter, curve


def summarise(blotter: pd.DataFrame, start_equity: float = 10_000.0) -> dict:
    if blotter.empty:
        return {}
    eq = pd.concat([pd.Series([start_equity], index=[blotter.entry_time.iloc[0]]),
                    blotter.set_index("exit_time")["equity"]])
    years = (blotter.exit_time.iloc[-1] - blotter.entry_time.iloc[0]).days / 365.25
    dd = (eq / eq.cummax() - 1.0).min()
    monthly = blotter.set_index("exit_time")["pnl"].groupby(
        pd.Grouper(freq="ME")).sum() / start_equity
    wins, losses = blotter.R[blotter.R > 0], blotter.R[blotter.R <= 0]
    return dict(
        trades=len(blotter), years=round(years, 2),
        trades_per_week=len(blotter) / (years * 52.0),
        total_return=eq.iloc[-1] / start_equity - 1.0,
        cagr=(eq.iloc[-1] / start_equity) ** (1 / years) - 1.0 if years > 0 else np.nan,
        avg_R=blotter.R.mean(), median_R=blotter.R.median(),
        win_rate=(blotter.R > 0).mean(),
        avg_win_R=wins.mean() if len(wins) else np.nan,
        avg_loss_R=losses.mean() if len(losses) else np.nan,
        expectancy_R=blotter.R.mean(),
        profit_factor=blotter.pnl[blotter.pnl > 0].sum() / abs(blotter.pnl[blotter.pnl < 0].sum())
        if (blotter.pnl < 0).any() else np.nan,
        max_dd=dd, avg_leverage=blotter.leverage.mean(),
        stop_rate=(blotter.reason == "stop").mean(),
        monthly_mean=monthly.mean(), monthly_median=monthly.median(),
        monthly_std=monthly.std(), worst_month=monthly.min(), best_month=monthly.max(),
        pct_months_positive=(monthly > 0).mean(),
    )
