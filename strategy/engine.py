"""Backtest engine.

Execution model
---------------
A signal computed from the close of bar *t* is executed at the **open of bar
t+1** and the resulting position is held until the open of bar t+2.  P&L is
therefore measured open-to-open, which is the only accounting that cannot
accidentally use information from inside the bar it trades on.

Positions are expressed as a signed fraction of *total account equity*, so
several symbols can be summed into one portfolio without re-normalising.

Costs charged
-------------
* half spread + slippage on every unit of turnover, both entry and exit;
* commission in bps of the traded notional;
* overnight financing on the held notional, pro-rated by actual calendar days
  (so weekends are charged three days, like a real broker).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs as costs_mod


@dataclass
class SleeveResult:
    symbol: str
    frame: pd.DataFrame                     # per-bar diagnostics
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def net(self) -> pd.Series:
        return self.frame["net"]

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.frame["net"]).cumprod()


def simulate(df: pd.DataFrame, target: pd.Series, symbol: str,
             cost_model: costs_mod.CostModel | None = None,
             charge_financing: bool = True) -> SleeveResult:
    """Run one symbol.  `target` is the desired weight decided at each bar close."""
    cm = cost_model or costs_mod.get(symbol)
    px_open = df["open"].to_numpy(float)
    idx = df.index

    # Position held from open[i] to open[i+1] comes from the signal at close[i-1].
    pos = target.reindex(idx).shift(1).fillna(0.0).to_numpy(float)

    n = len(idx)
    ret_oo = np.zeros(n)
    ret_oo[:-1] = px_open[1:] / px_open[:-1] - 1.0     # open[i] -> open[i+1]

    prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev)
    trade_cost = turnover * cm.half_turn_frac(px_open)

    days = np.zeros(n)
    days[:-1] = np.diff(idx.values).astype("timedelta64[s]").astype(float) / 86400.0
    if charge_financing:
        long_rate, short_rate = costs_mod.financing_rates(symbol, idx, cm)
        rate = np.where(pos >= 0, long_rate, short_rate)
        carry = np.abs(pos) * rate / 360.0 * days
    else:
        carry = np.zeros(n)

    gross = pos * ret_oo
    net = gross - trade_cost - carry

    frame = pd.DataFrame(
        dict(open=px_open, pos=pos, ret_oo=ret_oo, gross=gross,
             trade_cost=trade_cost, carry=carry, net=net),
        index=idx,
    )
    return SleeveResult(symbol=symbol, frame=frame, trades=_extract_trades(frame, symbol))


def _extract_trades(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Collapse the position path into round-trip trades for reporting."""
    pos = frame["pos"].to_numpy(float)
    net = frame["net"].to_numpy(float)
    idx = frame.index
    rows, start, side = [], None, 0.0
    for i in range(len(pos)):
        flat_before = start is None
        if flat_before and pos[i] != 0.0:
            start, side = i, np.sign(pos[i])
        elif not flat_before and (pos[i] == 0.0 or np.sign(pos[i]) != side):
            rows.append(dict(symbol=symbol, entry=idx[start], exit=idx[i],
                             side="long" if side > 0 else "short",
                             bars=i - start, pnl=net[start:i + 1].sum()))
            start = (i if pos[i] != 0.0 else None)
            side = np.sign(pos[i]) if pos[i] != 0.0 else 0.0
    if start is not None:
        rows.append(dict(symbol=symbol, entry=idx[start], exit=idx[-1],
                         side="long" if side > 0 else "short",
                         bars=len(pos) - 1 - start, pnl=net[start:].sum()))
    return pd.DataFrame(rows)


def trading_date(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map bars to the trading day they belong to.

    The CFD day rolls over at 18:00 New York, so a bar stamped 23:00 UTC and
    one stamped 01:00 UTC the next morning are the *same* session.  Bucketing
    on the UTC calendar date would split them across two rows and understate
    how correlated two intraday sleeves really are.
    """
    ny = index.tz_convert("America/New_York")
    return pd.DatetimeIndex((ny + pd.Timedelta(hours=6)).normalize().tz_localize(None))


def to_daily(x: pd.Series) -> pd.Series:
    return x.groupby(trading_date(x.index)).sum()


def combine(sleeves: list[SleeveResult]) -> pd.DataFrame:
    """Sum sleeve returns onto a common session calendar."""
    parts = {}
    for s in sleeves:
        daily = to_daily(s.net)
        key = s.symbol
        while key in parts:
            key += "*"
        parts[key] = daily
    out = pd.DataFrame(parts).sort_index()
    out = out.fillna(0.0)
    out["total"] = out.sum(axis=1)
    first = out["total"].ne(0).idxmax()
    return out.loc[first:]
