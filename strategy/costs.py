"""Per-symbol transaction and carry costs for a retail CFD account.

Two very different things are modelled here and both matter:

1. **Turn costs** - spread, slippage and commission.  Paid per trade.
2. **Overnight financing** - paid every night the position is open.  On CFDs
   the long side pays `benchmark + markup` and is credited the index dividend
   yield, while the short side is credited `benchmark - markup` and pays the
   dividend away.  With a 4-5% benchmark this is worth 5-6% a year on a
   permanently-long index book, i.e. it is the difference between a strategy
   that works and one that does not.  Ignoring it is the most common way a
   CFD backtest lies to you.

The broker profile matches the account described by the user: indices are
commission-free, gold is cheap, FX pays a real commission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Approximate calendar-year average of the USD overnight benchmark
# (effective fed funds / SOFR).  Used to make financing time-varying instead
# of pinning a 2024 rate onto the 2010s.
USD_OVERNIGHT = {
    1999: 4.97, 2000: 6.24, 2001: 3.89, 2002: 1.67, 2003: 1.13, 2004: 1.35,
    2005: 3.22, 2006: 4.97, 2007: 5.02, 2008: 1.92, 2009: 0.16, 2010: 0.18,
    2011: 0.10, 2012: 0.14, 2013: 0.11, 2014: 0.09, 2015: 0.13, 2016: 0.40,
    2017: 1.00, 2018: 1.83, 2019: 2.16, 2020: 0.36, 2021: 0.08, 2022: 1.68,
    2023: 5.02, 2024: 5.15, 2025: 4.30, 2026: 3.75,
}
DEFAULT_RATE = 3.00          # used for years outside the table (percent)
FINANCING_MARKUP = 2.50      # broker mark-up over the benchmark, percent


@dataclass(frozen=True)
class CostModel:
    spread: float           # typical full spread, in price units
    slippage: float         # extra adverse fill allowance per side, price units
    commission_bps: float   # round-turn commission, bps of notional
    dividend_yield: float   # annual index dividend yield, percent (0 for non-equity)
    markup: float = FINANCING_MARKUP
    financed: bool = True   # False disables overnight carry entirely

    def half_turn_frac(self, price: np.ndarray | float):
        """Cost of trading one unit of notional, one side, as a fraction."""
        return (self.spread / 2.0 + self.slippage) / price + self.commission_bps / 2.0 / 1e4


COSTS: dict[str, CostModel] = {
    # ---- indices: commission-free, financed, dividends credited to longs ----
    "us500": CostModel(spread=0.60, slippage=0.20, commission_bps=0.0, dividend_yield=1.60),
    "us100": CostModel(spread=2.00, slippage=0.80, commission_bps=0.0, dividend_yield=0.75),
    "us30":  CostModel(spread=4.00, slippage=1.50, commission_bps=0.0, dividend_yield=1.90),
    # ---- gold: tight spread, low commission (~$12 round turn per 100oz lot) ----
    "xauusd": CostModel(spread=0.25, slippage=0.10, commission_bps=0.30, dividend_yield=0.0),
    # ---- FX: ~$14 round turn per 100k notional => ~1.2 bps, plus spread ----
    "eurusd": CostModel(spread=0.00012, slippage=0.00004, commission_bps=1.2, dividend_yield=0.0),
    "gbpusd": CostModel(spread=0.00015, slippage=0.00005, commission_bps=1.2, dividend_yield=0.0),
    "audusd": CostModel(spread=0.00014, slippage=0.00005, commission_bps=1.2, dividend_yield=0.0),
    "eurgbp": CostModel(spread=0.00018, slippage=0.00006, commission_bps=1.2, dividend_yield=0.0),
}


def get(symbol: str) -> CostModel:
    return COSTS[symbol]


_BASE: dict[str, CostModel] = dict(COSTS)


def scaled(symbol: str, factor: float) -> CostModel:
    """Same model with turn costs multiplied by `factor` (sensitivity tests).

    Always derived from the pristine baseline, so repeatedly overwriting
    `COSTS` in a sweep cannot compound the multiplier.
    """
    c = _BASE[symbol]
    return CostModel(c.spread * factor, c.slippage * factor, c.commission_bps * factor,
                     c.dividend_yield, c.markup, c.financed)


def benchmark_rate(index: pd.DatetimeIndex) -> np.ndarray:
    """Annual overnight benchmark, in percent, aligned to `index`."""
    return np.array([USD_OVERNIGHT.get(int(y), DEFAULT_RATE) for y in index.year], float)


def financing_rates(symbol: str, index: pd.DatetimeIndex,
                    cm: CostModel | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Annual carry rate (as a fraction) charged to a long and to a short.

    Positive means the position *pays*.  A short in a low-dividend index during
    a high-rate regime can genuinely earn carry, hence the sign convention.
    """
    cm = cm or get(symbol)
    if not cm.financed:
        z = np.zeros(len(index))
        return z, z
    base = benchmark_rate(index)
    long_rate = (base + cm.markup - cm.dividend_yield) / 100.0
    short_rate = (-base + cm.markup + cm.dividend_yield) / 100.0
    return long_rate, short_rate
