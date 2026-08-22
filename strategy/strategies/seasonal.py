"""Turn-of-the-month sleeve on equity indices.

Equity indices earn an outsized share of their return in the days around a
month boundary: on US500 from 1999-2026 the window from four sessions before
month-end to three sessions after averages +7.2 bps a day (t=3.0) against
+0.9 bps (t=0.5) for the rest of the month, and it was positive in 23 of 28
calendar years.  The usual explanation is mechanical: payroll-linked
retirement contributions, index-fund reinvestment of dividends and monthly
rebalancing all land in the same few sessions.

Only about a third of the month is spent in the market, which matters on a
financed CFD: the other two-thirds of the year's financing bill is never paid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def turn_of_month(df: pd.DataFrame, *, days_before: int = 4, days_after: int = 3,
                  weight: float = 1.0) -> pd.Series:
    """Long during the turn-of-month window, flat otherwise (daily bars)."""
    idx = df.index
    month = pd.PeriodIndex(idx.tz_localize(None) if idx.tz else idx, freq="M").to_numpy()
    order = pd.Series(np.arange(len(idx)))
    fwd = order.groupby(month).cumcount().to_numpy()          # 0 = first session
    bwd = order.groupby(month).transform(                     # 0 = last session
        lambda s: np.arange(len(s))[::-1]).to_numpy()
    in_window = (fwd < days_after) | (bwd < days_before)
    return pd.Series(np.where(in_window, weight, 0.0), index=idx)
