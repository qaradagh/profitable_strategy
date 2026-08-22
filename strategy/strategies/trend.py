"""Trend following, used as the diversifying sleeve (gold).

Gold has no dividend and no earnings, so there is no valuation anchor to pull
it back: its returns are close to a random walk with long persistent regimes,
which is the classic habitat for time-series momentum.  A slow breakout /
moving-average filter with volatility-scaled sizing captures that, and its
returns are close to uncorrelated with a dip-buying equity book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr, donchian


def gold_trend(df: pd.DataFrame, *, fast: int = 20, slow: int = 100,
               breakout: int = 50, exit_n: int = 25, atr_len: int = 20,
               risk_per_unit: float = 0.30, max_weight: float = 1.0,
               allow_short: bool = True, weight: float = 1.0) -> pd.Series:
    """Donchian breakout gated by an MA trend filter, sized by inverse ATR."""
    close = df["close"]
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    hi, lo = donchian(df, breakout)
    xhi, xlo = donchian(df, exit_n)
    a = atr(df, atr_len)

    # Volatility target: size so that one ATR move is `risk_per_unit` of the
    # sleeve's notional, capped so a quiet market cannot produce huge leverage.
    size = (risk_per_unit * close / a.replace(0.0, np.nan)).clip(upper=max_weight / 1.0)
    size = size.fillna(0.0) * weight

    up_ok = (ma_f > ma_s).to_numpy(bool)
    dn_ok = (ma_f < ma_s).to_numpy(bool)
    c = close.to_numpy(float)
    hi_a, lo_a = hi.to_numpy(float), lo.to_numpy(float)
    xhi_a, xlo_a = xhi.to_numpy(float), xlo.to_numpy(float)
    sz = size.to_numpy(float)
    valid = (~np.isnan(hi_a)) & (~np.isnan(lo_a)) & ma_s.notna().to_numpy(bool)

    pos = np.zeros(len(c))
    state = 0
    for i in range(len(c)):
        if not valid[i]:
            continue
        if state == 0:
            if c[i] > hi_a[i] and up_ok[i]:
                state = 1
            elif allow_short and c[i] < lo_a[i] and dn_ok[i]:
                state = -1
        elif state == 1 and c[i] < xlo_a[i]:
            state = 0
        elif state == -1 and c[i] > xhi_a[i]:
            state = 0
        pos[i] = state * sz[i]
    return pd.Series(pos, index=df.index)
