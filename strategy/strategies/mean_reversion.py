"""Short-horizon mean reversion on equity indices.

Rationale
---------
Daily index returns are negatively autocorrelated (-0.09 to -0.12 measured on
9-27 years of data here).  Buying a short-term dip *inside an established
uptrend* harvests that reversal: you are being paid to supply liquidity to
forced sellers, and the long-term equity drift is a tailwind rather than a
headwind.  Holding periods are 2-5 days, so CFD financing barely bites - which
is what makes the setup viable on a commission-free index account.

Rules
-----
Long only.  A position is opened when

    close > SMA(trend_len)              trend filter, only buy dips in uptrends
    RSI(rsi_len) < entry                oversold short-term

and closed when either

    RSI(rsi_len) > exit_rsi             the snap-back has happened
    close > SMA(exit_ma)                price recovered above its short average
    max_hold bars elapsed               time stop, stops a dip becoming a trend
    close < entry_price * (1 - stop)    disaster stop

Signals are evaluated on the close and filled on the next open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import rsi


def index_mean_reversion(df: pd.DataFrame, *, rsi_len: int = 2, entry: float = 10.0,
                         exit_rsi: float = 65.0, exit_ma: int = 5, trend_len: int = 200,
                         max_hold: int = 6, stop: float = 0.06,
                         weight: float = 1.0) -> pd.Series:
    """Return the target weight series (0 or `weight`)."""
    close = df["close"]
    r = rsi(close, rsi_len)
    sma_trend = close.rolling(trend_len).mean()
    sma_exit = close.rolling(exit_ma).mean()

    ok = sma_trend.notna() & r.notna()
    want_in = (close > sma_trend) & (r < entry) & ok
    want_out = (r > exit_rsi) | (close > sma_exit)

    c = close.to_numpy(float)
    wi = want_in.to_numpy(bool)
    wo = want_out.to_numpy(bool)

    pos = np.zeros(len(c))
    held, entry_px = 0, np.nan
    for i in range(len(c)):
        if held == 0:
            if wi[i]:
                held, entry_px = 1, c[i]
                pos[i] = weight
        else:
            stopped = c[i] < entry_px * (1.0 - stop)
            if wo[i] or held >= max_hold or stopped:
                held, entry_px = 0, np.nan
            else:
                held += 1
                pos[i] = weight
    return pd.Series(pos, index=df.index)
