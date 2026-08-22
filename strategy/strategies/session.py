"""Gold Asian-session sleeve - the core edge of this setup.

What it trades
--------------
Gold is bought at the open of the first bar of each new trading session
(22:00 UTC in northern summer, 23:00 UTC in winter - the daily CFD rollover)
and sold a few hours later.  Measured over 2013-2026 that single bar returns
+5.9 bps on average with a t-statistic of 9.2 and was positive in **14 of 14
calendar years**, while every other bar of the day averages -0.3 bps.  Put
differently: essentially the whole of gold's multi-year drift is earned in the
first hours of the Asian session, and a buy-and-hold investor pays financing
for the other twenty hours to get it.

Why it should exist
-------------------
Asian trading hours are when physical demand prints - the Shanghai Gold
Exchange opens, and Asian refiners, jewellers and central banks buy in size
against a book that London and COMEX dealers have run flat into the close.
Dealers who are short from the US afternoon have to cover into that demand.
The mirror image of this - gold drifting lower into the London PM fix - is a
long-documented feature of the metal's intraday profile.  It is a flow effect,
not a forecasting model, which is why it does not decay the way a technical
pattern does.

Rules
-----
* Enter long at the open of the session's first bar.
* Exit `hold_bars` bars later (4h on H4 data, 1-3h on H1).
* Skip the Sunday re-open: it carries the week's worst spreads and is the
  weakest day in the sample (t=1.95 versus 4.3-6.3 for Mon-Thu).
* Optionally skip sessions where the previous session was unusually violent.

The entry is deliberately unfiltered.  Every conditioning variable tested
(trend, prior-day direction, prior-day range, volatility regime) left the edge
intact in *both* branches, which is evidence the effect is structural rather
than a subset found by searching.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr

ROLLOVER_TZ = "America/New_York"
ROLLOVER_HOUR = 18          # 17:00-18:00 NY is the daily CFD maintenance break


def session_open_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first bar of each trading session.

    The CFD day rolls over at 18:00 New York, which is 22:00 UTC in northern
    summer and 23:00 UTC in winter.  Anchoring on the New York clock instead of
    a UTC hour makes the rule survive both daylight-saving switches, and it
    works unchanged on 5m, H1 and H4 bars.
    """
    ny_hour = index.tz_convert(ROLLOVER_TZ).hour
    at_open = ny_hour == ROLLOVER_HOUR
    first = np.zeros(len(index), dtype=bool)
    first[0] = at_open[0]
    first[1:] = at_open[1:] & ~at_open[:-1]      # first bar of that hour only
    return first


def gold_session(df: pd.DataFrame, *, hold_bars: int = 1, skip_sunday: bool = True,
                 max_prev_range_atr: float = 0.0, atr_len: int = 20,
                 weight: float = 1.0) -> pd.Series:
    """Target-weight series for the session trade.

    The engine fills a target set at bar *i* on the open of bar *i+1*, so the
    signal is written onto the bar immediately preceding each session open.
    That bar closes at the exact price the session bar opens at (the feed is
    continuous), so the fill is the session-open price - which is where the
    move measured above actually starts.
    """
    idx = df.index
    is_open = session_open_mask(idx)

    enter = np.zeros(len(idx), dtype=bool)
    enter[:-1] = is_open[1:]          # signal on the bar before the session bar

    if skip_sunday:
        # Sunday re-open: worst spreads of the week, weakest measured edge
        # (t=1.95 versus 4.3-6.3 for the Monday-Thursday opens).  The test is
        # on the *session* bar (i+1), not on the bar carrying the signal.
        sunday = np.asarray(idx.dayofweek == 6)
        sess_is_sunday = np.zeros(len(idx), dtype=bool)
        sess_is_sunday[:-1] = sunday[1:]
        enter &= ~sess_is_sunday

    if max_prev_range_atr > 0:
        a = atr(df, atr_len)
        rng = (df["high"] - df["low"]) / a.replace(0.0, np.nan)
        enter &= (rng < max_prev_range_atr).fillna(True).to_numpy()

    pos = np.zeros(len(idx))
    for i in np.flatnonzero(enter):
        pos[i:i + hold_bars] = weight   # hold for `hold_bars` filled bars
    return pd.Series(pos, index=idx)
