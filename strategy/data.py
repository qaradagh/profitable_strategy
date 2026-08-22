"""Data loading utilities for the backtest dataset.

The raw CSVs come from TradingView exports: `time,open,high,low,close`
with ISO-8601 UTC timestamps.  Daily files are naive dates, intraday files
are tz-aware UTC.  Everything is normalised to tz-aware UTC here so that
symbols and timeframes can be joined without surprises.
"""

from __future__ import annotations

import functools
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data_for_backtest")

SYMBOLS = ["us500", "us100", "us30", "xauusd", "eurusd", "gbpusd", "audusd", "eurgbp"]
INDICES = ["us500", "us100", "us30"]
FX = ["eurusd", "gbpusd", "audusd", "eurgbp"]
TIMEFRAMES = ["5", "15", "60", "240", "1D"]

# Minutes per bar, used for annualisation and for resampling checks.
TF_MINUTES = {"5": 5, "15": 15, "60": 60, "240": 240, "1D": 1440}


def _find(symbol: str, timeframe: str) -> str:
    pattern = os.path.join(DATA_DIR, symbol, f"*, {timeframe}_*.csv")
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 file for {symbol}/{timeframe}, got {matches}")
    return matches[0]


@functools.lru_cache(maxsize=None)
def load(symbol: str, timeframe: str) -> pd.DataFrame:
    """Return an OHLC frame indexed by tz-aware UTC timestamps."""
    df = pd.read_csv(_find(symbol, timeframe))
    ts = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.drop(columns=["time"]).astype("float64")
    df.index = pd.DatetimeIndex(ts, name="time")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.attrs["symbol"] = symbol
    df.attrs["timeframe"] = timeframe
    return df


def load_all(timeframe: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    return {s: load(s, timeframe) for s in (symbols or SYMBOLS)}


def pip_size(symbol: str) -> float:
    """Smallest quoted increment used for cost bookkeeping ("1 point")."""
    if symbol in INDICES:
        return 1.0          # index points
    if symbol == "xauusd":
        return 0.01         # cents of an ounce
    if symbol.endswith("jpy"):
        return 0.01
    return 0.0001           # a pip on a 5-digit FX pair


def bars_per_year(timeframe: str) -> float:
    """Approximate tradable bars per year (24/5 market, ~252 trading days)."""
    if timeframe == "1D":
        return 252.0
    return 252.0 * (24 * 60 / TF_MINUTES[timeframe]) * (5 / 7) * (7 / 5)  # 24h x 252d


def describe() -> pd.DataFrame:
    rows = []
    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            df = load(s, tf)
            rows.append(
                dict(symbol=s, tf=tf, bars=len(df), start=df.index[0], end=df.index[-1])
            )
    return pd.DataFrame(rows)
