"""Indicators.  All are strictly causal: value at bar t uses bars <= t only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(100.0).where(avg_loss.notna())


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: float = 252.0) -> pd.Series:
    return np.log(close).diff().rolling(n).std() * np.sqrt(periods_per_year)


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n).mean()
    s = x.rolling(n).std()
    return (x - m) / s.replace(0.0, np.nan)


def donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """Prior-n-bar high/low, excluding the current bar (no look-ahead)."""
    return df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)
