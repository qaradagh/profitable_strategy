"""Performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    span_years = (index[-1] - index[0]).days / 365.25
    return len(index) / span_years if span_years > 0 else 252.0


def drawdown(returns: pd.Series) -> pd.Series:
    eq = (1.0 + returns).cumprod()
    return eq / eq.cummax() - 1.0


def stats(returns: pd.Series, rf: float = 0.0) -> dict:
    r = returns.dropna()
    if len(r) < 5:
        return {}
    af = _ann_factor(r.index)
    eq = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(af)
    sharpe = (r.mean() * af - rf) / vol if vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(af)
    sortino = (r.mean() * af - rf) / downside if downside > 0 else np.nan
    dd = drawdown(r)
    maxdd = dd.min()
    wins, losses = r[r > 0], r[r < 0]
    return dict(
        start=r.index[0].date(), end=r.index[-1].date(), years=round(years, 2),
        cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino, maxdd=maxdd,
        calmar=cagr / abs(maxdd) if maxdd < 0 else np.nan,
        profit_factor=wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
        hit=(r > 0).mean(), best_day=r.max(), worst_day=r.min(),
        exposure=(r != 0).mean(), n_periods=len(r),
        ret_total=eq.iloc[-1] - 1.0,
    )


def summary_row(name: str, returns: pd.Series) -> dict:
    return dict(strategy=name, **stats(returns))


def fmt(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ["cagr", "vol", "maxdd", "hit", "exposure", "ret_total", "best_day", "worst_day"]:
        if c in df:
            df[c] = (df[c] * 100).round(2)
    for c in ["sharpe", "sortino", "calmar", "profit_factor"]:
        if c in df:
            df[c] = df[c].round(2)
    return df


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    m = (1.0 + returns).resample("ME").prod() - 1.0
    t = pd.DataFrame(dict(y=m.index.year, mo=m.index.month, r=m.values * 100))
    piv = t.pivot_table(index="y", columns="mo", values="r")
    piv["YEAR"] = ((1 + t.groupby("y")["r"].apply(lambda x: (1 + x / 100).prod() - 1)) - 1) * 100
    return piv.round(2)


def yearly(returns: pd.Series) -> pd.Series:
    return ((1.0 + returns).resample("YE").prod() - 1.0) * 100
