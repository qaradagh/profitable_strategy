"""Assemble the sleeves into one book and size it.

Two sizing layers:

1. **Per sleeve** - inverse realised volatility, so each sleeve contributes a
   stable amount of risk instead of quietly becoming the whole portfolio when
   its market gets busy.
2. **Portfolio** - a single leverage multiplier chosen to hit a target
   volatility.  This is the only knob the user needs to turn: everything about
   the *shape* of the equity curve is fixed by the sleeves, and this scales it.

The volatility estimate is always lagged by one bar, so no position is ever
sized with information from the bar it is filled on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as data_mod
from . import engine, metrics
from .indicators import realized_vol
from .strategies import index_mean_reversion
from .strategies.seasonal import turn_of_month
from .strategies.session import gold_session

INDICES = ["us500", "us100", "us30"]


@dataclass
class Config:
    gold_tf: str = "240"          # "240" for the 2013+ history, "60" for the live spec
    gold_hold: int = 1            # bars held (1 bar on H4 = 4h, 3 bars on H1 = 3h)
    vol_lookback: int = 60
    vol_target_sleeve: float = 0.15
    max_scale: float = 4.0
    w_gold: float = 0.70
    w_idx_mr: float = 0.15
    w_tom: float = 0.15
    use_index_sleeves: bool = True
    idx_mr_params: dict = field(default_factory=lambda: dict(
        rsi_len=2, entry=20, exit_rsi=70, exit_ma=10 ** 9,
        max_hold=24, trend_len=200, stop=0.05))


def _inv_vol(close: pd.Series, lookback: int, bars_per_year: float,
             target: float, cap: float) -> pd.Series:
    rv = realized_vol(close, lookback, periods_per_year=bars_per_year).shift(1)
    return (target / rv).clip(upper=cap).fillna(0.0)


def build(cfg: Config = Config()) -> tuple[dict[str, engine.SleeveResult], pd.DataFrame]:
    """Run every sleeve and return them plus the combined daily return frame."""
    sleeves: dict[str, engine.SleeveResult] = {}

    # ---- core: gold Asian session -------------------------------------
    g = data_mod.load("xauusd", cfg.gold_tf)
    bars_yr = 252.0 * (24 * 60 / data_mod.TF_MINUTES[cfg.gold_tf])
    scale = _inv_vol(g["close"], cfg.vol_lookback, bars_yr,
                     cfg.vol_target_sleeve, cfg.max_scale)
    tgt = gold_session(g, hold_bars=cfg.gold_hold) * scale * cfg.w_gold
    sleeves["gold_session"] = engine.simulate(g, tgt, "xauusd")

    if cfg.use_index_sleeves:
        # ---- index short-term mean reversion (H4) ----------------------
        for s in INDICES:
            d = data_mod.load(s, "240")
            sc = _inv_vol(d["close"], cfg.vol_lookback, 252.0 * 6,
                          cfg.vol_target_sleeve, cfg.max_scale)
            t = index_mean_reversion(d, **cfg.idx_mr_params) * sc * (cfg.w_idx_mr / len(INDICES))
            sleeves[f"idx_mr_{s}"] = engine.simulate(d, t, s)

        # ---- index turn of month (D1) ----------------------------------
        for s in INDICES:
            d = data_mod.load(s, "1D")
            sc = _inv_vol(d["close"], cfg.vol_lookback, 252.0,
                          cfg.vol_target_sleeve, cfg.max_scale)
            t = turn_of_month(d) * sc * (cfg.w_tom / len(INDICES))
            sleeves[f"tom_{s}"] = engine.simulate(d, t, s)

    combined = engine.combine(list(sleeves.values()))
    combined.columns = list(sleeves.keys()) + ["total"]
    return sleeves, combined


def apply_leverage(daily_returns: pd.Series, leverage: float) -> pd.Series:
    return daily_returns * leverage


def leverage_for_vol(daily_returns: pd.Series, target_vol: float) -> float:
    realised = daily_returns.std() * np.sqrt(252.0)
    return target_vol / realised if realised > 0 else 0.0


def frontier(daily_returns: pd.Series, levers: list[float]) -> pd.DataFrame:
    rows = []
    for L in levers:
        r = daily_returns * L
        st = metrics.stats(r)
        monthly = (1 + r).resample("ME").prod() - 1
        rows.append(dict(leverage=L, cagr=st["cagr"], vol=st["vol"], sharpe=st["sharpe"],
                         maxdd=st["maxdd"], calmar=st["calmar"],
                         monthly_mean=monthly.mean(), monthly_median=monthly.median(),
                         worst_month=monthly.min(), pct_months_pos=(monthly > 0).mean()))
    return pd.DataFrame(rows)
