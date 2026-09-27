"""Sizing: where accounts actually die. Size for the path, not the destination."""
from __future__ import annotations

import pandas as pd

from .config import RiskConfig


def position_size(capital, entry, stop, risk_pct=0.01, max_position_pct=0.20):
    """
    risk_pct: fraction of capital lost if the stop hits
    """
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        raise ValueError("stop cannot equal entry")

    units = (capital * risk_pct) / risk_per_unit
    notional = units * entry

    cap = capital * max_position_pct
    if notional > cap:
        units = cap / entry
        notional = cap

    return {
        "units": round(units, 6),
        "notional": round(notional, 2),
        "pct_of_capital": round(notional / capital * 100, 1),
        "loss_if_stopped": round(units * risk_per_unit, 2),
    }


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR using only past and current bars (no centering, no bfill)."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def exposure_fraction(bars: pd.DataFrame, risk: RiskConfig) -> pd.Series:
    """
    Max fraction of equity per unit of signal, using exactly the position_size()
    math with a stop at `atr_mult` ATRs:  notional/capital = risk_pct * entry / stop_distance,
    capped at max_position_pct. Computed causally, so backtest and live size identically.
    """
    stop_dist = risk.atr_mult * atr(bars, risk.atr_period)
    frac = risk.risk_pct * bars["close"] / stop_dist
    return frac.clip(upper=risk.max_position_pct).fillna(0.0)


def target_exposure(signal: pd.Series, bars: pd.DataFrame, risk: RiskConfig) -> pd.Series:
    """Strategy signal (-1..1) → fraction of equity to hold, after sizing and shorting rules."""
    lo = -1.0 if risk.allow_short else 0.0
    return signal.clip(lo, 1.0) * exposure_fraction(bars, risk)
