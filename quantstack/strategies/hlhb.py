from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .rsi_reversion import rsi


def _crossed_above(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossed_below(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift(1) >= b.shift(1))


def adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX, trailing only."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a = 1 / period
    atr = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean()


class Hlhb(Strategy):
    """Port of freqtrade-strategies `hlhb.py` (4h). Entry rules are the original's.

    Deliberately NOT ported: its minimal_roi table, stoploss=-0.3211 and trailing-stop
    numbers, which were optimized on past data and would enter here as free trials.
    The original's exit needs all three conditions at once and relies on those fitted
    ROI/stop numbers to actually get out, so exits here fire on EITHER the RSI or the
    EMA cross-down. That is a documented deviation, not a tuned one.
    """

    name = "hlhb"
    mechanism = (
        "Trend-initiation breakout: a momentum turn (RSI through 50) coinciding with a "
        "fast/slow average cross while trend strength is high (ADX > 25). Counterparty "
        "as in time-series momentum: slow-to-react and forced traders who provide "
        "liquidity to the move and chase it later. Honest caveat: this is closer to a "
        "pattern than a mechanism; it is included because it is the one freqtrade "
        "community strategy whose entry logic was not hyperopted."
    )
    param_grid = {"adx_min": [25]}    # the original's value; one trial
    default_timeframe = "4h"

    def signal(self, bars: pd.DataFrame, adx_min: float = 25) -> pd.Series:
        hl2 = (bars["close"] + bars["open"]) / 2      # the original's (sic) hl2 definition
        r = rsi(hl2, 10)
        ema5 = bars["close"].ewm(span=5, adjust=False).mean()
        ema10 = bars["close"].ewm(span=10, adjust=False).mean()
        strong = adx(bars) > adx_min
        entry = _crossed_above(r, 50) & _crossed_above(ema5, ema10) & strong
        exit_ = _crossed_below(r, 50) | _crossed_below(ema5, ema10)
        events = pd.Series(np.nan, index=bars.index)
        events[entry] = 1.0
        events[exit_] = 0.0
        return events.ffill().fillna(0.0)
