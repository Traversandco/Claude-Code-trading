"""Hypothesis batch 1. Written and gridded before any real data was examined.
Every grid point is a trial; the ledger counts them all."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def _hold_state(entry: pd.Series, exit_: pd.Series, index) -> pd.Series:
    ev = pd.Series(np.nan, index=index)
    ev[entry] = 1.0
    ev[exit_] = 0.0          # exit wins ties
    return ev.ffill().fillna(0.0)


class DonchianBreakout(Strategy):
    name = "donchian_breakout"
    mechanism = (
        "Turtle-style breakout. New N-day highs sit above clustered stop-losses of "
        "shorts and breakout-buy orders; crossing them forces buying (stops, short "
        "liquidations) that extends the move. Counterparty: shorts stopped out and "
        "late breakout chasers. Fails in ranges, where breakouts are faded."
    )
    param_grid = {"entry": [20, 55], "exit": [10, 20]}

    def signal(self, bars: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
        c = bars["close"]
        hi = bars["high"].rolling(entry, min_periods=entry).max().shift(1)   # prior N bars only
        lo = bars["low"].rolling(exit, min_periods=exit).min().shift(1)
        return _hold_state(c > hi, c < lo, bars.index)


class VolSqueezeBreakout(Strategy):
    name = "vol_squeeze"
    mechanism = (
        "Volatility clusters, and quiet periods let leverage build up because margin "
        "looks cheap. When price breaks out of a low-volatility range, that leverage is "
        "forced out (stops, liquidations), producing an outsized move. Counterparty: "
        "positions levered up during the calm. Long-only: only upside breaks traded."
    )
    param_grid = {"pct": [0.2, 0.3]}

    def signal(self, bars: pd.DataFrame, pct: float = 0.2, win: int = 20, lookback: int = 365) -> pd.Series:
        c = bars["close"]
        vol = np.log(c).diff().rolling(win, min_periods=win).std()
        # Percentile of today's vol within the trailing year (trailing only).
        rank = vol.rolling(lookback, min_periods=lookback // 2).rank(pct=True)
        squeezed = rank.shift(1) <= pct
        breakout = c > bars["high"].rolling(win, min_periods=win).max().shift(1)
        ma = c.rolling(win, min_periods=win).mean()
        return _hold_state(squeezed & breakout, c < ma, bars.index)


class FundingSqueeze(Strategy):
    name = "funding_squeeze"
    requires = ("funding",)
    mechanism = (
        "Pure contrarian leg of the funding thesis: when funding turns negative, shorts "
        "are crowded and paying longs to hold. Any rally forces them to cover into it. "
        "Counterparty: the crowded short. Buy on negative funding, hold a fixed number "
        "of days. Fails in genuine bear markets where shorts are right and stay paid."
    )
    param_grid = {"threshold": [0.0, -0.0001], "hold": [5, 10]}

    def signal(self, bars: pd.DataFrame, threshold: float = 0.0, hold: int = 5) -> pd.Series:
        if "funding" not in bars:
            return pd.Series(0.0, index=bars.index)
        f = bars["funding"].rolling("3D", min_periods=1).mean()
        trigger = (f < threshold).astype(float)
        # Long for `hold` bars after any trigger (trailing window = causal).
        return (trigger.rolling(hold, min_periods=1).max() > 0).astype(float)


class BullDipBuy(Strategy):
    name = "bull_dip"
    mechanism = (
        "In established uptrends, sharp pullbacks are usually leverage flushes: "
        "over-levered longs liquidated into thin books, selling below fair value. "
        "Buying the flush while the long-term trend is intact collects the discount. "
        "Counterparty: the liquidated long. Fails when the pullback is the start of a "
        "real trend change (hence the 200-day trend condition)."
    )
    param_grid = {"dip": [0.10, 0.15]}

    def signal(self, bars: pd.DataFrame, dip: float = 0.10, trend_len: int = 200) -> pd.Series:
        c = bars["close"]
        trend = c > c.rolling(trend_len, min_periods=trend_len).mean()
        high90 = c.rolling(90, min_periods=90).max()
        dd = 1 - c / high90
        entry = trend & (dd >= dip) & (dd <= dip + 0.15)
        recovered = c >= c.rolling(30, min_periods=30).max()
        return _hold_state(entry, recovered | ~trend, bars.index)
