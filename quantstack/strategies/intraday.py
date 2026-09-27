"""Hypothesis batch 2: intraday (15m native). Pre-registered before any real
intraday data was examined. Every grid point is a trial."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .batch1 import _hold_state


class SessionBreakout(Strategy):
    name = "session_breakout"
    default_timeframe = "15m"
    mechanism = (
        "Asia-session range breakout. From 00:00 to 08:00 UTC crypto trades thinly and "
        "builds a range; stop orders and breakout entries cluster just beyond its "
        "edges. When European, then US, liquidity arrives, a break above the range "
        "triggers those stops (short covering) and breakout buying. Counterparty: "
        "shorts stopped out at the range high. Long-only; flat by the exit hour so no "
        "position is carried into the next thin session. At most one trade a day."
    )
    param_grid = {"exit_hour": [16, 23]}

    def signal(self, bars: pd.DataFrame, exit_hour: int = 16, range_end: int = 8) -> pd.Series:
        idx = bars.index
        day = idx.floor("1D")
        hour = idx.hour + idx.minute / 60
        in_range = hour < range_end
        # Range high/low of today's 00:00-range_end bars, carried forward within the day.
        hi = bars["high"].where(in_range).groupby(day).cummax().groupby(day).ffill()
        lo = bars["low"].where(in_range).groupby(day).cummin().groupby(day).ffill()
        active = (hour >= range_end) & (hour < exit_hour)
        entry = active & (bars["close"] > hi)
        exit_ = ~active | (bars["close"] < lo)
        return _hold_state(entry, exit_, idx)


class LiquidationRebound(Strategy):
    name = "liquidation_rebound"
    default_timeframe = "15m"
    mechanism = (
        "A single-bar crash far outside normal volatility on a volume spike is the "
        "footprint of a liquidation cascade: exchanges market-sell levered longs into a "
        "thin book regardless of price. That selling is forced, not informed, so price "
        "tends to recover part of it once the cascade exhausts. Counterparty: the "
        "liquidation engine selling for liquidated longs. Buy after the shock bar, hold "
        "a fixed number of bars."
    )
    param_grid = {"sigma": [3.0, 4.0], "hold": [4, 16]}

    def signal(self, bars: pd.DataFrame, sigma: float = 3.0, hold: int = 4) -> pd.Series:
        r = np.log(bars["close"]).diff()
        # Trailing 7-day baselines (time-based so they mean the same on any bar size),
        # excluding the current bar so the shock cannot inflate its own baseline.
        sd = r.rolling("7D", min_periods=20).std().shift(1)
        vol_med = bars["volume"].rolling("7D", min_periods=20).median().shift(1)
        shock = (r < -sigma * sd) & (bars["volume"] > 3 * vol_med)
        return (shock.astype(float).rolling(hold, min_periods=1).max() > 0).astype(float)
