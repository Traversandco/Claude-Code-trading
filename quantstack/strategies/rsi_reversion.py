from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


class RsiReversion(Strategy):
    name = "rsi_reversion"
    mechanism = (
        "After sharp sell-offs, forced sellers (liquidations, margin calls, panic "
        "redemptions) demand immediacy and accept prices below fair value. A patient "
        "buyer provides that liquidity and is paid a premium as prices recover once "
        "forced flow is exhausted. Counterparty: the liquidated or panicking seller. "
        "Fails in sustained crashes where selling is informed, not forced — hence the "
        "trend filter."
    )
    param_grid = {"period": [7, 14], "entry": [25, 30], "exit": [50, 60]}

    def signal(self, bars: pd.DataFrame, period: int = 14, entry: float = 30,
               exit: float = 55, trend_len: int = 200) -> pd.Series:
        close = bars["close"]
        r = rsi(close, period)
        # Only buy dips while the long-term trend is intact.
        trend_ok = close > close.rolling(trend_len, min_periods=trend_len).mean()
        # Vectorised state machine: 1 on entry, 0 on exit, hold in between.
        events = pd.Series(np.nan, index=close.index)
        events[(r < entry) & trend_ok] = 1.0
        events[(r > exit) | ~trend_ok] = 0.0
        return events.ffill().fillna(0.0)
