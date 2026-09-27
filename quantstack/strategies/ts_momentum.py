from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class TimeSeriesMomentum(Strategy):
    name = "ts_momentum"
    mechanism = (
        "Crypto price discovery is slow and flow-driven: leveraged longs/shorts are "
        "force-liquidated into existing moves and retail allocations chase recent "
        "returns with a lag. The counterparty is the late or forced trader who pays "
        "the trend follower for liquidity in the direction of the move. The edge "
        "should fail in range-bound regimes, where the trend follower becomes the "
        "one paying."
    )
    param_grid = {"lookback": [20, 50, 100], "vol_scale": [True, False]}

    def signal(self, bars: pd.DataFrame, lookback: int = 50, vol_scale: bool = True) -> pd.Series:
        close = bars["close"]
        mom = np.log(close / close.shift(lookback))
        direction = np.sign(mom)
        if not vol_scale:
            return direction.fillna(0.0)
        # Scale conviction by trend strength relative to realised vol over the same window.
        vol = np.log(close / close.shift(1)).rolling(lookback).std() * np.sqrt(lookback)
        strength = (mom / vol).clip(-2, 2) / 2
        return strength.fillna(0.0)
