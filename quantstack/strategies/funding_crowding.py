from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy

# Bybit/Binance linear perps charge a baseline 0.01% per 8h when longs and shorts
# are balanced. Sustained funding well above that means leveraged longs are paying
# up to stay in; below zero means shorts are.
BASELINE_8H = 0.0001


class FundingCrowding(Strategy):
    """Pre-registered before seeing any data. Do not tune thresholds to history:
    every change is a new trial and the ledger will count it."""

    name = "funding_crowding"
    requires = ("funding",)
    mechanism = (
        "Perpetual-futures funding measures how crowded leveraged positioning is. "
        "When funding runs far above its 0.01%/8h baseline, leveraged longs are "
        "paying heavily to hold, and that positioning is fragile: small drops trigger "
        "liquidation cascades. Crypto carry research (e.g. Schmeling, Schrimpf & "
        "Todorov, BIS 2023) finds high funding predicts crash risk and weak returns. "
        "When funding turns negative, shorts are crowded and paying longs, and squeezes "
        "follow. Counterparties: over-levered perp longs who get liquidated at the top, "
        "and crowded shorts who get squeezed at the bottom. Base exposure is a simple "
        "trend filter so the strategy is not long through sustained bear markets. "
        "Fails if funding stops reflecting leverage demand (e.g. basis traders "
        "arbitrage it flat) or in a slow grind lower with neutral funding."
    )
    # 2 x 2 = 4 trials. Crowding thresholds are 3x and 5x the baseline rate.
    param_grid = {"lookback": [50, 100], "crowded": [0.0003, 0.0005]}

    def signal(self, bars: pd.DataFrame, lookback: int = 100, crowded: float = 0.0003,
               smooth: str = "7D") -> pd.Series:
        close = bars["close"]
        trend = (np.log(close / close.shift(lookback)) > 0).astype(float)

        if "funding" in bars:
            # Time-based trailing window: uses bars up to and including this one only.
            f = bars["funding"].rolling(smooth, min_periods=1).mean()
        else:
            f = pd.Series(np.nan, index=bars.index)

        sig = trend.copy()
        sig[f < 0] = 1.0            # shorts crowded: squeeze risk favours longs
        sig[f > crowded] = 0.0      # longs crowded: step aside
        return sig.where(close.shift(lookback).notna(), 0.0).fillna(0.0)
