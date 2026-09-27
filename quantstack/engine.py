"""The backtest engine. Build it once, build it correctly, never rewrite it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def backtest(prices: pd.Series, signal: pd.Series, cfg: Config) -> pd.DataFrame:
    """
    prices: close prices, datetime index
    signal: target exposure (fraction of equity, -1..+1), computed from data
            available AT that timestamp (i.e. at that bar's close)
    """
    signal = signal.reindex(prices.index)

    # THE MOST IMPORTANT LINE IN THIS PACKAGE.
    # You act on the NEXT bar, not the one you just saw.
    position = signal.shift(1).fillna(0).clip(-cfg.max_leverage, cfg.max_leverage)

    returns = np.log(prices / prices.shift(1)).fillna(0)
    gross = position * returns

    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (cfg.fee_bps + cfg.slippage_bps) / 1e4

    net = gross - costs
    equity = cfg.initial_capital * np.exp(net.cumsum())

    return pd.DataFrame({
        "position": position,
        "gross": gross,
        "costs": costs,
        "net": net,
        "equity": equity,
    })
