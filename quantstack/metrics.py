"""Metrics that don't lie. Return alone tells you nothing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def metrics(net: pd.Series, cfg: Config, min_obs: int = 100) -> dict:
    r = net.dropna()
    if len(r) < min_obs:
        return {"error": "insufficient_data", "n_obs": len(r)}

    ann_return = r.mean() * cfg.periods_per_year
    ann_vol = r.std() * np.sqrt(cfg.periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    equity = np.exp(r.cumsum())
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # how long you sat underwater — the number that
    # actually decides whether you'd have held on
    underwater = (dd < 0).astype(int)
    longest_dd = underwater.groupby((underwater != underwater.shift()).cumsum()).sum().max()

    return {
        "sharpe": round(float(sharpe), 2),
        "ann_return": round(float(ann_return) * 100, 1),
        "max_drawdown": round(float(max_dd) * 100, 1),
        "longest_dd_bars": int(longest_dd),
        "calmar": round(float(ann_return / abs(max_dd)), 2) if max_dd else 0.0,
        "n_obs": len(r),
    }


def per_period_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd) if sd > 0 else 0.0
