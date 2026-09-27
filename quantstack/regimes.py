"""Regime split: if the edge exists in only one regime, say so directly."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .metrics import metrics


def regime_labels(close: pd.Series, ma_len: int = 200, band: float = 0.02) -> pd.Series:
    """Bull/bear/chop relative to the 200-DAY average of daily closes, for any bar
    size. Each day's average only becomes usable after that day has closed."""
    daily = close.resample("1D").last().dropna()
    ma = daily.rolling(ma_len, min_periods=ma_len).mean().shift(1)
    ma = ma.reindex(close.index.floor("1D")).set_axis(close.index)
    lab = pd.Series("chop", index=close.index)
    lab[close > ma * (1 + band)] = "bull"
    lab[close < ma * (1 - band)] = "bear"
    lab[ma.isna()] = "warmup"
    return lab


def regime_fractions(close: pd.Series, ma_len: int = 200) -> dict:
    lab = regime_labels(close, ma_len)
    lab = lab[lab != "warmup"]
    if lab.empty:
        return {"bull": 0.0, "bear": 0.0, "chop": 0.0}
    vc = lab.value_counts(normalize=True)
    return {k: round(float(vc.get(k, 0.0)), 3) for k in ("bull", "bear", "chop")}


def regime_split(close: pd.Series, net: pd.Series, cfg: Config, ma_len: int = 200) -> dict:
    # Label known at bar t's close applies to the return earned over bar t+1.
    lab = regime_labels(close, ma_len).shift(1).reindex(net.index)
    out = {}
    for regime in ("bull", "bear", "chop"):
        r = net[lab == regime]
        out[regime] = metrics(r, cfg, min_obs=20)
    positive = [k for k, m in out.items() if m.get("sharpe", 0) > 0]
    out["edge_only_in"] = positive[0] if len(positive) == 1 else None
    return out
