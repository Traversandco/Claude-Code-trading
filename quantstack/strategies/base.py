"""Strategy interface.

A strategy is a hypothesis with a stated economic mechanism, a causal signal
function and a small parameter grid. "Give me a mechanism, not a pattern."
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import pandas as pd

from ..config import Config, RiskConfig
from ..engine import backtest
from ..metrics import per_period_sharpe
from ..sizing import target_exposure


class Strategy(ABC):
    name: str = ""
    # Who is on the other side, and why they lose. Required.
    mechanism: str = ""
    param_grid: dict[str, list] = {}
    # Extra data columns beyond OHLCV the signal needs (e.g. "funding").
    requires: tuple[str, ...] = ()
    # Timeframe the strategy was designed for; forward testing defaults to it.
    default_timeframe: str | None = None

    @abstractmethod
    def signal(self, bars: pd.DataFrame, **params) -> pd.Series:
        """Target direction in [-1, 1] at each bar using ONLY bars up to and including it."""

    def param_combos(self) -> list[dict]:
        keys = list(self.param_grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*self.param_grid.values())]

    def exposure(self, bars: pd.DataFrame, params: dict, risk: RiskConfig) -> pd.Series:
        return target_exposure(self.signal(bars, **params), bars, risk)

    def fit(self, train: pd.DataFrame, cfg: Config, risk: RiskConfig) -> dict:
        """Pick params by in-sample net Sharpe. Only ever called on training data."""
        best, best_sr = None, -float("inf")
        for params in self.param_combos():
            bt = backtest(train["close"], self.exposure(train, params, risk), cfg)
            sr = per_period_sharpe(bt["net"])
            if sr > best_sr:
                best, best_sr = params, sr
        return best
