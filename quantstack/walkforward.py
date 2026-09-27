"""Walk-forward: the only test that matters. Optimize on the past, trade the future, roll forward."""
from __future__ import annotations

import pandas as pd

from .config import Config, RiskConfig
from .engine import backtest
from .metrics import metrics
from .strategies.base import Strategy


def walk_forward(bars: pd.DataFrame, strategy: Strategy, cfg: Config, risk: RiskConfig,
                 train_bars: int = 180, test_bars: int = 60,
                 min_in_market: float = 0.25) -> dict:
    """
    Each fold: fit params on the train window only, then compute exposure for the
    test window. Indicators may warm up on any bars BEFORE the test window (that
    data was known at the time); the leakage gate proves the signal is causal.
    Test-window exposures are stitched and backtested once, so turnover and costs
    at fold boundaries are charged exactly as they would have been live.
    """
    if len(bars) < train_bars + test_bars:
        raise ValueError(f"need at least {train_bars + test_bars} bars, have {len(bars)}")

    pieces, fold_meta = [], []
    i = 0
    while i + train_bars + test_bars <= len(bars):
        train = bars.iloc[i: i + train_bars]
        end = i + train_bars + test_bars
        params = strategy.fit(train, cfg, risk)               # fit on train only
        hist = bars.iloc[:end]                                # causal history for warm-up
        exp = strategy.exposure(hist, params, risk).iloc[i + train_bars: end]
        pieces.append(exp)
        fold_meta.append({"start": exp.index[0], "end": exp.index[-1], "params": params})
        i += test_bars

    oos_exposure = pd.concat(pieces)
    oos_prices = bars["close"].loc[oos_exposure.index[0]: oos_exposure.index[-1]]
    # Seed the price series with the bar before the first test bar so the first
    # OOS return is real, then drop it.
    first = bars.index.get_loc(oos_exposure.index[0])
    prices = bars["close"].iloc[max(first - 1, 0): first + len(oos_prices)]
    bt = backtest(prices, oos_exposure.reindex(prices.index).fillna(0.0), cfg).iloc[1:]

    rows = []
    for meta in fold_meta:
        fold = bt.loc[meta["start"]: meta["end"]]
        m = metrics(fold["net"], cfg, min_obs=max(test_bars // 2, 10))
        in_market = float((fold["position"].abs() > 1e-9).mean())
        rows.append({"start": meta["start"], "params": meta["params"],
                     "in_market": round(in_market, 2), **m})

    folds = pd.DataFrame(rows)
    # A fold spent (almost) entirely flat has no Sharpe worth judging: its "returns"
    # are one exit fee over a near-zero std. Judge only folds where the strategy traded.
    active = folds[folds["in_market"] >= min_in_market]
    judged = active if len(active) else folds
    return {
        "folds": folds,
        "n_active_folds": len(active),
        "mean_sharpe": round(float(judged["sharpe"].mean()), 2),
        "positive_folds": f"{int((judged['sharpe'] > 0).sum())}/{len(judged)}",
        "positive_frac": float((judged["sharpe"] > 0).mean()) if len(active) else 0.0,
        "worst_fold": round(float(judged["sharpe"].min()), 2),
        "oos": bt,
        "oos_metrics": metrics(bt["net"], cfg, min_obs=min(100, len(bt))),
        # What live trading uses: the same fit procedure on the most recent window.
        "live_params": strategy.fit(bars.iloc[-train_bars:], cfg, risk),
    }
