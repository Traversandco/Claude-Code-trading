from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config, RiskConfig
from ..engine import backtest
from ..metrics import per_period_sharpe
from .base import Strategy

WINDOWS = (3, 12, 48)
L2 = 5.0        # strong ridge penalty: training windows are short


def features(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale-free trailing features in the spirit of intelligent-trading-bot:
    distance from moving average, trailing return, trailing volatility."""
    c = bars["close"]
    lr = np.log(c)
    f = {}
    for w in WINDOWS:
        f[f"ma{w}"] = c / c.rolling(w).mean() - 1
        f[f"ret{w}"] = lr.diff(w)
        f[f"vol{w}"] = lr.diff().rolling(w).std()
    return pd.DataFrame(f, index=bars.index)


def _logistic(X: np.ndarray, y: np.ndarray, l2: float = L2, iters: int = 50) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by Newton's method (numpy only)."""
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(k + 1)
    reg = np.full(k + 1, l2)
    reg[-1] = 0.0                       # don't shrink the intercept
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + reg * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w[:-1], float(w[-1])


def train_model(bars: pd.DataFrame, horizon: int) -> dict | None:
    """Label: is the close `horizon` bars later higher? Labels are computed INSIDE
    the given slice, so the last `horizon` rows have none and are dropped: the model
    never sees a label that needs prices beyond the data it was given (purging)."""
    X = features(bars)
    lr = np.log(bars["close"])
    y = (lr.shift(-horizon) - lr > 0).astype(float).where(lr.shift(-horizon).notna())
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < 30:
        return None
    Xt, yt = X[ok].to_numpy(), y[ok].to_numpy()
    mu, sd = Xt.mean(axis=0), Xt.std(axis=0)
    sd[sd == 0] = 1.0
    coef, b = _logistic((Xt - mu) / sd, yt)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": coef.tolist(), "intercept": b}


class MlDirection(Strategy):
    """The core idea of asavinov/intelligent-trading-bot, on daily bars: a model trained
    on trailing features scores the chance price is higher `horizon` bars out; trade
    when the score clears a threshold. The original grid-searches 10 x 10 thresholds
    (100 trials); here the grid is 2 x 2 and chosen on held-out data."""

    name = "ml_direction"
    mechanism = (
        "Statistical, not structural: assumes short-horizon crypto returns carry weak, "
        "slowly-changing dependence on recent trend and volatility, learnable by a "
        "regularized linear model refitted on a rolling window. No named counterparty, "
        "so by this project's own standard it is a pattern until the gates say "
        "otherwise. Included because it is the representative ML approach from the "
        "listed repositories."
    )
    param_grid = {"horizon": [5, 10], "threshold": [0.05, 0.10]}

    def signal(self, bars: pd.DataFrame, horizon: int = 5, threshold: float = 0.05,
               model: dict | None = None, train_bars: int = 180) -> pd.Series:
        if model is None:
            # Only reached for the trial ledger's full-sample bookkeeping: train on the
            # first window and apply forward (causal after that point).
            model = train_model(bars.iloc[:train_bars], horizon)
        if model is None:
            return pd.Series(0.0, index=bars.index)
        X = features(bars)
        z = (X.to_numpy() - np.array(model["mu"])) / np.array(model["sd"])
        p = 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))
        score = pd.Series(p - 0.5, index=bars.index)
        return (score > threshold).astype(float).where(X.notna().all(axis=1), 0.0)

    def fit(self, train: pd.DataFrame, cfg: Config, risk: RiskConfig) -> dict:
        """Choose (horizon, threshold) on a held-out tail of the training window:
        train on the first 70%, score on the rest. Then refit on the whole window."""
        cut = int(len(train) * 0.7)
        best, best_sr = self.param_combos()[0], -float("inf")
        for params in self.param_combos():
            m = train_model(train.iloc[:cut], params["horizon"])
            if m is None:
                continue
            exp = self.exposure(train, {**params, "model": m}, risk)
            net = backtest(train["close"], exp, cfg)["net"].iloc[cut:]
            sr = per_period_sharpe(net)
            if sr > best_sr:
                best, best_sr = params, sr
        return {**best, "model": train_model(train, best["horizon"])}
