"""Production: a validated strategy is not a finished strategy. It decays."""
from __future__ import annotations

import numpy as np
import pandas as pd


def health_check(live_returns: pd.Series, backtest_metrics: dict, window: int = 30,
                 periods_per_year: int = 365) -> dict:
    """Run every bar. Halt when it fires. Kill conditions are fixed at deploy time."""
    r = live_returns.dropna()
    if len(r) < window:
        return {"live_sharpe": None, "current_dd": None, "alerts": [], "action": "CONTINUE",
                "note": f"warming up ({len(r)}/{window} bars)"}

    recent = r.tail(window)
    sd = recent.std()
    live_sharpe = recent.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    equity = np.exp(r.cumsum())
    peak = equity.cummax().iloc[-1]
    dd = (equity.iloc[-1] - peak) / peak

    # A short-window Sharpe is very noisy (s.e. ~ sqrt(ppy/window) annualized), so
    # the article's "below half of backtest" rule alone fires on healthy strategies.
    # Require the shortfall to also be statistically significant (one-sided, 95%).
    bt_sr = backtest_metrics["sharpe"]
    se = np.sqrt(periods_per_year / max(window - 1, 1))
    z = (live_sharpe - bt_sr) / se

    alerts = []
    if live_sharpe < bt_sr * 0.5 and z < -1.645:
        alerts.append("SHARPE_DECAY")
    if dd < backtest_metrics["max_drawdown"] / 100 * 1.5:
        alerts.append("DRAWDOWN_EXCEEDED")

    return {
        "live_sharpe": round(float(live_sharpe), 2),
        "current_dd": round(float(dd) * 100, 1),
        "decay_z": round(float(z), 2),
        "alerts": alerts,
        "action": "HALT" if alerts else "CONTINUE",
    }
