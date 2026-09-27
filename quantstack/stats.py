"""Multiple testing: the part everyone skips.

Deflated Sharpe Ratio, Bailey & López de Prado (2014). Everything here is in
PER-PERIOD units (not annualized) because the estimator's standard error,
sqrt(1/(T-1)), is per-period. Mixing an annualized Sharpe with sqrt(n_obs)
wildly overstates significance for high-Sharpe strategies and understates it
for modest ones — which is the bug in the formula as usually quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurt, norm, skew as _skew

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among n_trials strategies with zero true skill."""
    if n_trials <= 1:
        return 0.0
    z = (1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * z)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """P(true Sharpe > benchmark). sr in per-period units; kurtosis is NOT excess."""
    denom = 1 - skew * sr + ((kurtosis - 1) / 4) * sr ** 2
    if denom <= 0 or n_obs < 2:
        return 0.0
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denom)))


def deflated_sharpe_from_returns(returns: pd.Series, n_trials: int,
                                 trial_sharpes: list[float] | None = None,
                                 periods_per_year: int = 365,
                                 threshold: float = 0.95) -> dict:
    """
    returns: per-period net returns of the strategy you intend to trade
    n_trials: how many variations you tested. Be honest — the TrialLedger counts for you.
    trial_sharpes: per-period Sharpes of every trial, used to estimate the
                   cross-trial variance. Falls back to the null-hypothesis
                   estimator variance 1/(T-1) when there are too few.
    """
    r = returns.dropna()
    n_obs = len(r)
    sr = float(r.mean() / r.std()) if r.std() > 0 else 0.0
    sk = float(_skew(r)) if n_obs > 2 else 0.0
    ku = float(_kurt(r, fisher=False)) if n_obs > 3 else 3.0

    null_var = 1.0 / max(n_obs - 1, 1)
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        # Never let a suspiciously tight cluster of trials make the bar easier than the null.
        sr_var = max(float(np.var(trial_sharpes, ddof=1)), null_var)
    else:
        sr_var = null_var

    sr0 = expected_max_sharpe(max(int(n_trials), 1), sr_var)
    dsr = probabilistic_sharpe(sr, sr0, n_obs, sk, ku)
    ann = np.sqrt(periods_per_year)
    return {
        "deflated_sharpe": round(dsr, 3),
        "sharpe_annualized": round(float(sr * ann), 2),
        "expected_max_from_noise_annualized": round(float(sr0 * ann), 2),
        "n_trials": int(n_trials),
        "n_obs": n_obs,
        "skew": round(sk, 3),
        "kurtosis": round(ku, 3),
        "verdict": "PASS" if dsr > threshold else "REJECT",
    }


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0,
                    kurtosis: float = 3.0, periods_per_year: int = 365) -> dict:
    """Article-compatible signature: takes an ANNUALIZED Sharpe and converts it."""
    sr = sharpe / np.sqrt(periods_per_year)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), 1.0 / max(n_obs - 1, 1))
    dsr = probabilistic_sharpe(sr, sr0, n_obs, skew, kurtosis)
    return {
        "deflated_sharpe": round(dsr, 3),
        "expected_max_from_noise": round(sr0 * np.sqrt(periods_per_year), 2),
        "verdict": "PASS" if dsr > 0.95 else "REJECT",
    }
