import numpy as np
import pandas as pd

from quantstack.config import RiskConfig
from quantstack.health import health_check
from quantstack.sizing import atr, exposure_fraction, position_size


def test_position_size_matches_article():
    out = position_size(10_000, 100, 95, risk_pct=0.01, max_position_pct=0.20)
    assert out["loss_if_stopped"] == 20.0 or out["pct_of_capital"] == 20.0
    out = position_size(10_000, 100, 50, risk_pct=0.01)
    assert out["units"] == 2.0 and out["loss_if_stopped"] == 100.0


def test_exposure_fraction_equals_position_size_math(noise_bars):
    risk = RiskConfig(risk_pct=0.01, max_position_pct=1.0, atr_mult=2.0)
    frac = exposure_fraction(noise_bars, risk)
    t = 500
    price = noise_bars["close"].iloc[t]
    stop = price - 2.0 * atr(noise_bars, risk.atr_period).iloc[t]
    ps = position_size(10_000, price, stop, risk_pct=0.01, max_position_pct=1.0)
    assert np.isclose(frac.iloc[t] * 10_000, ps["notional"], rtol=1e-4)


def test_exposure_capped():
    risk = RiskConfig(max_position_pct=0.2)
    idx = pd.date_range("2021-01-01", periods=60, freq="D", tz="UTC")
    calm = pd.DataFrame({"open": 100.0, "high": 100.01, "low": 99.99, "close": 100.0}, index=idx)
    assert exposure_fraction(calm, risk).max() <= 0.2 + 1e-12


def test_health_check_drawdown_and_decay():
    bt = {"sharpe": 1.5, "max_drawdown": -10.0}
    good = pd.Series(np.full(120, 0.002) + np.random.default_rng(0).normal(0, 0.01, 120))
    assert health_check(good, bt, 90)["action"] == "CONTINUE"
    crash = pd.Series(np.r_[np.full(60, 0.001), np.full(60, -0.004)])
    out = health_check(crash, bt, 90)
    assert out["action"] == "HALT" and "DRAWDOWN_EXCEEDED" in out["alerts"]
    bleed = pd.Series(np.full(120, -0.0005) + np.random.default_rng(1).normal(0, 0.001, 120))
    assert "SHARPE_DECAY" in health_check(bleed, {"sharpe": 1.5, "max_drawdown": -90.0}, 90)["alerts"]
    assert health_check(good.head(10), bt, 90)["action"] == "CONTINUE"   # warming up
