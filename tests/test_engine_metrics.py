import numpy as np
import pandas as pd

from quantstack.config import Config
from quantstack.engine import backtest
from quantstack.metrics import metrics


def _series(vals):
    idx = pd.date_range("2021-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def test_position_is_signal_shifted_by_one_bar():
    prices = _series(np.linspace(100, 120, 20))
    sig = _series(np.r_[np.zeros(5), np.ones(10), np.zeros(5)])
    bt = backtest(prices, sig, Config())
    assert (bt["position"] == sig.shift(1).fillna(0)).all()


def test_perfect_foresight_signal_does_not_earn_the_move_it_saw():
    # Signal = "price went up this bar". Unshifted, it would capture every up-move.
    rng = np.random.default_rng(0)
    prices = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    saw_up = (prices.diff() > 0).astype(float)
    cfg = Config(fee_bps=0.001, slippage_bps=0.001)
    bt = backtest(prices, saw_up, cfg)
    assert abs(metrics(bt["net"], cfg)["sharpe"]) < 1.5   # no fantasy Sharpe


def test_costs_charged_on_turnover():
    prices = _series(np.full(10, 100.0))
    sig = _series([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    cfg = Config(fee_bps=5, slippage_bps=3)
    bt = backtest(prices, sig, cfg)
    turnover = bt["position"].diff().abs().fillna(0).sum()
    assert np.isclose(bt["costs"].sum(), turnover * 8 / 1e4)
    assert bt["net"].sum() < 0   # flat prices + trading = losing money


def test_metrics_insufficient_data_and_drawdown():
    cfg = Config()
    assert metrics(_series(np.zeros(50)), cfg)["error"] == "insufficient_data"
    r = _series(np.r_[np.full(50, 0.01), np.full(50, -0.02), np.full(50, 0.01)])
    m = metrics(r, cfg)
    assert m["max_drawdown"] < 0 and m["longest_dd_bars"] >= 50


def test_config_rejects_unsafe_settings(tmp_path):
    import pytest
    from quantstack.config import load_config
    p = tmp_path / "c.yaml"
    for bad in ("fee_bps: 0\n", "mode: yolo\n", "risk:\n  allow_short: true\n", "risk:\n  risk_pct: 0.2\n"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            load_config(p)
    p.write_text(open("config.example.yaml").read())
    assert load_config(p).mode == "paper"
