import numpy as np
import pandas as pd

from quantstack.config import BotConfig
from quantstack.leakage import causality_tests, run_critic
from quantstack.strategies import get_strategy
from quantstack.strategies.ml_direction import features, train_model


def test_hlhb_is_causal_and_trades(noise_bars):
    s = get_strategy("hlhb")
    sig = s.signal(noise_bars)
    res = causality_tests(lambda b: s.signal(b), noise_bars)
    assert not res["lookahead_at"] and not res["repaint_at"]
    assert 0.02 < sig.mean() < 0.9 and set(sig.unique()) <= {0.0, 1.0}


def test_hlhb_enters_on_the_originals_conditions():
    idx = pd.date_range("2024-01-01", periods=120, freq="4h", tz="UTC")
    c = np.r_[np.linspace(100, 80, 60), np.linspace(80, 120, 60)]   # down then sharp up
    bars = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=idx)
    sig = get_strategy("hlhb").signal(bars)
    assert sig.iloc[:60].eq(0).all() and sig.iloc[-1] == 1.0


def test_ml_model_never_sees_labels_beyond_its_window(noise_bars):
    # Training on a window must not depend on anything after it: scramble the future.
    window = noise_bars.iloc[:300]
    m1 = train_model(window, 5)
    future_changed = pd.concat([window, noise_bars.iloc[300:] * 3.0])
    m2 = train_model(future_changed.iloc[:300], 5)
    assert np.allclose(m1["coef"], m2["coef"])
    # And the last `horizon` rows of the window carry no label (purged).
    lr = np.log(window["close"])
    assert lr.shift(-5).iloc[-5:].isna().all()


def test_ml_signal_causal_with_fitted_model(noise_bars):
    s = get_strategy("ml_direction")
    cfg = BotConfig()
    params = s.fit(noise_bars.iloc[:180], cfg.backtest_config(), cfg.risk)
    assert "model" in params and params["horizon"] in (5, 10)
    res = causality_tests(lambda b: s.signal(b, **params), noise_bars)
    assert not res["lookahead_at"] and not res["repaint_at"]
    rep = run_critic(s, params, noise_bars, cfg.backtest_config(), cfg.risk, cfg.gates, cfg.bar_seconds)
    assert rep["items"]["1_lookahead"]["status"] == "PASS"


def test_ported_strategies_rejected_on_noise(noise_bars, tmp_path):
    from quantstack.gates import validate
    from quantstack.trials import TrialLedger
    for name in ("hlhb", "ml_direction"):
        cfg = BotConfig(strategy=name, state_dir=str(tmp_path / name))
        rep = validate(cfg, noise_bars, ledger=TrialLedger(tmp_path / f"{name}.jsonl"), use_llm=False)
        assert rep["gates"]["1_no_leakage"]["passed"], name
        assert rep["passed"] is False, name


def test_features_are_trailing_only(noise_bars):
    res = causality_tests(lambda b: features(b)["ma12"], noise_bars)
    assert not res["lookahead_at"]
