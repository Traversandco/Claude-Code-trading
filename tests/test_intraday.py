import numpy as np
import pandas as pd

from quantstack.config import BotConfig, load_config
from quantstack.data import synthetic_bars
from quantstack.leakage import causality_tests
from quantstack.regimes import regime_fractions, regime_labels
from quantstack.strategies import get_strategy
from quantstack.trials import TrialLedger

M15 = 900


def _day(prices, start="2024-03-01"):
    idx = pd.date_range(start, periods=len(prices), freq="15min", tz="UTC")
    c = np.asarray(prices, float)
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c,
                         "volume": np.full(len(c), 100.0)}, index=idx)


def test_session_breakout_enters_after_range_and_exits_at_exit_hour():
    # 00:00-08:00 range around 100, then a breakout to 102 at 10:00.
    p = np.r_[np.full(32, 100.0), np.full(8, 100.0), np.full(56, 102.0)]
    bars = _day(p)
    sig = get_strategy("session_breakout").signal(bars, exit_hour=16)
    hours = bars.index.hour
    assert sig[hours < 8].eq(0).all()                       # never inside the range window
    assert sig[(hours >= 10) & (hours < 16)].eq(1).all()    # in after the break
    assert sig[hours >= 16].eq(0).all()                     # flat at the exit hour


def test_session_breakout_is_causal_and_flat_overnight():
    bars = synthetic_bars(20_000, seed=5, bar_seconds=M15)
    s = get_strategy("session_breakout")
    for p in s.param_combos():
        res = causality_tests(lambda b: s.signal(b, **p), bars)
        assert not res["lookahead_at"] and not res["repaint_at"]
    sig = s.signal(bars, exit_hour=23)
    assert sig[bars.index.hour < 8].eq(0).all()


def test_liquidation_rebound_fires_on_a_planted_cascade():
    rng = np.random.default_rng(0)
    n = 2000
    r = rng.normal(0, 0.002, n)
    r[1500] = -0.05                                         # the cascade bar
    c = 100 * np.exp(np.cumsum(r))
    bars = _day(c)
    bars.iloc[1500, bars.columns.get_loc("volume")] = 2000  # 20x volume
    s = get_strategy("liquidation_rebound")
    sig = s.signal(bars, sigma=4.0, hold=4)
    assert sig.iloc[1500:1504].eq(1).all() and sig.iloc[1504:].eq(0).all()
    assert sig.iloc[:1500].eq(0).all()
    res = causality_tests(lambda b: s.signal(b, sigma=4.0, hold=4), bars)
    assert not res["lookahead_at"] and not res["repaint_at"]


def test_day_based_windows_scale_with_timeframe():
    cfg = load_config("config.intraday.example.yaml")
    assert cfg.timeframe == "15m" and cfg.train_bars == 2880 and cfg.test_bars == 672
    assert BotConfig().train_bars == 180                     # bar-based default unchanged
    assert "hlhb" not in str(cfg.forward.candidates)


def test_regimes_use_200_days_on_intraday_bars():
    bars = synthetic_bars(35_040 * 2, seed=6, bar_seconds=M15)
    lab = regime_labels(bars["close"])
    first = lab[lab != "warmup"].index[0]
    assert (first - bars.index[0]).days >= 200               # 200 DAYS of warm-up, not 200 bars
    assert sum(regime_fractions(bars["close"]).values()) > 0.99


def test_regime_labels_are_causal():
    bars = synthetic_bars(900, seed=7)
    res = causality_tests(lambda b: (regime_labels(b["close"]) == "bull").astype(float), bars)
    assert not res["lookahead_at"] and not res["repaint_at"]


def test_trials_count_across_timeframes(tmp_path):
    L = TrialLedger(tmp_path / "t.jsonl")
    L.record("a", {}, "bybit:BTC/USDT:1d", 0.01, 100)
    L.record("a", {}, "bybit:BTC/USDT:15m", 0.01, 100)
    L.record("a", {}, "bybit:ETH/USDT:15m", 0.01, 100)
    assert L.n_trials_prefix("bybit:BTC/USDT:") == 2
