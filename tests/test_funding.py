import numpy as np
import pandas as pd

from quantstack.config import BotConfig
from quantstack.data import attach_funding, fetch_funding, load_history, perp_symbol, synthetic_bars
from quantstack.leakage import causality_tests, run_critic
from quantstack.strategies import get_strategy
from quantstack.strategies.funding_crowding import BASELINE_8H

DAY = 86_400
H8 = 8 * 3_600


def synthetic_funding(bars: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Funding every 8h that tracks recent returns (crowding), like the real thing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(bars.index[0], bars.index[-1] + pd.Timedelta(hours=16), freq="8h", tz="UTC")
    r = np.log(bars["close"]).diff().rolling(14, min_periods=1).mean().reindex(idx, method="ffill").fillna(0)
    return pd.Series(BASELINE_8H + 0.05 * r.to_numpy() + rng.normal(0, 0.00005, len(idx)), index=idx)


def with_funding(n=1500, seed=0):
    bars = synthetic_bars(n, seed=seed)
    return attach_funding(bars, synthetic_funding(bars, seed), DAY)


def test_funding_lands_in_the_bar_it_settled_in():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    f = pd.Series([0.1, 0.2, 0.3, 0.9],
                  index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00",
                                        "2024-01-02 00:00"], utc=True))
    out = attach_funding(bars, f, DAY)
    assert np.isclose(out["funding"].iloc[0], 0.2)       # only day-1 settlements
    assert np.isclose(out["funding"].iloc[1], 0.9)       # next midnight belongs to next bar
    assert np.isnan(out["funding"].iloc[2])


def test_perp_symbol_derivation():
    assert perp_symbol(BotConfig(symbol="BTC/USDT")) == "BTC/USDT:USDT"
    assert perp_symbol(BotConfig(symbol="ETH/USDT", perp_symbol="ETH/USDC:USDC")) == "ETH/USDC:USDC"


class FakeFundingExchange:
    def __init__(self, start_ms, n, listed_after_ms=0):
        self.rows = [{"timestamp": start_ms + i * H8 * 1000, "fundingRate": 0.0001 * (1 + i % 3)}
                     for i in range(n) if start_ms + i * H8 * 1000 >= listed_after_ms]
        self.calls = 0

    def fetch_funding_rate_history(self, symbol, since=None, limit=200):
        self.calls += 1
        end = since + limit * H8 * 1000
        return [r for r in self.rows if since <= r["timestamp"] < end][:limit]


def test_fetch_funding_paginates_and_skips_prelisting_gap():
    start = 1_600_000_000_000
    ex = FakeFundingExchange(start, 1000, listed_after_ms=start + 300 * H8 * 1000)
    now = (start + 1000 * H8 * 1000) / 1000
    s = fetch_funding(ex, "BTC/USDT:USDT", start, now=now)
    assert len(s) == 700 and ex.calls >= 4
    assert s.index.is_monotonic_increasing and str(s.index.tz) == "UTC"


def test_funding_strategy_is_causal_including_funding_column():
    bars = with_funding()
    s = get_strategy("funding_crowding")
    for p in s.param_combos():
        res = causality_tests(lambda b: s.signal(b, **p), bars)
        assert not res["lookahead_at"] and not res["repaint_at"], p


def test_critic_catches_strategy_peeking_at_future_funding():
    from quantstack.strategies.base import Strategy

    class Peek(Strategy):
        name, mechanism, param_grid = "peek_f", "test", {"x": [1]}
        requires = ("funding",)

        def signal(self, bars, x=1):
            return (bars["funding"].shift(-1) < 0).astype(float)
    cfg = BotConfig()
    rep = run_critic(Peek(), {"x": 1}, with_funding(), cfg.backtest_config(), cfg.risk, cfg.gates, DAY)
    assert "1_lookahead" in rep["failed"]


def test_rules_follow_the_mechanism():
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    up = pd.DataFrame({"close": np.linspace(100, 200, 300)}, index=idx)
    s = get_strategy("funding_crowding")
    neutral = s.signal(up.assign(funding=BASELINE_8H), lookback=50, crowded=0.0003)
    assert neutral.iloc[-1] == 1.0                                   # uptrend, normal funding: long
    crowded = s.signal(up.assign(funding=0.001), lookback=50, crowded=0.0003)
    assert crowded.iloc[-1] == 0.0                                   # longs crowded: flat
    down = pd.DataFrame({"close": np.linspace(200, 100, 300)}, index=idx)
    assert s.signal(down.assign(funding=BASELINE_8H), lookback=50).iloc[-1] == 0.0
    assert s.signal(down.assign(funding=-0.0002), lookback=50).iloc[-1] == 1.0   # shorts crowded
    assert s.signal(up, lookback=50).iloc[-1] == 1.0                 # no funding column: trend only
    assert s.signal(up.assign(funding=0.001), lookback=50).iloc[:50].eq(0).all()  # warm-up flat


def test_full_validation_runs_on_funding_strategy(tmp_path):
    from quantstack.gates import validate
    from quantstack.trials import TrialLedger
    cfg = BotConfig(strategy="funding_crowding", state_dir=str(tmp_path))
    rep = validate(cfg, with_funding(), ledger=TrialLedger(tmp_path / "t.jsonl"), use_llm=False)
    assert rep["gates"]["1_no_leakage"]["passed"]
    assert rep["passed"] is False            # no real edge in synthetic noise


def test_load_history_fetches_and_caches_funding(tmp_path, monkeypatch):
    import time as _time
    now = 1_700_000_000.0
    monkeypatch.setattr(_time, "time", lambda: now)
    day0 = int(now // DAY - 400) * DAY

    class Ex(FakeFundingExchange):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            rows = [[(day0 + i * DAY) * 1000, 100.0, 101.0, 99.0, 100.5, 1.0] for i in range(401)]
            return [r for r in rows if r[0] >= since][:limit]

    ex = Ex(day0 * 1000, 1203)
    cfg = BotConfig(strategy="funding_crowding", history_bars=300, data_dir=str(tmp_path))
    bars = load_history(cfg, exchange=ex)
    assert "funding" in bars and bars["funding"].notna().mean() > 0.95
    offline = load_history(cfg, refresh=False)                      # cached, incl. funding
    assert np.allclose(offline["funding"].dropna(), bars["funding"].dropna())
