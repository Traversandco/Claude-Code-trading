import json
from pathlib import Path

import pytest

from quantstack import forward as fw
from quantstack.broker import CcxtBroker
from quantstack.data import attach_funding
from quantstack.runner import Runner
from tests.test_funding import synthetic_funding

DAY = 86_400


@pytest.fixture
def bars(edge_bars):
    return attach_funding(edge_bars, synthetic_funding(edge_bars), DAY)


class Clock:
    def __init__(self, bars, i):
        self.bars, self.i = bars, i

    def now(self):
        return self.bars.index[self.i - 1].timestamp() + DAY + 30

    def feed_factory(self, c):
        return lambda: self.bars.iloc[: self.i]


def test_forward_runs_all_candidates_without_validation_on_separate_accounts(cfg, bars):
    clk = Clock(bars, 900)
    for _ in range(40):
        fw.run_forward(cfg, once=True, feed_factory=clk.feed_factory, clock=clk.now)
        clk.i += 1
    reg = fw.load_registry(cfg)
    assert {r["strategy"] for r in reg} == {"ts_momentum", "rsi_reversion", "funding_crowding"}
    accounts = [Path(cfg.state_dir, "forward", r["id"], "paper_account.json") for r in reg]
    assert all(a.exists() for a in accounts)
    states = [json.loads(Path(cfg.state_dir, "forward", r["id"], "runner.json").read_text()) for r in reg]
    assert all(len(s["equity"]) == 40 for s in states)
    assert any(json.loads(a.read_text())["fills"] for a in accounts)      # something traded
    assert not Path(cfg.report_dir).exists()                              # no validation needed


def test_code_change_retires_record_and_counts_both(cfg, bars, monkeypatch):
    cfg.forward.candidates = [{"strategy": "ts_momentum"}]
    first = fw.sync_registry(cfg, now=1_000)
    assert len(first) == 1 and first[0]["status"] == "active"
    monkeypatch.setattr(fw, "code_hash", lambda s: "f" * 16)                # "edit" the strategy
    reg = fw.sync_registry(cfg, now=2_000)
    assert [r["status"] for r in reg] == ["retired", "active"]
    assert reg[0]["retired_reason"] == "code changed"
    cfg.forward.candidates = [{"strategy": "rsi_reversion"}]
    reg = fw.sync_registry(cfg, now=3_000)
    assert reg[1]["retired_reason"] == "removed from config"
    assert fw.scoreboard(cfg)["n_trials"] == 3                              # nothing forgotten


def test_scoreboard_too_early_then_verdict(cfg, bars):
    cfg.forward.candidates = [{"strategy": "ts_momentum"}, {"strategy": "funding_crowding"}]
    cfg.forward.min_bars = 30
    clk = Clock(bars, 900)
    fw.run_forward(cfg, once=True, feed_factory=clk.feed_factory, clock=clk.now)
    early = fw.scoreboard(cfg)
    assert all(r["verdict"].startswith("TOO EARLY") for r in early["candidates"])
    for _ in range(60):
        clk.i += 1
        fw.run_forward(cfg, once=True, feed_factory=clk.feed_factory, clock=clk.now)
    sb = fw.scoreboard(cfg)
    assert sb["n_trials"] == 2
    for r in sb["candidates"]:
        assert r["bars"] == 60 and not r["verdict"].startswith("TOO EARLY")
    text = fw.format_scoreboard(sb)
    assert "n=2" in text and "ts_momentum" in text


def test_forward_mode_refuses_real_exchange_broker(cfg, bars):
    class Ex:
        def load_markets(self): pass
        def market(self, s): return {"base": "BTC", "quote": "USDT", "limits": {}}
    with pytest.raises(ValueError, match="paper-only"):
        Runner(cfg, CcxtBroker(Ex(), "BTC/USDT"), lambda: bars, forward=True)


def test_forward_is_paper_even_when_config_says_live(cfg, bars):
    cfg.mode = "live"
    clk = Clock(bars, 900)
    runners = fw.build_runners(cfg, clk.feed_factory, clk.now)
    assert runners and all(r.cfg.mode == "paper" for _, r in runners)
    assert all(type(r.broker).__name__ == "PaperBroker" for _, r in runners)


def test_shared_cache_keeps_funding_when_other_strategy_refreshes(tmp_path, monkeypatch):
    import time as _time

    import numpy as np

    from quantstack.config import BotConfig
    from quantstack.data import load_history
    from tests.test_funding import FakeFundingExchange
    now = 1_700_000_000.0
    monkeypatch.setattr(_time, "time", lambda: now)
    day0 = int(now // DAY - 400) * DAY

    class Ex(FakeFundingExchange):
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
            rows = [[(day0 + i * DAY) * 1000, 100.0, 101.0, 99.0, 100.5, 1.0] for i in range(401)]
            return [r for r in rows if r[0] >= since][:limit]
    ex = Ex(day0 * 1000, 1203)
    f_cfg = BotConfig(strategy="funding_crowding", history_bars=300, data_dir=str(tmp_path))
    load_history(f_cfg, exchange=ex)
    load_history(BotConfig(strategy="ts_momentum", history_bars=300, data_dir=str(tmp_path)), exchange=ex)
    again = load_history(f_cfg, refresh=False)
    assert again["funding"].notna().mean() > 0.95
