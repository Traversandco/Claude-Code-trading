import json
import random
from types import SimpleNamespace

import pytest

from quantstack import forward as fw
from quantstack import llm
from quantstack.config import BotConfig, GateConfig
from quantstack.data import attach_funding, synthetic_bars
from quantstack.strategies.generated import (FAMILIES, GeneratedStrategy, random_spec, spec_id,
                                             validate_spec)
from tests.test_funding import synthetic_funding

M15 = 900


@pytest.fixture(scope="module")
def bars15():
    b = synthetic_bars(6000, seed=11, bar_seconds=M15)
    return attach_funding(b, synthetic_funding(b, 11), M15)


@pytest.fixture
def cfg15(tmp_path):
    c = BotConfig(timeframe="15m", state_dir=str(tmp_path / "state"), log_dir=str(tmp_path / "logs"),
                  report_dir=str(tmp_path / "reports"), gates=GateConfig(train_days=2, test_days=1))
    c.forward.candidates = ["session_breakout", "liquidation_rebound", "ts_momentum"]
    c.forward.rotate = True
    c.forward.round_hours = 6          # 24 bars per round
    c.forward.review_days = 1          # review every 4 rounds
    c.forward.max_active = 6
    c.forward.keep_top = 3
    return c


class Clock:
    def __init__(self, bars, i):
        self.bars, self.i = bars, i

    def now(self):
        return self.bars.index[self.i - 1].timestamp() + M15 + 30

    def feed_factory(self, c):
        return lambda: self.bars.iloc[: self.i]


def run(cfg, clk, n):
    for _ in range(n):
        fw.run_forward(cfg, once=True, feed_factory=clk.feed_factory, clock=clk.now)
        clk.i += 1


def test_spec_validation_rejects_anything_off_menu():
    with pytest.raises(ValueError):
        validate_spec({"family": "import os", "params": {}})
    with pytest.raises(ValueError):
        validate_spec({"family": "trend", "params": {"lookback_h": 7}})
    with pytest.raises(ValueError):
        validate_spec({"family": "trend", "params": {"lookback_h": 24, "evil": 1}})
    with pytest.raises(ValueError):
        validate_spec({"family": "trend", "filters": [{"type": "hours", "start": 16, "end": 8}]})
    ok = validate_spec({"family": "trend", "params": {"lookback_h": 24.0, "min_z": 0.5}})
    assert ok["params"]["lookback_h"] == 24 and isinstance(ok["params"]["lookback_h"], int)


def test_spec_ids_stable_and_distinct():
    rng = random.Random(0)
    specs = [random_spec(rng) for _ in range(200)]
    ids = {spec_id(s) for s in specs}
    assert len(ids) > 150                                   # mostly distinct
    assert spec_id(specs[0]) == spec_id(validate_spec(json.loads(json.dumps(specs[0]))))


def test_rounds_rotate_and_count_every_trial(cfg15, bars15):
    clk = Clock(bars15, 3000)
    run(cfg15, clk, 24 * 3 + 1)                            # three 6h rounds
    rounds = fw.load_rounds(cfg15)
    assert [r["round"] for r in rounds] == [1, 2, 3]
    reg = fw.load_registry(cfg15)
    active = [r for r in reg if r["status"] == "active"]
    assert len(active) == cfg15.forward.max_active          # topped up with new hypotheses
    assert sum(r["status"] == "retired" for r in reg) >= 2  # worst retired each round
    assert all(r["retired"] for r in rounds)
    new = [r for r in reg if r.get("origin") == "rotation"]
    assert new and all(r["spec"] for r in new)
    # retired candidates stay in the registry, so they keep counting as trials
    assert fw.scoreboard(cfg15)["n_trials"] == len(reg)
    # generated candidates actually trade on their own paper accounts
    states = [fw._dir(cfg15) / r["id"] / "runner.json" for r in new]
    assert any(p.exists() for p in states)
    text = fw.format_rounds(rounds)
    assert "round 3" in text and "new hypothesis" in text


def test_weekly_review_keeps_top_by_sharpe_and_reports(cfg15, bars15):
    clk = Clock(bars15, 3000)
    run(cfg15, clk, 24 * 4 + 1)                            # 4 rounds = one "week" here
    rep = fw.load_weekly(cfg15)
    assert rep and rep["week"] == 1
    assert len(rep["top"]) == cfg15.forward.keep_top
    sharpes = [r.get("sharpe") or 0 for r in rep["top"]]
    assert sharpes == sorted(sharpes, reverse=True)
    for r in rep["top"]:
        for k in ("return_pct", "sharpe", "max_dd_pct", "trades", "fees", "bars", "verdict"):
            assert k in r
    reg = {r["id"]: r for r in fw.load_registry(cfg15)}
    assert all(reg[r["id"]]["status"] == "active" for r in rep["top"])
    assert all(reg[r["id"]]["status"] == "retired" for r in rep["retired"])
    assert sum(r["status"] == "active" for r in reg.values()) == cfg15.forward.max_active
    text = fw.format_weekly(rep)
    assert "WEEK 1 REVIEW" in text and "sharpe" in text and "maxDD%" in text


def test_config_sync_never_retires_rotation_candidates(cfg15, bars15):
    clk = Clock(bars15, 3000)
    run(cfg15, clk, 25)
    born = [r["id"] for r in fw.load_registry(cfg15) if r.get("origin") == "rotation"]
    fw.sync_registry(cfg15, now=clk.now())
    reg = {r["id"]: r for r in fw.load_registry(cfg15)}
    assert all(reg[i]["status"] == "active" for i in born)


def test_generated_strategies_trade_through_runner_without_code_execution(cfg15, bars15):
    spec = {"family": "reversion", "params": {"window_h": 12, "z_entry": 1.5, "z_exit": 0.0}}
    g = GeneratedStrategy(spec)
    assert g.name.startswith("gen_") and g.fit(None, None, None) == {"_": 0}
    sig = g.signal(bars15)
    assert 0 < sig.mean() < 1


def test_claude_proposer_is_menu_bound_and_validated(monkeypatch):
    captured = {}

    def reply(payload):
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    good = {"family": "breakout", "params": [{"name": "entry_h", "value": 24}, {"name": "exit_h", "value": 4}],
            "filters": [{"type": "trend_filter", "values": [{"name": "days", "value": 50}]}],
            "mechanism": "stops above 24h highs"}

    def create(**kw):
        captured.update(kw)
        return reply(good)
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=create))))
    spec = llm.propose_spec([], ["gen_deadbeef"], "BTC/USDT", "15m")
    assert spec["family"] == "breakout" and spec["params"] == {"entry_h": 24, "exit_h": 4}
    fmt = captured["output_config"]["format"]["schema"]
    assert set(fmt["properties"]["family"]["enum"]) == set(FAMILIES)
    assert "gen_deadbeef" in captured["messages"][0]["content"]

    bad = dict(good, params=[{"name": "entry_h", "value": 999}])
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: reply(bad)))))
    with pytest.raises(ValueError):
        llm.propose_spec([], [], "BTC/USDT", "15m")


def test_claude_failure_falls_back_to_builtin(cfg15, monkeypatch):
    cfg15.forward.generator = "claude"

    def boom(*a, **k):
        raise RuntimeError("no API key")
    monkeypatch.setattr(llm, "propose_spec", boom)
    spec = fw.new_hypothesis(cfg15, 1, set(), [])
    assert spec["source"] == "generator"


def test_rotation_retired_config_candidate_is_not_resurrected(cfg15, bars15):
    clk = Clock(bars15, 3000)
    run(cfg15, clk, 24 * 3 + 5)
    reg = fw.load_registry(cfg15)
    ids = [r["id"] for r in reg]
    assert len(ids) == len(set(ids))
    # every config strategy appears at most once: nothing re-registered behind our back
    for name in cfg15.forward.candidates:
        assert sum(r["strategy"] == name for r in reg) == 1, name
    # trials = everything ever registered, and it grows only by rotation additions
    n_rot = sum(r.get("origin") == "rotation" for r in reg)
    assert fw.scoreboard(cfg15)["n_trials"] == len(cfg15.forward.candidates) + n_rot


def test_removed_then_readded_config_candidate_gets_one_fresh_record(cfg15):
    fw.sync_registry(cfg15, now=1)
    cfg15.forward.candidates = ["session_breakout"]
    fw.sync_registry(cfg15, now=2)
    cfg15.forward.candidates = ["session_breakout", "ts_momentum"]
    for t in (3, 4, 5):
        fw.sync_registry(cfg15, now=t)
    ts = [r for r in fw.load_registry(cfg15) if r["strategy"] == "ts_momentum"]
    assert [r["status"] for r in ts] == ["retired", "active"]
