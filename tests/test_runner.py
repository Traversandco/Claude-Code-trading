import json
from pathlib import Path

import pytest

from quantstack.broker import PaperBroker
from quantstack.gates import save_report, validate
from quantstack.runner import Runner, clear_halt
from quantstack.trials import TrialLedger


class Replay:
    """Feeds closed bars up to a moving cursor, and a clock just after that bar closes."""

    def __init__(self, bars, start, bar_seconds):
        self.bars, self.i, self.bs = bars, start, bar_seconds

    def feed(self):
        return self.bars.iloc[: self.i]

    def clock(self):
        return self.bars.index[self.i - 1].timestamp() + self.bs + 30

    def step(self):
        self.i += 1


@pytest.fixture
def validated(cfg, edge_bars, tmp_path):
    rep = validate(cfg, edge_bars, ledger=TrialLedger(tmp_path / "t.jsonl"), use_llm=False)
    assert rep["passed"]
    # Replay the tail of the same history through the live loop; stamp the report
    # as created at the replay start so the age check is meaningful.
    rep["created_at"] = edge_bars.index[1199].timestamp() + cfg.bar_seconds
    save_report(cfg, rep)
    return rep


def make(cfg, bars, start=1200):
    rp = Replay(bars, start, cfg.bar_seconds)
    broker = PaperBroker(Path(cfg.state_dir) / "paper.json", 10_000, cfg.fee_bps, cfg.slippage_bps)
    return Runner(cfg, broker, rp.feed, clock=rp.clock), rp, broker


def test_refuses_to_trade_without_validation(cfg, edge_bars):
    runner, rp, broker = make(cfg, edge_bars)
    assert runner.preflight()
    out = runner.tick()
    assert out["event"] == "halt" and "NOT_VALIDATED" in out["reason"]
    assert broker.state["fills"] == []


def test_trades_autonomously_after_validation(cfg, edge_bars, validated):
    cfg.gates.report_max_age_days = 365   # isolate trading mechanics from re-validation
    runner, rp, broker = make(cfg, edge_bars, start=900)   # an uptrend: long-only should engage
    assert runner.preflight() == []
    events = []
    for _ in range(150):
        events.append(runner.tick())
        rp.step()
    assert all(e["event"] == "bar" for e in events), [e for e in events if e["event"] != "bar"][:1]
    assert len(broker.state["fills"]) > 0
    # Sizing: never more than max_position_pct of equity (+ slippage/rounding).
    for e in events:
        assert e["exposure"] <= cfg.risk.max_position_pct + 1e-9
    # Refit happened on the walk-forward schedule.
    log = [json.loads(l) for l in Path(cfg.log_dir, "bot.jsonl").read_text().splitlines()]
    assert any(r["event"] == "refit" for r in log)


def test_same_bar_is_processed_once(cfg, edge_bars, validated):
    runner, rp, broker = make(cfg, edge_bars)
    assert runner.tick()["event"] == "bar"
    assert runner.tick()["event"] == "no_new_bar"


def test_stale_data_is_not_traded(cfg, edge_bars, validated):
    runner, rp, broker = make(cfg, edge_bars)
    runner.clock = lambda: rp.clock() + 5 * cfg.bar_seconds
    assert runner.tick()["reason"] == "stale_data"
    assert broker.state["fills"] == []


def test_kill_switch_flattens_and_halts_until_human_resumes(cfg, edge_bars, validated):
    cfg.gates.report_max_age_days = 365
    runner, rp, broker = make(cfg, edge_bars, start=900)
    for _ in range(40):
        runner.tick()
        rp.step()
    assert broker.position_units() > 0                   # holding something to flatten
    Path(cfg.state_dir, "KILL").write_text("x")
    out = runner.tick()
    assert out["event"] == "halt" and out["reason"] == "KILL_SWITCH"
    assert broker.position_units() == 0
    rp.step()
    assert runner.tick()["event"] == "halted"          # stays halted on later bars
    clear_halt(cfg)
    runner.state = runner._load_state()
    assert runner.tick()["event"] == "bar"


def test_max_drawdown_circuit_breaker(cfg, edge_bars, validated):
    runner, rp, broker = make(cfg, edge_bars)
    runner.tick()
    rp.step()
    broker.state["cash"] *= 0.5                         # simulate a disaster
    broker._save()
    out = runner.tick()
    assert out["event"] == "halt" and out["reason"].startswith(("DAILY_LOSS", "MAX_DRAWDOWN"))


def test_code_change_after_validation_blocks_trading(cfg, edge_bars, validated):
    rep = dict(validated, code_hash="0000")
    save_report(cfg, rep)
    runner, rp, broker = make(cfg, edge_bars)
    out = runner.tick()
    assert out["event"] == "halt" and "code changed" in out["reason"]


def test_expired_report_triggers_autonomous_revalidation(cfg, edge_bars, validated):
    runner, rp, broker = make(cfg, edge_bars)
    runner.report["created_at"] -= 60 * 86_400
    out = runner.tick()
    log = [json.loads(l) for l in Path(cfg.log_dir, "bot.jsonl").read_text().splitlines()]
    assert any(r["event"] == "revalidate" for r in log)
    assert out["event"] in ("bar", "halt")               # trades if it re-passes, halts if not


def test_failed_revalidation_halts_and_flattens(cfg, edge_bars, validated):
    # Replaying from bar 1200, the report expires at day 30; the fresh walk-forward on
    # data available then fails gate 3, so the bot must stop on its own.
    runner, rp, broker = make(cfg, edge_bars)
    last = None
    for _ in range(40):
        last = runner.tick()
        if last["event"] in ("halt", "halted"):
            break
        rp.step()
    assert last["event"] == "halt" and "NOT_VALIDATED" in last["reason"]
    assert broker.position_units() == 0


def test_order_notional_cap(cfg, edge_bars, validated):
    cfg.risk.max_order_notional = 50.0
    cfg.gates.report_max_age_days = 365
    runner, rp, broker = make(cfg, edge_bars, start=900)
    for _ in range(60):
        runner.tick()
        rp.step()
    assert broker.state["fills"]
    assert all(f["units"] * f["price"] <= 50.0 * 1.01 for f in broker.state["fills"])


def test_paper_broker_fees_and_persistence(tmp_path):
    b = PaperBroker(tmp_path / "p.json", 1_000, fee_bps=10, slippage_bps=10)
    b.set_price(100.0)
    f = b.market_order("buy", 5)
    assert f.price == pytest.approx(100.1) and f.fee == pytest.approx(5 * 100.1 * 0.001)
    b2 = PaperBroker(tmp_path / "p.json", 999_999, 10, 10)
    b2.set_price(100.0)
    assert b2.position_units() == 5 and b2.cash() < 1_000
    b2.market_order("sell", 100)                         # can't sell more than held
    assert b2.position_units() == 0
    assert b2.equity() < 1_000                           # round trip costs money


def test_retrying_a_bar_does_not_duplicate_equity(cfg, edge_bars, validated):
    cfg.gates.report_max_age_days = 365
    runner, rp, broker = make(cfg, edge_bars, start=900)
    runner.tick()
    rp.step()
    boom = {"n": 0}
    real = broker.market_order

    def flaky(side, units):
        boom["n"] += 1
        if boom["n"] == 1:
            raise ConnectionError("exchange hiccup")
        return real(side, units)
    broker.market_order = flaky
    for _ in range(30):                     # find a bar that trades, fail it once, retry
        try:
            runner.tick()
        except ConnectionError:
            runner.tick()
            break
        rp.step()
    bars_seen = [b for b, _ in runner.state["equity"]]
    assert len(bars_seen) == len(set(bars_seen))


def test_run_forever_refuses_to_start_unvalidated(cfg, edge_bars):
    runner, rp, broker = make(cfg, edge_bars)
    with pytest.raises(SystemExit) as e:
        runner.run_forever(sleep=lambda s: None)
    assert e.value.code == 2


def test_run_forever_stops_on_kill(cfg, edge_bars, validated):
    cfg.gates.report_max_age_days = 365
    runner, rp, broker = make(cfg, edge_bars, start=900)
    ticks = {"n": 0}

    def fake_sleep(_):
        ticks["n"] += 1
        rp.step()
        if ticks["n"] == 25:
            Path(cfg.state_dir, "KILL").write_text("x")
    runner.run_forever(sleep=fake_sleep)
    assert runner.state["halted"] and runner.state["halt_reason"] == "KILL_SWITCH"
    assert broker.position_units() == 0
