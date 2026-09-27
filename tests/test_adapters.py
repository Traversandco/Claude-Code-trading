"""External adapters, tested against fakes (no network, no keys)."""
import json
from types import SimpleNamespace

import pandas as pd

from quantstack import llm
from quantstack.broker import CcxtBroker
from quantstack.data import closed_only, fetch_ohlcv
from quantstack.strategies import get_strategy

DAY = 86_400


class FakeExchange:
    def __init__(self, n=2500, start=1_600_000_000):
        self.rows = [[(start + i * DAY) * 1000, 100 + i, 101 + i, 99 + i, 100.5 + i, 10.0] for i in range(n)]
        self.calls = 0
        self.orders = []
        self.balance = {"BTC": {"total": 0.0}, "USDT": {"total": 10_000.0}}

    # data
    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        self.calls += 1
        return [r for r in self.rows if r[0] >= since][:limit]

    # trading
    def load_markets(self):
        pass

    def market(self, symbol):
        return {"base": "BTC", "quote": "USDT",
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 10.0}}}

    def fetch_ticker(self, symbol):
        return {"last": 20_000.0}

    def fetch_balance(self):
        return self.balance

    def amount_to_precision(self, symbol, amount):
        return f"{int(amount * 1e4) / 1e4:.4f}"

    def create_order(self, symbol, type_, side, amount):
        self.orders.append((type_, side, amount))
        return {"id": str(len(self.orders))}

    def fetch_order(self, oid, symbol):
        type_, side, amount = self.orders[int(oid) - 1]
        return {"id": oid, "filled": amount, "average": 20_010.0, "fee": {"cost": 1.23}}


def test_closed_only_drops_forming_candle():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    bars = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    now = (idx[-1] + pd.Timedelta(hours=5)).timestamp()     # last candle still open
    assert len(closed_only(bars, DAY, now)) == 2


def test_fetch_ohlcv_paginates_and_returns_closed_utc_bars():
    ex = FakeExchange()
    now = ex.rows[-1][0] / 1000 + DAY / 2                     # mid-way through the last bar
    df = fetch_ohlcv(ex, "BTC/USDT", "1d", 1500, DAY, now=now)
    assert ex.calls >= 2                                     # needed more than one page
    assert len(df) == 1500 and str(df.index.tz) == "UTC"
    assert df.index[-1].timestamp() * 1000 == ex.rows[-2][0]  # forming bar excluded
    assert df.index.is_monotonic_increasing and df.index.is_unique


def test_ccxt_broker_respects_precision_and_minimums():
    ex = FakeExchange()
    b = CcxtBroker(ex, "BTC/USDT")
    assert b.market_order("buy", 0.0001) is None             # below min amount
    assert b.market_order("buy", 0.0004) is None             # 0.0004*20k = $8 < $10 min cost
    f = b.market_order("buy", 0.012345)
    assert ex.orders[-1] == ("market", "buy", 0.0123)        # precision applied
    assert f.price == 20_010.0 and f.fee == 1.23 and f.units == 0.0123
    assert b.equity(20_000.0) == 10_000.0


def _fake_client(payload):
    resp = SimpleNamespace(stop_reason="end_turn", stop_details=None,
                           content=[SimpleNamespace(type="text", text=json.dumps(payload))])
    captured = {}

    def create(**kw):
        captured.update(kw)
        return resp
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))), captured


def _items(status_by_num):
    return {"items": [{"number": n, "name": f"item{n}", "status": s, "quoted_lines": ["x"],
                       "explanation": "y"} for n, s in status_by_num.items()]}


def test_llm_critic_blocks_on_present_leakage(monkeypatch):
    client, captured = _fake_client(_items({1: "PRESENT", 2: "PRESENT", 3: "ABSENT", 4: "ABSENT",
                                            5: "ABSENT", 6: "ABSENT", 7: "ABSENT", 8: "ABSENT"}))
    monkeypatch.setattr(llm, "_client", lambda: client)
    out = llm.llm_critic(get_strategy("ts_momentum"))
    assert not out["passed"] and out["blocking"] == ["1. item1"]   # 2 is data-level, not blocking
    assert captured["model"] == "claude-opus-5"
    assert captured["fallbacks"] == "default"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert "Do not summarize. Quote lines." in captured["messages"][0]["content"]
    assert "signal.shift(1)" in captured["messages"][0]["content"]  # the engine source was sent


def test_llm_critic_passes_when_all_absent(monkeypatch):
    client, _ = _fake_client(_items({n: "ABSENT" for n in range(1, 9)}))
    monkeypatch.setattr(llm, "_client", lambda: client)
    assert llm.llm_critic(get_strategy("rsi_reversion"))["passed"]


def test_llm_refusal_raises(monkeypatch):
    resp = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="x"), content=[])
    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: resp)))
    monkeypatch.setattr(llm, "_client", lambda: client)
    try:
        llm.hypothesize("BTC/USDT", "4h")
    except RuntimeError as e:
        assert "declined" in str(e)
    else:
        raise AssertionError("refusal must not be treated as an answer")


def test_cli_demo_runs_offline(capsys):
    from quantstack.cli import main
    assert main(["demo", "--bars", "900"]) == 0
    out = capsys.readouterr().out
    assert "pure noise" in out and "planted momentum edge" in out
