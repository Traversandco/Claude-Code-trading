import pandas as pd

from quantstack.gates import check_report, code_hash, save_report, load_report, validate
from quantstack.leakage import causality_tests, run_critic
from quantstack.strategies import get_strategy
from quantstack.strategies.base import Strategy
from quantstack.trials import TrialLedger


class CenteredMA(Strategy):
    """Repaints: a centered rolling mean uses bars from the future."""
    name = "centered_ma"
    mechanism = "none — this is a test of the critic"
    param_grid = {"n": [20]}

    def signal(self, bars, n=20):
        ma = bars["close"].rolling(n, center=True).mean()
        return (bars["close"] > ma).astype(float)


class FullSampleZ(Strategy):
    """Leaks: z-scores against the mean/std of the WHOLE sample."""
    name = "full_sample_z"
    mechanism = "none — this is a test of the critic"
    param_grid = {"k": [1.0]}

    def signal(self, bars, k=1.0):
        r = bars["close"].pct_change()
        z = (r - r.mean()) / r.std()
        return (z < -k).astype(float)


class TomorrowPeek(Strategy):
    name = "peek"
    mechanism = "none — this is a test of the critic"
    param_grid = {"x": [1]}

    def signal(self, bars, x=1):
        return (bars["close"].shift(-1) > bars["close"]).astype(float)


def test_honest_strategies_pass_causality(noise_bars):
    for name in ("ts_momentum", "rsi_reversion"):
        s = get_strategy(name)
        p = s.param_combos()[0]
        res = causality_tests(lambda b: s.signal(b, **p), noise_bars)
        assert not res["lookahead_at"] and not res["repaint_at"], name


def test_critic_catches_centered_ma_full_sample_z_and_peek(noise_bars, cfg):
    for strat in (CenteredMA(), FullSampleZ(), TomorrowPeek()):
        rep = run_critic(strat, strat.param_combos()[0], noise_bars, cfg.backtest_config(),
                         cfg.risk, cfg.gates, cfg.bar_seconds)
        assert "1_lookahead" in rep["failed"], strat.name
        assert not rep["passed"]


def test_critic_flags_bad_data_alignment(noise_bars, cfg):
    bad = noise_bars.copy()
    bad.index = bad.index.tz_localize(None)       # naive timestamps
    s = get_strategy("ts_momentum")
    rep = run_critic(s, s.param_combos()[0], bad, cfg.backtest_config(), cfg.risk, cfg.gates, cfg.bar_seconds)
    assert "8_data_alignment" in rep["failed"]


def test_critic_flags_implausible_sharpe(noise_bars, cfg):
    s = get_strategy("ts_momentum")
    rep = run_critic(s, s.param_combos()[0], noise_bars, cfg.backtest_config(), cfg.risk, cfg.gates,
                     cfg.bar_seconds, oos_sharpe=3.4)
    assert "9_sharpe_sanity" in rep["failed"]


def test_noise_is_rejected(noise_bars, cfg, tmp_path):
    rep = validate(cfg, noise_bars, ledger=TrialLedger(tmp_path / "t.jsonl"), use_llm=False)
    assert rep["passed"] is False
    assert check_report(cfg, rep)            # non-empty = not cleared to trade


def test_planted_edge_passes_all_three_gates(edge_bars, cfg, tmp_path):
    rep = validate(cfg, edge_bars, ledger=TrialLedger(tmp_path / "t.jsonl"), use_llm=False)
    assert all(g["passed"] for g in rep["gates"].values()), {k: v.get("reasons") for k, v in rep["gates"].items()}
    save_report(cfg, rep)
    assert check_report(cfg, load_report(cfg)) == []


def test_trials_from_other_strategies_raise_the_bar(edge_bars, cfg, tmp_path):
    ledger = TrialLedger(tmp_path / "t.jsonl")
    for i in range(300):   # someone quietly tried 300 other things on this market
        ledger.record("other", {"i": i}, "binance:BTC/USDT:1d", 0.0, 1500)
    rep = validate(cfg, edge_bars, ledger=ledger, use_llm=False)
    base = validate(cfg, edge_bars, ledger=TrialLedger(tmp_path / "fresh.jsonl"), use_llm=False)
    assert rep["gates"]["2_deflated_sharpe"]["n_trials"] > 300
    assert (rep["gates"]["2_deflated_sharpe"]["deflated_sharpe"]
            < base["gates"]["2_deflated_sharpe"]["deflated_sharpe"])


def test_report_invalidated_by_mismatch_age_and_code_change(edge_bars, cfg, tmp_path):
    rep = validate(cfg, edge_bars, ledger=TrialLedger(tmp_path / "t.jsonl"), use_llm=False)
    assert check_report(cfg, rep) == []
    assert any("symbol" in p for p in check_report(cfg, {**rep, "symbol": "ETH/USDT"}))
    old = {**rep, "created_at": rep["created_at"] - 90 * 86_400}
    assert any("days old" in p for p in check_report(cfg, old))
    assert any("code changed" in p for p in check_report(cfg, {**rep, "code_hash": "deadbeef"}))
    assert rep["code_hash"] == code_hash(get_strategy("ts_momentum"))


def test_walk_forward_handles_folds_shorter_than_100_bars(noise_bars, cfg):
    from quantstack.walkforward import walk_forward
    wf = walk_forward(noise_bars, get_strategy("ts_momentum"), cfg.backtest_config(), cfg.risk, 180, 60)
    assert "sharpe" in wf["folds"].columns and len(wf["folds"]) == (1500 - 180) // 60
    assert isinstance(wf["oos"].index, pd.DatetimeIndex)
