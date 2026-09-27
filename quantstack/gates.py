"""Three hard gates before any strategy touches real money.

1. The critic finds no leakage
2. The deflated Sharpe clears the multiple-testing bar
3. Out-of-sample performance survives walk-forward

The runner refuses to trade without a passing report whose code hash matches
the code that is running, so editing a strategy silently un-validates it.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, sizing, walkforward
from .config import BotConfig
from .leakage import run_critic
from .metrics import per_period_sharpe
from .regimes import regime_split
from .stats import deflated_sharpe_from_returns, expected_max_sharpe
from .strategies import get_strategy
from .strategies import base as strategy_base
from .strategies.base import Strategy
from .trials import TrialLedger


def code_hash(strategy: Strategy) -> str:
    src = "".join(inspect.getsource(m) for m in (engine, sizing, walkforward, strategy_base))
    src += inspect.getsource(type(strategy))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def universe_key(cfg: BotConfig) -> str:
    return f"{cfg.exchange}:{cfg.symbol}:{cfg.timeframe}"


def report_path(cfg: BotConfig) -> Path:
    sym = cfg.symbol.replace("/", "-").replace(":", "-")
    return Path(cfg.report_dir) / f"{cfg.strategy}_{sym}_{cfg.timeframe}.json"


def validate(cfg: BotConfig, bars: pd.DataFrame, strategy: Strategy | None = None,
             ledger: TrialLedger | None = None, use_llm: bool | None = None) -> dict:
    strategy = strategy or get_strategy(cfg.strategy)
    bcfg, risk, g = cfg.backtest_config(), cfg.risk, cfg.gates
    universe = universe_key(cfg)
    ledger = ledger or TrialLedger(Path(cfg.state_dir) / "trials.jsonl")

    # Honest accounting: every grid point is a trial, whether or not it gets picked.
    # Full-sample runs are recorded here and NEVER used to select parameters.
    for params in strategy.param_combos():
        bt = engine.backtest(bars["close"], strategy.exposure(bars, params, risk), bcfg)
        ledger.record(strategy.name, params, universe, per_period_sharpe(bt["net"]), len(bt))

    # Gate 3 — walk-forward.
    wf = walkforward.walk_forward(bars, strategy, bcfg, risk, g.train_bars, g.test_bars)
    oos_m = wf["oos_metrics"]
    oos_sharpe = oos_m.get("sharpe", 0.0)
    gate3_reasons = []
    min_active = max(3, int(0.3 * len(wf["folds"])))
    if wf["n_active_folds"] < min_active:
        gate3_reasons.append(f"only {wf['n_active_folds']} folds with a position (need {min_active})")
    if wf["positive_frac"] < g.min_positive_fold_frac:
        gate3_reasons.append(f"positive folds {wf['positive_folds']} < {g.min_positive_fold_frac:.0%}")
    fold_se = np.sqrt(cfg.periods_per_year / g.test_bars)
    worst_floor = round(-g.worst_fold_tolerance * expected_max_sharpe(max(wf["n_active_folds"], 2), fold_se ** 2), 2)
    if wf["worst_fold"] < worst_floor:
        gate3_reasons.append(f"worst fold Sharpe {wf['worst_fold']} < {worst_floor} (zero-skill expectation)")
    if oos_sharpe < g.min_oos_sharpe:
        gate3_reasons.append(f"OOS Sharpe {oos_sharpe} < {g.min_oos_sharpe}")

    # Gate 1 — critic (deterministic, plus Claude if enabled).
    critic = run_critic(strategy, wf["live_params"], bars, bcfg, risk, g, cfg.bar_seconds, oos_sharpe)
    use_llm = g.llm_critic if use_llm is None else use_llm
    llm = None
    if use_llm:
        from .llm import llm_critic
        llm = llm_critic(strategy)
    gate1_ok = critic["passed"] and (llm is None or llm["passed"])

    # Gate 2 — deflated Sharpe on the stitched out-of-sample returns.
    dsr = deflated_sharpe_from_returns(
        wf["oos"]["net"], ledger.n_trials(universe), ledger.sharpes(universe),
        cfg.periods_per_year, g.dsr_threshold,
    )

    gates = {
        "1_no_leakage": {"passed": gate1_ok, "critic": critic, "llm_critic": llm},
        "2_deflated_sharpe": {"passed": dsr["verdict"] == "PASS", **dsr},
        "3_walk_forward": {
            "passed": not gate3_reasons, "reasons": gate3_reasons,
            "positive_folds": wf["positive_folds"], "worst_fold": wf["worst_fold"],
            "active_folds": f"{wf['n_active_folds']}/{len(wf['folds'])}",
            "worst_fold_floor": worst_floor,
            "mean_sharpe": wf["mean_sharpe"], "oos_metrics": oos_m,
        },
    }
    passed = all(v["passed"] for v in gates.values())
    folds = wf["folds"].assign(start=lambda d: d["start"].astype(str)).to_dict("records")
    return {
        "passed": passed,
        "strategy": strategy.name,
        "mechanism": strategy.mechanism,
        "exchange": cfg.exchange, "symbol": cfg.symbol, "timeframe": cfg.timeframe,
        "code_hash": code_hash(strategy),
        "created_at": time.time(),
        "data": {"start": str(bars.index[0]), "end": str(bars.index[-1]), "n_bars": len(bars)},
        "costs": {"fee_bps": cfg.fee_bps, "slippage_bps": cfg.slippage_bps},
        "live_params": wf["live_params"],
        "backtest_metrics": oos_m,   # what the live health check compares against
        "gates": gates,
        "regimes": regime_split(bars["close"], wf["oos"]["net"], bcfg),
        "folds": folds,
    }


def save_report(cfg: BotConfig, report: dict) -> Path:
    p = report_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p


def load_report(cfg: BotConfig) -> dict | None:
    p = report_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check_report(cfg: BotConfig, report: dict | None, strategy: Strategy | None = None,
                 now: float | None = None) -> list[str]:
    """Empty list = cleared to trade. Otherwise, every reason it is not."""
    if report is None:
        return ["no validation report — run `quantstack validate` first"]
    strategy = strategy or get_strategy(cfg.strategy)
    problems = []
    if not report.get("passed"):
        failed = [k for k, v in report.get("gates", {}).items() if not v.get("passed")]
        problems.append(f"validation failed gates: {failed}")
    for k in ("strategy", "exchange", "symbol", "timeframe"):
        want = cfg.strategy if k == "strategy" else getattr(cfg, k)
        if report.get(k) != want:
            problems.append(f"report {k}={report.get(k)!r} does not match config {want!r}")
    if report.get("code_hash") != code_hash(strategy):
        problems.append("strategy/engine code changed since validation — re-validate")
    age_days = ((now or time.time()) - report.get("created_at", 0)) / 86_400
    if age_days > cfg.gates.report_max_age_days:
        problems.append(f"report is {age_days:.0f} days old (max {cfg.gates.report_max_age_days})")
    return problems
