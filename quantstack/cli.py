"""quantstack command line.

  demo         full pipeline on synthetic data (offline, no keys)
  fetch        download/refresh OHLCV history
  validate     run the three gates, write the report
  run          start the autonomous loop (paper | testnet | demo | live per config)
  forward      forward-test all candidates side by side on simulated accounts
  scoreboard   forward-test results, deflated for every candidate ever registered
  search       validate every strategy on real data, ranked; stops at the first that passes
  rounds       forward-test round history: each 24h result, what was retired and added
  weekly       the weekly review: top strategies with return, Sharpe, drawdown, trades, fees
  status       show bot state and validation status
  kill         create the kill switch (bot flattens and halts on next check)
  resume       clear a halt (human decision)
  hypothesize  ask Claude for a hypothesis with a named counterparty
  review       hostile risk-manager review of the current report
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import BotConfig, load_config


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Never overrides variables already set."""
    import os
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _cfg(args) -> BotConfig:
    _load_dotenv(str(Path(args.config).resolve().parent / ".env"))
    return load_config(args.config)


def _summary(report: dict) -> dict:
    g = report["gates"]
    return {
        "PASSED": report["passed"],
        "strategy": report["strategy"], "symbol": report["symbol"], "timeframe": report["timeframe"],
        "gate1_no_leakage": {"passed": g["1_no_leakage"]["passed"],
                             "failed_items": g["1_no_leakage"]["critic"]["failed"],
                             "llm_blocking": (g["1_no_leakage"]["llm_critic"] or {}).get("blocking")},
        "gate2_deflated_sharpe": {k: g["2_deflated_sharpe"][k] for k in
                                  ("passed", "deflated_sharpe", "sharpe_annualized",
                                   "expected_max_from_noise_annualized", "n_trials")},
        "gate3_walk_forward": {k: g["3_walk_forward"][k] for k in
                               ("passed", "positive_folds", "worst_fold", "mean_sharpe", "reasons")},
        "oos_metrics": report["backtest_metrics"],
        "regimes": {k: (v.get("sharpe") if isinstance(v, dict) else v) for k, v in report["regimes"].items()},
        "live_params": report["live_params"],
    }


def cmd_demo(args) -> int:
    from .data import synthetic_bars
    from .gates import validate
    from .trials import TrialLedger

    tmp = Path(tempfile.mkdtemp(prefix="quantstack-demo-"))
    cfg = BotConfig(state_dir=str(tmp / "state"), report_dir=str(tmp / "reports"),
                    log_dir=str(tmp / "logs"), data_dir=str(tmp / "data"))
    for label, strength in (("pure noise", 0.0), ("planted momentum edge", args.edge)):
        bars = synthetic_bars(n=args.bars, seed=args.seed, trend_strength=strength)
        cfg.strategy = "ts_momentum"
        ledger = TrialLedger(tmp / "state" / f"trials_{strength}.jsonl")
        report = validate(cfg, bars, ledger=ledger, use_llm=False)
        print(f"\n=== {label} (trend_strength={strength}) ===")
        print(json.dumps(_summary(report), indent=2, default=str))
    return 0


def cmd_fetch(args) -> int:
    from .data import load_history
    cfg = _cfg(args)
    bars = load_history(cfg)
    print(f"{len(bars)} closed bars {bars.index[0]} -> {bars.index[-1]}")
    return 0


def cmd_validate(args) -> int:
    from .data import load_history
    from .gates import save_report, validate
    cfg = _cfg(args)
    bars = load_history(cfg, refresh=not args.offline)
    report = validate(cfg, bars, use_llm=args.llm or None)
    path = save_report(cfg, report)
    print(json.dumps(_summary(report), indent=2, default=str))
    print(f"\nreport -> {path}")
    return 0 if report["passed"] else 1


def build_runner(cfg: BotConfig):
    from .broker import CcxtBroker, PaperBroker
    from .data import load_history, make_exchange
    from .runner import Runner

    feed = lambda: load_history(cfg)  # noqa: E731 — closed bars only, cached + incremental
    if cfg.mode == "paper":
        broker = PaperBroker(Path(cfg.state_dir) / "paper_account.json", cfg.initial_capital,
                             cfg.fee_bps, cfg.slippage_bps)
    else:
        ex = make_exchange(cfg.exchange, testnet=(cfg.mode == "testnet"),
                           demo=(cfg.mode == "demo"), auth=True)
        broker = CcxtBroker(ex, cfg.symbol)
    return Runner(cfg, broker, feed)


def cmd_run(args) -> int:
    cfg = _cfg(args)
    if cfg.mode == "live":
        print("*** LIVE MODE: real money. Gates, sizing and kill conditions are enforced. ***",
              file=sys.stderr)
    runner = build_runner(cfg)
    if args.once:
        problems = runner.preflight()
        if problems:
            print(json.dumps({"refuse": problems}, indent=2))
            return 2
        runner.tick()
        return 0
    runner.run_forever()
    return 0 if not runner.state["halted"] or runner.state["halt_reason"] == "KILL_SWITCH" else 3


def cmd_forward(args) -> int:
    from .forward import run_forward
    cfg = _cfg(args)
    print("forward test: simulated accounts only, no orders reach the exchange", file=sys.stderr)
    run_forward(cfg, once=args.once)
    return 0


def cmd_scoreboard(args) -> int:
    from .forward import format_scoreboard, scoreboard
    sb = scoreboard(_cfg(args))
    print(json.dumps(sb, indent=2, default=str) if args.json else format_scoreboard(sb))
    return 0


def search(cfg: BotConfig, names: list[str], target: float | None, offline: bool,
           loader=None) -> list[dict]:
    """Validate each strategy in turn (every run is logged as trials), rank them, and
    stop at the first that passes all three gates (and reaches `target`, if given)."""
    import copy

    from .data import load_history
    from .gates import save_report, validate
    loader = loader or (lambda c: load_history(c, refresh=not offline))
    cap = cfg.gates.max_plausible_sharpe
    if target is not None and target > cap:
        print(f"note: target Sharpe {target} is above the plausibility cap {cap}. Anything that "
              f"reaches it fails gate 1 as suspected leakage; it will be listed as FLAGGED.",
              file=sys.stderr)
    rows = []
    for name in names:
        c = copy.deepcopy(cfg)
        c.strategy = name
        try:
            bars = loader(c)
            rep = validate(c, bars)
        except Exception as e:
            rows.append({"strategy": name, "verdict": f"ERROR {e}"})
            continue
        save_report(c, rep)
        g = rep["gates"]
        sharpe = rep["backtest_metrics"].get("sharpe")
        failed = [k for k, v in g.items() if not v["passed"]]
        only_cap = failed == ["1_no_leakage"] and g["1_no_leakage"]["critic"]["failed"] == ["9_sharpe_sanity"]
        hit = rep["passed"] and (target is None or (sharpe or 0) >= target)
        verdict = ("PASS" if hit else
                   "FLAGGED: Sharpe above cap, treat as leakage" if only_cap else
                   "passed gates, below target" if rep["passed"] else
                   "fail " + ",".join(k.split("_", 1)[0] for k in failed))
        rows.append({"strategy": name, "sharpe": sharpe,
                     "dsr": g["2_deflated_sharpe"]["deflated_sharpe"],
                     "n_trials": g["2_deflated_sharpe"]["n_trials"],
                     "pos_folds": g["3_walk_forward"]["positive_folds"],
                     "worst_fold": g["3_walk_forward"]["worst_fold"], "verdict": verdict})
        print(f"  {name}: {verdict}", file=sys.stderr, flush=True)
        if hit:
            break
    return rows


def cmd_search(args) -> int:
    from .strategies import REGISTRY
    cfg = _cfg(args)
    names = args.strategies or list(REGISTRY)
    rows = search(cfg, names, args.target_sharpe, args.offline)
    rows.sort(key=lambda r: -(r.get("sharpe") or -99))
    print(f"search: {cfg.symbol} {cfg.timeframe}, {len(rows)} strategies validated "
          f"(target: {'pass all gates' if args.target_sharpe is None else f'pass + Sharpe >= {args.target_sharpe}'})")
    print(f"{'strategy':20s} {'sharpe':>7s} {'dsr':>6s} {'trials':>6s} {'folds+':>7s} {'worst':>6s}  verdict")
    for r in rows:
        print(f"{r['strategy']:20s} {str(r.get('sharpe', '-')):>7s} {str(r.get('dsr', '-')):>6s} "
              f"{str(r.get('n_trials', '-')):>6s} {str(r.get('pos_folds', '-')):>7s} "
              f"{str(r.get('worst_fold', '-')):>6s}  {r['verdict']}")
    return 0 if any(r["verdict"] == "PASS" for r in rows) else 1


def cmd_rounds(args) -> int:
    from .forward import format_rounds, load_rounds
    print(format_rounds(load_rounds(_cfg(args)), last=args.last))
    return 0


def cmd_weekly(args) -> int:
    from .forward import format_weekly, load_weekly
    print(format_weekly(load_weekly(_cfg(args), args.week)))
    return 0


def cmd_status(args) -> int:
    from .gates import check_report, load_report
    cfg = _cfg(args)
    sp = Path(cfg.state_dir) / "runner.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    report = load_report(cfg)
    out = {
        "mode": cfg.mode,
        "halted": state.get("halted"), "halt_reason": state.get("halt_reason"),
        "last_bar": state.get("last_bar"), "params": state.get("params"),
        "equity_last": state.get("equity", [[None, None]])[-1][1] if state.get("equity") else None,
        "kill_switch_armed": (Path(cfg.state_dir) / "KILL").exists(),
        "cleared_to_trade": not check_report(cfg, report),
        "validation_problems": check_report(cfg, report),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_kill(args) -> int:
    cfg = _cfg(args)
    p = Path(cfg.state_dir) / "KILL"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("kill requested\n", encoding="utf-8")
    print(f"kill switch armed at {p}; the bot flattens and halts on its next check")
    return 0


def cmd_resume(args) -> int:
    from .runner import clear_halt
    clear_halt(_cfg(args))
    print("halt cleared; the bot will trade again on its next bar if validation still passes")
    return 0


def cmd_hypothesize(args) -> int:
    from .llm import hypothesize
    cfg = _cfg(args)
    text = hypothesize(cfg.symbol, cfg.timeframe)
    out = Path("research/hypotheses")
    out.mkdir(parents=True, exist_ok=True)
    import time
    path = out / f"{int(time.time())}_{cfg.symbol.replace('/', '-')}_{cfg.timeframe}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved -> {path}  (implement as a Strategy subclass and register it to test)")
    return 0


def cmd_review(args) -> int:
    from .gates import load_report
    from .llm import adversarial_review
    from .strategies import get_strategy
    cfg = _cfg(args)
    report = load_report(cfg)
    if report is None:
        print("no report; run validate first")
        return 1
    print(adversarial_review(get_strategy(cfg.strategy), _summary(report)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quantstack", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo")
    d.add_argument("--bars", type=int, default=1500)
    d.add_argument("--seed", type=int, default=2)
    d.add_argument("--edge", type=float, default=0.5, help="strength of the planted persistent drift")
    d.set_defaults(fn=cmd_demo)

    for name, fn in (("fetch", cmd_fetch), ("validate", cmd_validate), ("run", cmd_run),
                     ("forward", cmd_forward), ("scoreboard", cmd_scoreboard), ("search", cmd_search),
                     ("rounds", cmd_rounds), ("weekly", cmd_weekly),
                     ("status", cmd_status), ("kill", cmd_kill), ("resume", cmd_resume),
                     ("hypothesize", cmd_hypothesize), ("review", cmd_review)):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", default="config.yaml")
        p.set_defaults(fn=fn)
        if name == "validate":
            p.add_argument("--llm", action="store_true", help="also run the Claude critic")
            p.add_argument("--offline", action="store_true", help="use cached data only")
        if name in ("run", "forward"):
            p.add_argument("--once", action="store_true", help="process one bar and exit")
        if name == "scoreboard":
            p.add_argument("--json", action="store_true")
        if name == "rounds":
            p.add_argument("--last", type=int, default=5)
        if name == "weekly":
            p.add_argument("--week", type=int, default=None)
        if name == "search":
            p.add_argument("strategies", nargs="*", help="default: every registered strategy")
            p.add_argument("--target-sharpe", type=float, default=None)
            p.add_argument("--offline", action="store_true", help="use cached data only")

    # Windows consoles and pipes (e.g. `| Set-Clipboard`) default to cp1252, which
    # cannot encode every character Claude or a strategy description may contain.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
