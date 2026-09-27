"""Forward testing: pre-registered candidates, each on its own simulated account,
trading live prices bar by bar. The data did not exist when the rules were
written, so this is the most honest out-of-sample test there is.

Rules that keep it honest:
- Paper only. Unvalidated strategies never touch an exchange or its keys.
- Candidates are registered with a timestamp and a code hash. Change the code
  and the old record is retired; a new candidate starts from zero.
- Nothing is ever deleted from the registry. Every candidate ever registered
  counts as a trial when the scoreboard deflates Sharpe ratios.
- Real money still requires `quantstack validate` to pass. The scoreboard
  informs; it does not unlock anything.
"""
from __future__ import annotations

import copy
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .config import BotConfig
from .gates import code_hash
from .runner import Runner
from .stats import deflated_sharpe_from_returns
from .strategies import REGISTRY, get_strategy
from .strategies.generated import GeneratedStrategy, random_spec, spec_id


def _dir(cfg: BotConfig) -> Path:
    return Path(cfg.state_dir) / "forward"


def _registry_path(cfg: BotConfig) -> Path:
    return _dir(cfg) / "registry.json"


def load_registry(cfg: BotConfig) -> list[dict]:
    p = _registry_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _save_registry(cfg: BotConfig, reg: list[dict]) -> None:
    p = _registry_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def wanted_candidates(cfg: BotConfig) -> list[dict]:
    raw = cfg.forward.candidates or [{"strategy": name} for name in REGISTRY]
    out = []
    for c in raw:
        c = {"strategy": c} if isinstance(c, str) else c
        strat = get_strategy(c["strategy"])     # fails loudly on typos
        symbol = c.get("symbol", cfg.symbol)
        tf = c.get("timeframe", strat.default_timeframe or cfg.timeframe)
        h = code_hash(strat)
        cid = f"{strat.name}_{symbol.replace('/', '-')}_{tf}_{h[:8]}"
        out.append({"id": cid, "strategy": strat.name, "symbol": symbol,
                    "timeframe": tf, "code_hash": h})
    return out


def _base(r: dict) -> str:
    return r.get("base_id", r["id"])


def sync_registry(cfg: BotConfig, now: float | None = None) -> list[dict]:
    """Register new config candidates and retire ones whose code changed or that were
    removed from the config. Candidates retired by rotation or a weekly review stay
    retired even though they are still listed in the config: the experiment decided."""
    now = now if now is not None else time.time()
    reg = load_registry(cfg)
    want = {c["id"]: c for c in wanted_candidates(cfg)}
    for r in reg:
        if r.get("origin", "config") != "config":
            continue           # rotation-born candidates are managed by rounds, not config
        if r["status"] == "active" and _base(r) not in want:
            same_slot = any(c["strategy"] == r["strategy"] and c["symbol"] == r["symbol"]
                            and c["timeframe"] == r["timeframe"] for c in want.values())
            r.update(status="retired", retired_at=now,
                     retired_reason="code changed" if same_slot else "removed from config")
    for cid, c in want.items():
        history = [r for r in reg if _base(r) == cid]
        if not history:
            reg.append({**c, "base_id": cid, "status": "active", "registered_at": now, "origin": "config"})
        elif any(r["status"] == "active" for r in history):
            continue
        elif history[-1].get("retired_reason") in ("removed from config", "code changed"):
            # Put back in the config after being taken out: a fresh record (a new trial).
            reg.append({**c, "id": f"{cid}_r{int(now)}", "base_id": cid, "status": "active",
                        "registered_at": now, "origin": "config"})
        # else: retired by rotation / weekly review -> stays retired
    _save_registry(cfg, reg)
    return reg


def strategy_for(cand: dict):
    return GeneratedStrategy(cand["spec"]) if cand.get("spec") else get_strategy(cand["strategy"])


def candidate_config(cfg: BotConfig, cand: dict) -> BotConfig:
    c = copy.deepcopy(cfg)
    c.strategy, c.symbol, c.timeframe = cand["strategy"], cand["symbol"], cand["timeframe"]
    c.mode = "paper"
    c.state_dir = str(_dir(cfg) / cand["id"])
    c.log_dir = str(Path(cfg.log_dir) / "forward" / cand["id"])
    return c


_TICK_CACHE: dict = {}


def _shared_feed(c: BotConfig, strat):
    """Candidates on the same market share one download per tick."""
    from .data import load_history
    key = (c.exchange, c.symbol, c.timeframe, "funding" in strat.requires)

    def feed():
        if key not in _TICK_CACHE:
            _TICK_CACHE[key] = load_history(c, strategy=strat)
        return _TICK_CACHE[key]
    return feed


def build_runners(cfg: BotConfig, feed_factory=None, clock=time.time,
                  sync: bool = True) -> list[tuple[dict, Runner]]:
    reg = sync_registry(cfg, now=clock()) if sync else load_registry(cfg)
    runners = []
    for cand in reg:
        if cand["status"] != "active":
            continue
        c = candidate_config(cfg, cand)
        strat = strategy_for(cand)
        broker = PaperBroker(Path(c.state_dir) / "paper_account.json", c.initial_capital,
                             c.fee_bps, c.slippage_bps)
        feed = feed_factory(c) if feed_factory else _shared_feed(c, strat)
        runners.append((cand, Runner(c, broker, feed, strategy=strat, clock=clock, forward=True)))
    return runners


def build_demo_runner(cfg: BotConfig, feed_factory=None, clock=time.time,
                      exchange_factory=None) -> tuple[dict, Runner] | None:
    """The one candidate that also trades the exchange demo account."""
    if not cfg.forward.demo_candidate:
        return None
    from .broker import CcxtBroker
    from .data import make_exchange
    want = cfg.forward.demo_candidate
    want = {"strategy": want} if isinstance(want, str) else dict(want)
    tmp = copy.deepcopy(cfg)
    tmp.forward.candidates = [want]
    cand = wanted_candidates(tmp)[0]
    c = candidate_config(cfg, cand)
    c.mode = "demo"
    c.state_dir = str(_dir(cfg) / f"{cand['id']}_exchange_demo")
    c.log_dir = str(Path(cfg.log_dir) / "forward" / f"{cand['id']}_exchange_demo")
    ex = (exchange_factory or (lambda: make_exchange(cfg.exchange, demo=True, auth=True)))()
    broker = CcxtBroker(ex, c.symbol)
    strat = strategy_for(cand)
    feed = feed_factory(c) if feed_factory else _shared_feed(c, strat)
    # Runner re-checks that this broker really is a demo endpoint.
    return {**cand, "id": f"{cand['id']}_exchange_demo"}, Runner(c, broker, feed, strategy=strat,
                                                                 clock=clock, forward=True)


def run_forward(cfg: BotConfig, once: bool = False, feed_factory=None, clock=time.time,
                sleep=time.sleep, delay_seconds: float = 15.0, exchange_factory=None) -> None:
    runners = build_runners(cfg, feed_factory, clock)
    demo = build_demo_runner(cfg, feed_factory, clock, exchange_factory)
    ensure_round(cfg, clock())
    print(json.dumps({"event": "forward_start", "candidates": [c["id"] for c, _ in runners]}), flush=True)
    while True:
        _TICK_CACHE.clear()
        for cand, r in runners + ([demo] if demo else []):
            try:
                r.tick()
            except Exception as e:     # one candidate failing never stops the others
                r.log("error", candidate=cand["id"], error=repr(e), trace=traceback.format_exc(limit=3))
        if round_due(cfg, clock()):
            summary = end_round(cfg, clock())
            print(json.dumps({"event": "round_end", **summary}, default=str), flush=True)
            runners = build_runners(cfg, feed_factory, clock, sync=False)
        if once:
            return
        now = clock()
        step = min(cfg.bar_seconds, 3_600)
        sleep(max((now // step + 1) * step + delay_seconds - now, 1.0))


# ---------------- rounds ----------------

def _round_path(cfg: BotConfig) -> Path:
    return _dir(cfg) / "round.json"


def _rounds_log(cfg: BotConfig) -> Path:
    return _dir(cfg) / "rounds.jsonl"


def _fill_count(cfg: BotConfig, cand: dict) -> int:
    return len(_fills(cfg, cand))


def _last_equity(cfg: BotConfig, cand: dict, default: float) -> float:
    eq = _equity(cfg, cand)
    return float(eq.iloc[-1]) if len(eq) else default


def _snapshot(cfg: BotConfig) -> dict:
    return {r["id"]: {"equity": _last_equity(cfg, r, cfg.initial_capital), "fills": _fill_count(cfg, r)}
            for r in load_registry(cfg) if r["status"] == "active"}


def ensure_round(cfg: BotConfig, now: float) -> dict:
    p = _round_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    state = {"round": 1, "started_at": now, "snap": _snapshot(cfg)}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def round_due(cfg: BotConfig, now: float) -> bool:
    state = ensure_round(cfg, now)
    return now >= state["started_at"] + cfg.forward.round_hours * 3_600


def new_hypothesis(cfg: BotConfig, round_no: int, taken: set[str], history: list[dict]) -> dict:
    """A fresh spec not tried before: from Claude if configured, else the built-in generator."""
    if cfg.forward.generator == "claude":
        try:
            from .llm import propose_spec
            spec = propose_spec(history, sorted(taken), cfg.symbol, cfg.timeframe)
            if spec_id(spec) not in taken:
                return {**spec, "source": "claude"}
        except Exception as e:     # never let an API problem stop the experiment
            print(json.dumps({"event": "claude_hypothesis_failed", "error": repr(e)}), flush=True)
    rng = random.Random(f"{cfg.forward.seed}:{round_no}:{len(taken)}")
    for _ in range(1000):
        spec = random_spec(rng)
        if spec_id(spec) not in taken:
            return spec
    raise RuntimeError("hypothesis space exhausted")


def end_round(cfg: BotConfig, now: float) -> dict:
    fc = cfg.forward
    state = ensure_round(cfg, now)
    reg = load_registry(cfg)
    rows = []
    for r in reg:
        if r["status"] != "active":
            continue
        snap = state["snap"].get(r["id"], {"equity": cfg.initial_capital, "fills": 0})
        eq_now = _last_equity(cfg, r, cfg.initial_capital)
        r["rounds"] = r.get("rounds", 0) + 1
        rows.append({
            "id": r["id"], "strategy": r["strategy"], "rounds": r["rounds"],
            "round_return_pct": round((eq_now / snap["equity"] - 1) * 100, 3),
            "round_trades": _fill_count(cfg, r) - snap["fills"],
            "total_return_pct": round((eq_now / cfg.initial_capital - 1) * 100, 3),
        })

    retired, added, weekly = None, [], None
    rounds_per_review = max(int(round(fc.review_days * 24 / fc.round_hours)), 1)
    is_review = fc.rotate and state["round"] % rounds_per_review == 0
    if is_review:
        _save_registry(cfg, reg)                       # scoreboard reads the saved registry
        weekly = weekly_review(cfg, state["round"] // rounds_per_review, now)
        reg = load_registry(cfg)
    if fc.rotate and not is_review:
        eligible = [x for x in rows if x["rounds"] >= fc.min_rounds]
        if eligible and len(rows) > 1:
            worst = min(eligible, key=lambda x: x["total_return_pct"])
            for r in reg:
                if r["id"] == worst["id"]:
                    r.update(status="retired", retired_at=now,
                             retired_reason=f"rotated out: lowest total return "
                                            f"({worst['total_return_pct']}%) after {worst['rounds']} rounds")
            retired = worst["id"]
    if fc.rotate:
        history = [json.loads(l) for l in _rounds_log(cfg).read_text(encoding="utf-8").splitlines()] \
            if _rounds_log(cfg).exists() else []
        taken = {r["strategy"] for r in reg}
        while sum(r["status"] == "active" for r in reg) < fc.max_active:
            spec = new_hypothesis(cfg, state["round"], taken, history[-20:] + [{"round": state["round"], "results": rows}])
            sid = spec_id(spec)
            taken.add(sid)
            strat = GeneratedStrategy(spec)
            reg.append({"id": f"{sid}_{cfg.symbol.replace('/', '-')}_{cfg.timeframe}", "strategy": sid,
                        "symbol": cfg.symbol, "timeframe": cfg.timeframe, "code_hash": code_hash(strat),
                        "spec": strat.spec, "status": "active", "registered_at": now,
                        "origin": "rotation", "round_born": state["round"] + 1})
            added.append({"strategy": sid, "family": spec["family"], "params": spec["params"],
                          "filters": spec["filters"], "source": spec.get("source", "generator")})
    _save_registry(cfg, reg)

    summary = {"round": state["round"], "started_at": state["started_at"], "ended_at": now,
               "results": sorted(rows, key=lambda x: -x["round_return_pct"]),
               "retired": retired, "added": added,
               "weekly_review": weekly["week"] if weekly else None}
    with _rounds_log(cfg).open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=str) + "\n")
    nxt = {"round": state["round"] + 1, "started_at": now, "snap": _snapshot(cfg)}
    _round_path(cfg).write_text(json.dumps(nxt, indent=2), encoding="utf-8")
    return summary


def _weekly_dir(cfg: BotConfig) -> Path:
    return _dir(cfg) / "weekly"


def weekly_review(cfg: BotConfig, week: int, now: float) -> dict:
    """Rank active candidates by live Sharpe; keep the top `keep_top`, retire the rest,
    and save the report. Candidates without enough data to have a Sharpe rank last."""
    sb = scoreboard(cfg)
    active = [r for r in sb["candidates"] if r["status"] == "active"]
    ranked = sorted(active, key=lambda r: (r.get("sharpe") is not None, r.get("sharpe") or 0,
                                           r.get("return_pct") or 0), reverse=True)
    keep = ranked[: cfg.forward.keep_top]
    drop = ranked[cfg.forward.keep_top:]
    reg = load_registry(cfg)
    drop_ids = {r["id"] for r in drop}
    specs = {r["id"]: r.get("spec") for r in reg}
    for r in reg:
        if r["id"] in drop_ids:
            r.update(status="retired", retired_at=now,
                     retired_reason=f"weekly review {week}: not in top {cfg.forward.keep_top} by Sharpe")
    _save_registry(cfg, reg)
    report = {
        "week": week, "at": now, "n_trials": sb["n_trials"], "keep_top": cfg.forward.keep_top,
        "top": [{**r, "rank": i + 1, "spec": specs.get(r["id"])} for i, r in enumerate(keep)],
        "retired": [{"id": r["id"], "strategy": r["strategy"], "sharpe": r.get("sharpe"),
                     "return_pct": r.get("return_pct")} for r in drop],
    }
    d = _weekly_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"week_{week:03d}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def load_weekly(cfg: BotConfig, week: int | None = None) -> dict | None:
    d = _weekly_dir(cfg)
    files = sorted(d.glob("week_*.json")) if d.exists() else []
    if week is not None:
        files = [f for f in files if f.name == f"week_{week:03d}.json"]
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def format_weekly(rep: dict | None) -> str:
    if not rep:
        return "no weekly review yet (the first happens after 7 daily rounds)"
    at = str(pd.Timestamp(rep["at"], unit="s", tz="UTC"))[:16]
    lines = [f"WEEK {rep['week']} REVIEW ({at} UTC): top {rep['keep_top']} kept, "
             f"{len(rep['retired'])} retired. Every candidate ever tried counts: n={rep['n_trials']}.",
             "",
             f"{'#':>2} {'strategy':22s} {'return%':>8s} {'sharpe':>7s} {'maxDD%':>7s} {'trades':>6s} "
             f"{'fees':>7s} {'bars':>6s}  verdict",
             "-" * 96]
    for r in rep["top"]:
        f = lambda k: "-" if r.get(k) is None else str(r.get(k))   # noqa: E731
        lines.append(f"{r['rank']:>2} {r['strategy'][:22]:22s} {f('return_pct'):>8s} {f('sharpe'):>7s} "
                     f"{f('max_dd_pct'):>7s} {f('trades'):>6s} {f('fees'):>7s} {f('bars'):>6s}  "
                     f"{r.get('verdict', '')}")
    lines.append("")
    for r in rep["top"]:
        sp = r.get("spec")
        desc = (f"{sp['family']} {sp['params']} {sp['filters'] or ''}" if sp else "hand-written strategy")
        lines.append(f"   {r['rank']}. {r['strategy']}: {desc}")
    lines += ["", "One week of live data is too short to separate skill from luck: a Sharpe",
              "measured over 7 days has a standard error of roughly 7 (annualized). The verdict",
              "column, not the rank, is what says whether anything is real."]
    return "\n".join(lines)


def load_rounds(cfg: BotConfig) -> list[dict]:
    p = _rounds_log(cfg)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def format_rounds(rounds: list[dict], last: int = 5) -> str:
    out = []
    for rd in rounds[-last:]:
        out.append(f"round {rd['round']}  ({str(pd.Timestamp(rd['started_at'], unit='s', tz='UTC'))[:16]}"
                   f" -> {str(pd.Timestamp(rd['ended_at'], unit='s', tz='UTC'))[:16]} UTC)")
        for x in rd["results"]:
            out.append(f"   {x['strategy'][:22]:22s} day {x['round_return_pct']:+7.3f}%  "
                       f"trades {x['round_trades']:3d}  total {x['total_return_pct']:+7.3f}%  "
                       f"rounds {x['rounds']}")
        if rd.get("retired"):
            out.append(f"   retired: {rd['retired']}")
        for a in rd.get("added", []):
            out.append(f"   new hypothesis ({a['source']}): {a['strategy']} {a['family']} "
                       f"{a['params']} {a['filters'] or ''}")
    return "\n".join(out) if out else "no completed rounds yet"


# ---------------- scoreboard ----------------

def _equity(cfg: BotConfig, cand: dict) -> pd.Series:
    p = _dir(cfg) / cand["id"] / "runner.json"
    if not p.exists():
        return pd.Series(dtype=float)
    hist = json.loads(p.read_text(encoding="utf-8")).get("equity", [])
    return pd.Series([e for _, e in hist], index=pd.to_datetime([t for t, _ in hist], utc=True), dtype=float)


def _fills(cfg: BotConfig, cand: dict) -> list[dict]:
    p = _dir(cfg) / cand["id"] / "paper_account.json"
    return json.loads(p.read_text(encoding="utf-8")).get("fills", []) if p.exists() else []


def scoreboard(cfg: BotConfig) -> dict:
    reg = load_registry(cfg)
    ppy = cfg.periods_per_year
    rows, rets = [], {}
    for cand in reg:
        eq = _equity(cfg, cand)
        r = np.log(eq / eq.shift(1)).dropna()
        rets[cand["id"]] = r
        fills = _fills(cfg, cand)
        row = {
            "id": cand["id"], "strategy": cand["strategy"], "symbol": cand["symbol"],
            "status": cand["status"],
            "since": pd.Timestamp(cand["registered_at"], unit="s", tz="UTC").strftime("%Y-%m-%d"),
            "bars": len(r),
            "return_pct": round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2) if len(eq) > 1 else 0.0,
            "trades": len(fills),
            "fees": round(sum(f["fee"] for f in fills), 2),
        }
        if len(r) >= 2 and r.std() > 0:
            sr_pp = float(r.mean() / r.std())
            peak = eq.cummax()
            row.update(sharpe=round(sr_pp * np.sqrt(ppy), 2),
                       max_dd_pct=round(float(((eq - peak) / peak).min()) * 100, 2),
                       # bars for |t| = 2 at the current Sharpe: how long until this means anything
                       bars_needed=int(np.ceil((2 / abs(sr_pp)) ** 2)) if sr_pp else None)
        else:
            row.update(sharpe=None, max_dd_pct=None, bars_needed=None)
        rows.append(row)

    n_trials = len(reg)
    sharpes = [float(r.mean() / r.std()) for r in rets.values() if len(r) >= 2 and r.std() > 0]
    for row in rows:
        r = rets[row["id"]]
        if row["bars"] < cfg.forward.min_bars:
            row["verdict"] = f"TOO EARLY ({row['bars']}/{cfg.forward.min_bars} bars)"
            continue
        d = deflated_sharpe_from_returns(r, n_trials, sharpes, ppy, cfg.gates.dsr_threshold)
        row["deflated_sharpe"] = d["deflated_sharpe"]
        row["verdict"] = ("PROMISING - now run `quantstack validate`" if d["verdict"] == "PASS"
                          else "NOT DISTINGUISHABLE FROM NOISE")
    return {"n_trials": n_trials, "periods_per_year": ppy, "candidates": rows}


def format_scoreboard(sb: dict) -> str:
    cols = [("strategy", 16), ("symbol", 9), ("status", 7), ("since", 10), ("bars", 5),
            ("return_pct", 8), ("sharpe", 6), ("max_dd_pct", 7), ("trades", 6), ("fees", 7)]
    head = " ".join(name[:w].ljust(w) for name, w in cols) + "  verdict"
    lines = [f"forward test: {len(sb['candidates'])} candidates registered "
             f"(all count as trials: n={sb['n_trials']})", head, "-" * len(head)]
    for row in sb["candidates"]:
        cells = " ".join(str("-" if row.get(k) is None else row.get(k))[:w].ljust(w) for k, w in cols)
        need = row.get("bars_needed")
        extra = f"  (~{need} bars to mean anything)" if need and need > row["bars"] else ""
        lines.append(f"{cells}  {row.get('verdict', '')}{extra}")
    return "\n".join(lines)
