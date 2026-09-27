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
        strat = get_strategy(c["strategy"])     # fails loudly on typos
        symbol = c.get("symbol", cfg.symbol)
        h = code_hash(strat)
        cid = f"{strat.name}_{symbol.replace('/', '-')}_{cfg.timeframe}_{h[:8]}"
        out.append({"id": cid, "strategy": strat.name, "symbol": symbol,
                    "timeframe": cfg.timeframe, "code_hash": h})
    return out


def sync_registry(cfg: BotConfig, now: float | None = None) -> list[dict]:
    """Register new candidates, retire ones whose code changed or that were removed."""
    now = now if now is not None else time.time()
    reg = load_registry(cfg)
    by_id = {r["id"]: r for r in reg}
    want = {c["id"]: c for c in wanted_candidates(cfg)}
    for r in reg:
        if r["status"] == "active" and r["id"] not in want:
            same_slot = any(c["strategy"] == r["strategy"] and c["symbol"] == r["symbol"]
                            and c["timeframe"] == r["timeframe"] for c in want.values())
            r.update(status="retired", retired_at=now,
                     retired_reason="code changed" if same_slot else "removed from config")
    for cid, c in want.items():
        if cid not in by_id:
            reg.append({**c, "status": "active", "registered_at": now})
        elif by_id[cid]["status"] != "active":
            # Same code re-added later: its old record stays retired; start a fresh one.
            new = {**c, "id": f"{cid}_r{int(now)}", "status": "active", "registered_at": now}
            reg.append(new)
    _save_registry(cfg, reg)
    return reg


def candidate_config(cfg: BotConfig, cand: dict) -> BotConfig:
    c = copy.deepcopy(cfg)
    c.strategy, c.symbol = cand["strategy"], cand["symbol"]
    c.mode = "paper"
    c.state_dir = str(_dir(cfg) / cand["id"])
    c.log_dir = str(Path(cfg.log_dir) / "forward" / cand["id"])
    return c


def build_runners(cfg: BotConfig, feed_factory=None, clock=time.time) -> list[tuple[dict, Runner]]:
    from .data import load_history
    feed_factory = feed_factory or (lambda c: (lambda: load_history(c)))
    runners = []
    for cand in sync_registry(cfg, now=clock()):
        if cand["status"] != "active":
            continue
        c = candidate_config(cfg, cand)
        broker = PaperBroker(Path(c.state_dir) / "paper_account.json", c.initial_capital,
                             c.fee_bps, c.slippage_bps)
        runners.append((cand, Runner(c, broker, feed_factory(c), clock=clock, forward=True)))
    return runners


def run_forward(cfg: BotConfig, once: bool = False, feed_factory=None, clock=time.time,
                sleep=time.sleep, delay_seconds: float = 15.0) -> None:
    runners = build_runners(cfg, feed_factory, clock)
    print(json.dumps({"event": "forward_start", "candidates": [c["id"] for c, _ in runners]}), flush=True)
    while True:
        for cand, r in runners:
            try:
                r.tick()
            except Exception as e:     # one candidate failing never stops the others
                r.log("error", candidate=cand["id"], error=repr(e), trace=traceback.format_exc(limit=3))
        if once:
            return
        now = clock()
        step = min(cfg.bar_seconds, 3_600)
        sleep(max((now // step + 1) * step + delay_seconds - now, 1.0))


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
