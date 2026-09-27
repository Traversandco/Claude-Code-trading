"""The autonomous loop. Wakes at each bar close and does, in order:

  kill switch → halt state → data freshness → validation report (auto re-validate
  when expired) → account circuit breakers → health check → scheduled refit →
  size → rebalance → persist.

Anything unexpected halts and flattens. A halted bot stays halted until a human
runs `quantstack resume`: you will not be objective at the moment it triggers.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import Broker, PaperBroker, is_exchange_demo
from .config import BotConfig
from .gates import check_report, load_report, save_report, validate
from .health import health_check
from .strategies import get_strategy
from .strategies.base import Strategy

MAX_CONSECUTIVE_ERRORS = 5


class Runner:
    def __init__(self, cfg: BotConfig, broker: Broker, feed, strategy: Strategy | None = None,
                 clock=time.time, auto_revalidate: bool = True, forward: bool = False):
        # forward=True: an unvalidated candidate on a simulated account, to build an
        # out-of-sample record. Only ever allowed with a PaperBroker.
        if forward and not (isinstance(broker, PaperBroker) or is_exchange_demo(broker)):
            raise ValueError("forward testing is paper/demo-only: unvalidated strategies never "
                             "touch a real-money account")
        self.forward = forward
        self.cfg = cfg
        self.broker = broker
        self.feed = feed                    # feed() -> DataFrame of CLOSED bars
        self.strategy = strategy or get_strategy(cfg.strategy)
        self.clock = clock
        self.auto_revalidate = auto_revalidate
        self.dir = Path(cfg.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "runner.json"
        self.kill_path = self.dir / "KILL"
        self.log_path = Path(cfg.log_dir) / "bot.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.report = load_report(cfg)

    # ---------- persistence ----------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"halted": False, "halt_reason": None, "last_bar": None, "params": None,
                "bars_since_refit": 0, "equity": [], "peak": None, "day": None,
                "day_start_equity": None, "errors": 0}

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def log(self, event: str, **kw) -> dict:
        rec = {"ts": pd.Timestamp(self.clock(), unit="s", tz="UTC").isoformat(), "event": event,
               "mode": self.cfg.mode, "symbol": self.cfg.symbol, **kw}
        line = json.dumps(rec, default=str)
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec

    # ---------- safety ----------
    def preflight(self) -> list[str]:
        return check_report(self.cfg, self.report, self.strategy, now=self.clock())

    def flatten(self) -> None:
        units = self.broker.position_units()
        if units > 0:
            fill = self.broker.market_order("sell", units)  # emergency exit ignores the order cap
            self.log("flatten", fill=fill.__dict__ if fill else None)

    def halt(self, reason: str, flatten: bool = True) -> dict:
        if flatten:
            try:
                self.flatten()
            except Exception as e:  # still mark halted even if the exchange is down
                self.log("flatten_failed", error=repr(e))
        self.state.update(halted=True, halt_reason=reason)
        self._save()
        return self.log("halt", reason=reason)

    def resume(self) -> None:
        clear_halt(self.cfg)
        self.state = self._load_state()

    # ---------- one bar ----------
    def tick(self) -> dict:
        cfg, risk = self.cfg, self.cfg.risk
        now = self.clock()

        if self.state["halted"]:
            return {"event": "halted", "reason": self.state["halt_reason"]}
        if self.kill_path.exists():
            return self.halt("KILL_SWITCH")

        bars = self.feed()
        if bars is None or bars.empty:
            return self.log("skip", reason="no_data")
        last = bars.index[-1]
        if self.state["last_bar"] == str(last):
            return {"event": "no_new_bar", "bar": str(last)}
        since_close = now - (last.timestamp() + cfg.bar_seconds)
        if since_close > 2 * cfg.bar_seconds:
            return self.log("skip", reason="stale_data", last_bar=str(last), seconds_old=since_close)

        # Validation is a precondition for every single order, not just startup.
        problems = [] if self.forward else check_report(cfg, self.report, self.strategy, now=now)
        if problems and self.auto_revalidate and self.report is not None and all("days old" in p for p in problems):
            self.log("revalidate", reason=problems)
            self.report = validate(cfg, bars, self.strategy)
            save_report(cfg, self.report)
            self.state["params"] = None      # adopt freshly fitted params
            problems = check_report(cfg, self.report, self.strategy, now=now)
        if problems:
            return self.halt("NOT_VALIDATED: " + "; ".join(problems))

        price = float(bars["close"].iloc[-1])
        if isinstance(self.broker, PaperBroker) and self.broker.price_fn is None:
            self.broker.set_price(price)
        else:
            price = self.broker.last_price()
        equity = self.broker.equity(price)

        # ----- account bookkeeping + circuit breakers (fixed before deploy) -----
        hist = self.state["equity"]
        if hist and hist[-1][0] == str(last):     # retrying this bar after an error
            hist[-1] = [str(last), equity]
        else:
            hist.append([str(last), equity])
        self.state["equity"] = hist[-5000:]
        self.state["peak"] = max(self.state["peak"] or equity, equity)
        day = str(last.date())
        if self.state["day"] != day:
            self.state.update(day=day, day_start_equity=equity)
        if equity < self.state["day_start_equity"] * (1 - risk.max_daily_loss_pct):
            return self.halt(f"DAILY_LOSS {equity / self.state['day_start_equity'] - 1:.2%}")
        if equity < self.state["peak"] * (1 - risk.max_drawdown_pct):
            return self.halt(f"MAX_DRAWDOWN {equity / self.state['peak'] - 1:.2%}")

        hc = None
        if not self.forward:   # a forward candidate has no backtest to decay from
            eq = pd.Series([e for _, e in self.state["equity"]], dtype=float)
            live_returns = np.log(eq / eq.shift(1)).dropna()
            hc = health_check(live_returns, self.report["backtest_metrics"], risk.health_window,
                              cfg.periods_per_year)
            if hc["action"] == "HALT":
                return self.halt(f"HEALTH {hc['alerts']}")

        # ----- the walk-forward protocol, continued live -----
        g = cfg.gates
        if self.state["params"] is None:
            if self.forward or self.report is None:
                self.state["params"] = self.strategy.fit(bars.iloc[-g.train_bars:], cfg.backtest_config(), risk)
            else:
                self.state["params"] = self.report["live_params"]
            self.state["bars_since_refit"] = 0
        elif self.state["bars_since_refit"] >= g.test_bars:
            self.state["params"] = self.strategy.fit(bars.iloc[-g.train_bars:], cfg.backtest_config(), risk)
            self.state["bars_since_refit"] = 0
            self.log("refit", params=self.state["params"])
        params = self.state["params"]

        exposure = float(self.strategy.exposure(bars, params, risk).iloc[-1])
        target_units = max(exposure, 0.0 if not risk.allow_short else -1.0) * equity / price
        current = self.broker.position_units()
        delta = target_units - current
        notional = abs(delta) * price
        threshold = max(risk.min_trade_notional, risk.rebalance_threshold * equity)
        closing = target_units <= 0 < current and current * price >= risk.min_trade_notional

        fill = None
        if notional >= threshold or closing:
            side = "buy" if delta > 0 else "sell"
            units = abs(delta)
            if units * price > risk.max_order_notional:
                units = risk.max_order_notional / price   # the rest goes next bar
            fill = self.broker.market_order(side, units)

        self.state["last_bar"] = str(last)
        self.state["bars_since_refit"] += 1
        self.state["errors"] = 0
        self._save()
        return self.log(
            "bar", bar=str(last), price=price, equity=round(equity, 2), exposure=round(exposure, 4),
            target_units=round(target_units, 8), current_units=round(current, 8),
            fill=fill.__dict__ if fill else None, params=params, health=hc,
        )

    # ---------- forever ----------
    def run_forever(self, delay_seconds: float = 15.0, sleep=time.sleep) -> None:
        problems = self.preflight()
        if problems and not (self.auto_revalidate and self.report is not None
                             and all("days old" in p for p in problems)):
            self.log("refuse_to_start", problems=problems)
            raise SystemExit(2)
        self.log("start", params=self.report.get("live_params") if self.report else None)
        while not self.state["halted"]:
            failed = False
            try:
                self.tick()
            except Exception as e:
                failed = True
                self.state["errors"] = self.state.get("errors", 0) + 1
                self._save()
                self.log("error", error=repr(e), trace=traceback.format_exc(limit=5),
                         consecutive=self.state["errors"])
                if self.state["errors"] >= MAX_CONSECUTIVE_ERRORS:
                    self.halt(f"{MAX_CONSECUTIVE_ERRORS} consecutive errors")
                    break
            if self.state["halted"]:
                break
            # Wake at least hourly rather than at computed bar boundaries: exchanges
            # don't all align candles to the epoch (weekly bars open on Monday).
            # Re-processing is impossible — the last_bar check dedupes.
            now = self.clock()
            step = min(self.cfg.bar_seconds, 3_600)
            wait = max((now // step + 1) * step + delay_seconds - now, 1.0)
            sleep(min(wait, 60.0) if failed else wait)   # retry the same bar soon after an error
        self.log("stopped", reason=self.state["halt_reason"])


def clear_halt(cfg: BotConfig) -> None:
    """Human-only: un-halt the bot and disarm the kill switch."""
    d = Path(cfg.state_dir)
    sp = d / "runner.json"
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
        state.update(halted=False, halt_reason=None, errors=0)
        sp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    (d / "KILL").unlink(missing_ok=True)
    log = Path(cfg.log_dir) / "bot.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now(tz="UTC").isoformat(), "event": "resume"}) + "\n")
