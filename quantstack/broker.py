"""Execution. PaperBroker simulates fills; CcxtBroker talks to a real exchange
(testnet via ccxt sandbox mode, or live). Spot, long-only by default.

Use a dedicated sub-account: the bot treats the whole base-asset balance as its position.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Fill:
    ts: float
    side: str
    units: float
    price: float
    fee: float
    order_id: str = ""

    @property
    def notional(self) -> float:
        return self.units * self.price


class Broker(ABC):
    @abstractmethod
    def last_price(self) -> float: ...

    @abstractmethod
    def position_units(self) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def market_order(self, side: str, units: float) -> Fill | None: ...

    def equity(self, price: float | None = None) -> float:
        price = price if price is not None else self.last_price()
        return self.cash() + self.position_units() * price


class PaperBroker(Broker):
    """Fills at the latest price ± slippage, charges fees, persists to disk."""

    def __init__(self, state_path: str | Path, initial_cash: float, fee_bps: float,
                 slippage_bps: float, price_fn=None):
        self.path = Path(state_path)
        self.fee = fee_bps / 1e4
        self.slip = slippage_bps / 1e4
        self.price_fn = price_fn
        self._price: float | None = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"cash": float(initial_cash), "units": 0.0, "fills": []}
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def set_price(self, price: float) -> None:
        self._price = float(price)

    def last_price(self) -> float:
        if self.price_fn is not None:
            return float(self.price_fn())
        if self._price is None:
            raise RuntimeError("paper broker has no price yet")
        return self._price

    def position_units(self) -> float:
        return float(self.state["units"])

    def cash(self) -> float:
        return float(self.state["cash"])

    def market_order(self, side: str, units: float) -> Fill | None:
        if units <= 0:
            return None
        ref = self.last_price()
        px = ref * (1 + self.slip) if side == "buy" else ref * (1 - self.slip)
        if side == "sell":
            units = min(units, self.position_units())
        notional = units * px
        fee = notional * self.fee
        if side == "buy":
            if notional + fee > self.cash():
                units = self.cash() / (px * (1 + self.fee))
                notional, fee = units * px, units * px * self.fee
            self.state["cash"] -= notional + fee
            self.state["units"] += units
        else:
            self.state["cash"] += notional - fee
            self.state["units"] -= units
        fill = Fill(time.time(), side, units, px, fee, f"paper-{len(self.state['fills'])}")
        self.state["fills"].append(asdict(fill))
        self._save()
        return fill


class CcxtBroker(Broker):
    def __init__(self, exchange, symbol: str):
        self.ex = exchange
        self.symbol = symbol
        self.ex.load_markets()
        self.market = self.ex.market(symbol)
        self.base, self.quote = self.market["base"], self.market["quote"]

    def last_price(self) -> float:
        return float(self.ex.fetch_ticker(self.symbol)["last"])

    def _balance(self) -> dict:
        return self.ex.fetch_balance()

    def position_units(self) -> float:
        return float(self._balance().get(self.base, {}).get("total") or 0.0)

    def cash(self) -> float:
        return float(self._balance().get(self.quote, {}).get("total") or 0.0)

    def market_order(self, side: str, units: float) -> Fill | None:
        amount = float(self.ex.amount_to_precision(self.symbol, units))
        limits = self.market.get("limits", {})
        min_amt = (limits.get("amount") or {}).get("min") or 0
        min_cost = (limits.get("cost") or {}).get("min") or 0
        price = self.last_price()
        if amount <= 0 or amount < min_amt or amount * price < min_cost:
            return None
        # Some venues (Bybit classic accounts, OKX in quote mode) size market BUYS in
        # quote currency and need a price to convert; others ignore it for market orders.
        order = self.ex.create_order(self.symbol, "market", side, amount,
                                     price if side == "buy" else None)
        try:
            order = self.ex.fetch_order(order["id"], self.symbol)
        except Exception:
            pass  # some exchanges don't support fetch_order right after a market fill
        filled = float(order.get("filled") or amount)
        avg = float(order.get("average") or order.get("price") or price)
        fee = float((order.get("fee") or {}).get("cost") or 0.0)
        return Fill(time.time(), side, filled, avg, fee, str(order.get("id", "")))


def is_exchange_demo(broker: Broker) -> bool:
    """True only if the broker's private API points at an exchange DEMO endpoint."""
    if not isinstance(broker, CcxtBroker):
        return False
    api = broker.ex.urls.get("api", {})
    hosts = api.values() if isinstance(api, dict) else [api]
    return bool(hosts) and all("api-demo" in str(h) for h in hosts)
