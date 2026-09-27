"""Market data. Every series is UTC, indexed by bar OPEN time, closed bars only."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BotConfig

COLS = ["open", "high", "low", "close", "volume"]


def _to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts"] + COLS)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df.astype(float)


def closed_only(bars: pd.DataFrame, bar_seconds: int, now: float | None = None) -> pd.DataFrame:
    """Drop the still-forming candle. Trading on it is the live version of look-ahead."""
    now_ts = pd.Timestamp(now if now is not None else time.time(), unit="s", tz="UTC")
    return bars[bars.index + pd.Timedelta(seconds=bar_seconds) <= now_ts]


def make_exchange(exchange_id: str, testnet: bool = False, auth: bool = False, demo: bool = False):
    import os

    import ccxt
    params = {
        "enableRateLimit": True,
        # Spot bot: never let an exchange's default (Bybit: perpetual swaps) leak in.
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    }
    if auth:
        params["apiKey"] = os.environ["QUANTSTACK_API_KEY"]
        params["secret"] = os.environ["QUANTSTACK_API_SECRET"]
        if os.environ.get("QUANTSTACK_API_PASSWORD"):
            params["password"] = os.environ["QUANTSTACK_API_PASSWORD"]
    ex = getattr(ccxt, exchange_id)(params)
    if testnet:
        ex.set_sandbox_mode(True)
    if demo:
        # Demo trading (e.g. Bybit): real market prices, simulated account.
        ex.enable_demo_trading(True)
    return ex


def fetch_ohlcv(exchange, symbol: str, timeframe: str, n_bars: int, bar_seconds: int,
                now: float | None = None) -> pd.DataFrame:
    now = now if now is not None else time.time()
    since = int((now - (n_bars + 2) * bar_seconds) * 1000)
    rows: list[list] = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + bar_seconds * 1000
        if nxt <= since or nxt > now * 1000:
            break
        since = nxt
    if not rows:
        raise RuntimeError(f"no OHLCV returned for {symbol} {timeframe}")
    return closed_only(_to_frame(rows), bar_seconds, now).tail(n_bars)


def load_history(cfg: BotConfig, exchange=None, refresh: bool = True) -> pd.DataFrame:
    """Cached CSV + incremental refresh from the exchange."""
    path = Path(cfg.data_dir) / f"{cfg.exchange}_{cfg.symbol.replace('/', '-')}_{cfg.timeframe}.csv"
    cached = None
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8")
        cached.index = pd.to_datetime(cached.index, utc=True)
    if refresh:
        exchange = exchange or make_exchange(cfg.exchange)
        need = cfg.history_bars
        if cached is not None:
            missing = int((time.time() - cached.index[-1].timestamp()) // cfg.bar_seconds) + 5
            if missing >= cfg.history_bars:
                cached = None          # down too long: refetch rather than stitch a hole in
            else:
                need = missing
        fresh = fetch_ohlcv(exchange, cfg.symbol, cfg.timeframe, max(need, 2), cfg.bar_seconds)
        cached = fresh if cached is None else pd.concat([cached, fresh])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, encoding="utf-8")
    if cached is None:
        raise FileNotFoundError(f"no cached data at {path}")
    return cached.tail(cfg.history_bars)


def synthetic_bars(n: int = 1500, seed: int = 0, bar_seconds: int = 86_400,
                   start: str = "2020-01-01", trend_strength: float = 0.0) -> pd.DataFrame:
    """Regime-switching random walk with bull, bear and chop phases.

    With trend_strength == 0 the regimes are unpredictable noise: there is nothing
    to find, and the gates should reject every strategy. trend_strength > 0 adds a
    slowly-varying, persistent drift — a real, planted momentum edge — for testing
    that the gates can also pass something true.
    """
    rng = np.random.default_rng(seed)
    drift = {0: 0.0015, 1: -0.0015, 2: 0.0}
    vol = {0: 0.03, 1: 0.04, 2: 0.025}
    state, mu = 0, 0.0
    rets = np.empty(n)
    for i in range(n):
        if rng.random() < 1 / 120:
            state = int(rng.integers(0, 3))
        mu = 0.99 * mu + rng.normal(0, 0.0012) * trend_strength
        rets[i] = mu + rng.normal(drift[state] * (trend_strength == 0), vol[state])
    close = 20_000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.0003, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq=pd.Timedelta(seconds=bar_seconds), tz="UTC")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.lognormal(10, 1, n)}, index=idx)
