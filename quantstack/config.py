"""Configuration.

`Config` is the backtest config from the article. `BotConfig` is everything the
autonomous runner needs, loaded from YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Bars per year for a market that trades 24/7.
PERIODS_PER_YEAR = {
    "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460, "8h": 1_095,
    "12h": 730, "1d": 365, "1w": 52,
}

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1_800, "1h": 3_600, "2h": 7_200,
    "4h": 14_400, "6h": 21_600, "8h": 28_800, "12h": 43_200, "1d": 86_400,
    "1w": 604_800,
}


@dataclass
class Config:
    initial_capital: float = 10_000
    fee_bps: float = 5.0        # charged per unit of turnover (i.e. per side)
    slippage_bps: float = 3.0   # crypto spreads are wider than you think
    max_leverage: float = 1.0
    periods_per_year: int = 365  # must match the bar timeframe


@dataclass
class RiskConfig:
    # Sizing: a stop at `atr_mult` ATRs away risks `risk_pct` of equity (article section 6).
    risk_pct: float = 0.01
    max_position_pct: float = 0.20
    atr_period: int = 14
    atr_mult: float = 2.0
    allow_short: bool = False          # spot accounts cannot short
    # Account-level circuit breakers, decided before deploying.
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15     # hard stop regardless of backtest
    max_order_notional: float = 5_000  # fat-finger cap per order (quote ccy)
    min_trade_notional: float = 10.0   # below exchange minimums anyway
    rebalance_threshold: float = 0.02  # ignore target changes < 2% of equity
    health_window: int = 90            # bars of live returns in the decay test


@dataclass
class GateConfig:
    train_bars: int = 180
    test_bars: int = 60
    # If set, these override the bar counts: 30 days is 30 bars on 1d, 2,880 on 15m.
    train_days: float | None = None
    test_days: float | None = None
    min_positive_fold_frac: float = 0.6
    # A fold is judged only if a position was held this share of its bars. Folds
    # that are ~entirely flat have a meaningless Sharpe (one exit fee / tiny std).
    # 10%, not higher: dip-buying strategies are rightly out of the market most days.
    min_fold_in_market: float = 0.10
    # Worst fold may not be worse than the expected worst fold of a ZERO-skill
    # strategy over the same number of folds (x tolerance). A fixed floor like -1
    # is unreachable: a 60-bar daily fold Sharpe has a standard error of ~2.5.
    worst_fold_tolerance: float = 1.0
    min_oos_sharpe: float = 0.3
    dsr_threshold: float = 0.95
    max_plausible_sharpe: float = 2.0   # above this on crypto = leakage until proven otherwise
    min_regime_frac: float = 0.15       # both bull and bear must be at least this share of the sample
    llm_critic: bool = False            # also ask Claude for the 8-point review
    report_max_age_days: int = 30       # strategies decay; re-validate on a schedule


@dataclass
class ForwardConfig:
    # Each entry: {"strategy": name} plus optional "symbol". Empty = every
    # registered strategy on the main symbol.
    candidates: list = field(default_factory=list)
    min_bars: int = 90                  # no verdict before this many live bars
    # Optional: ONE candidate also trades on the exchange DEMO account (fake money,
    # real order path). e.g. "funding_crowding" or {"strategy": ..., "symbol": ...}
    demo_candidate: object = None
    # Rounds: every `round_hours` the day's results are recorded; with rotate on,
    # the worst candidate (by cumulative return, after min_rounds) is retired and
    # new hypotheses fill up to max_active. Every candidate ever counts as a trial.
    round_hours: float = 24
    rotate: bool = False
    max_active: int = 8
    min_rounds: int = 1
    generator: str = "builtin"          # builtin (free) | claude (API, one call per new hypothesis)
    seed: int = 0
    # Weekly review: every review_days, rank active candidates by live Sharpe, keep
    # the top keep_top, retire the rest, write a report (`quantstack weekly`).
    review_days: float = 7
    keep_top: int = 5


@dataclass
class BotConfig:
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    perp_symbol: str | None = None      # for funding data; default BTC/USDT -> BTC/USDT:USDT
    timeframe: str = "1d"
    strategy: str = "ts_momentum"
    mode: str = "paper"                 # paper | testnet | demo | live
    history_bars: int = 1_500
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    initial_capital: float = 10_000     # paper account starting cash
    data_dir: str = "data"
    state_dir: str = "state"
    report_dir: str = "reports"
    log_dir: str = "logs"
    risk: RiskConfig = field(default_factory=RiskConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    forward: ForwardConfig = field(default_factory=ForwardConfig)

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.timeframe]

    @property
    def bar_seconds(self) -> int:
        return TIMEFRAME_SECONDS[self.timeframe]

    @property
    def bars_per_day(self) -> float:
        return 86_400 / self.bar_seconds

    @property
    def train_bars(self) -> int:
        g = self.gates
        return int(round(g.train_days * self.bars_per_day)) if g.train_days else g.train_bars

    @property
    def test_bars(self) -> int:
        g = self.gates
        return int(round(g.test_days * self.bars_per_day)) if g.test_days else g.test_bars

    def backtest_config(self) -> Config:
        return Config(
            initial_capital=self.initial_capital,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
            max_leverage=1.0,
            periods_per_year=self.periods_per_year,
        )

    def validate(self) -> None:
        if self.mode not in ("paper", "testnet", "demo", "live"):
            raise ValueError(f"mode must be paper, testnet, demo or live, got {self.mode!r}")
        if self.timeframe not in PERIODS_PER_YEAR:
            raise ValueError(f"unsupported timeframe {self.timeframe!r}")
        if self.fee_bps <= 0 or self.slippage_bps <= 0:
            raise ValueError("fees and slippage must be positive; zero-cost backtests lie")
        if not 0 < self.risk.risk_pct <= 0.05:
            raise ValueError("risk_pct must be in (0, 0.05]")
        if self.risk.allow_short:
            # The backtest would short, the spot brokers cannot: validated != traded.
            raise ValueError("allow_short needs a margin/perp broker adapter, which is not implemented")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> BotConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    risk = RiskConfig(**raw.pop("risk", {}) or {})
    gates = GateConfig(**raw.pop("gates", {}) or {})
    forward = ForwardConfig(**raw.pop("forward", {}) or {})
    cfg = BotConfig(**raw, risk=risk, gates=gates, forward=forward)
    cfg.validate()
    return cfg
