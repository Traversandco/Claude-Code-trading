"""The critic, as code: eight ways your backtest is lying, each checked by an
executable test instead of by reading. Each item is PASS, FAIL or N/A with evidence.

The core test is future perturbation: scramble every bar after time t and
confirm that nothing the strategy said at or before t changes. A centered moving
average, a bfill, a full-sample z-score, an unshifted resample — all of them fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, GateConfig, RiskConfig
from .engine import backtest
from .regimes import regime_fractions
from .strategies.base import Strategy

PRICE_COLS = ["open", "high", "low", "close"]


def _perturb_after(bars: pd.DataFrame, t: int, rng: np.random.Generator) -> pd.DataFrame:
    out = bars.copy()
    n = len(bars) - (t + 1)
    if n <= 0:
        return out
    # A different random walk for the future, applied to all price columns alike.
    shock = np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    for c in PRICE_COLS:
        out.iloc[t + 1:, out.columns.get_loc(c)] = bars[c].iloc[t + 1:].to_numpy() * shock * 1.37
    if "volume" in out:
        out.iloc[t + 1:, out.columns.get_loc("volume")] = rng.permutation(bars["volume"].iloc[t + 1:].to_numpy())
    return out


def _same(a: pd.Series, b: pd.Series) -> bool:
    return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-12))


def causality_tests(fn, bars: pd.DataFrame, n_cuts: int = 6, seed: int = 7) -> dict:
    """fn(bars) -> Series. Returns look-ahead (perturbation) and repainting (truncation) results."""
    rng = np.random.default_rng(seed)
    full = fn(bars)
    lo = max(len(bars) // 4, 1)
    cuts = sorted(set(np.linspace(lo, len(bars) - 2, n_cuts).astype(int)))
    lookahead, repaint = [], []
    for t in cuts:
        pert = fn(_perturb_after(bars, t, rng))
        if not _same(full.iloc[: t + 1], pert.iloc[: t + 1]):
            diff = (full.iloc[: t + 1] - pert.iloc[: t + 1]).abs()
            lookahead.append(str(diff[diff > 1e-12].index[0]))
        trunc = fn(bars.iloc[: t + 1])
        if not _same(full.iloc[: t + 1], trunc):
            diff = (full.iloc[: t + 1] - trunc).abs()
            repaint.append(str(diff[diff > 1e-12].index[0]) if (diff > 1e-12).any() else str(bars.index[t]))
    return {"cuts_tested": len(cuts), "lookahead_at": lookahead, "repaint_at": repaint}


def engine_shift_check(cfg: Config) -> bool:
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    prices = pd.Series(np.linspace(100, 110, 10), index=idx)
    sig = pd.Series([0, 1, 0, 1, 1, 0, 0, 1, 0, 0], index=idx, dtype=float)
    pos = backtest(prices, sig, cfg)["position"]
    return _same(pos, sig.shift(1).fillna(0))


def data_alignment(bars: pd.DataFrame, bar_seconds: int) -> dict:
    idx = bars.index
    tz_ok = isinstance(idx, pd.DatetimeIndex) and idx.tz is not None and str(idx.tz) in ("UTC", "utc")
    mono = bool(idx.is_monotonic_increasing)
    unique = bool(idx.is_unique)
    gaps = idx.to_series().diff().dropna().dt.total_seconds()
    irregular = float((gaps != bar_seconds).mean()) if len(gaps) else 0.0
    ohlc_ok = bool(((bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9)
                    & (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9)).all())
    return {"utc": tz_ok, "monotonic": mono, "unique": unique,
            "irregular_gap_frac": round(irregular, 4), "ohlc_consistent": ohlc_ok,
            "ok": tz_ok and mono and unique and irregular < 0.01 and ohlc_ok}


def run_critic(strategy: Strategy, params: dict, bars: pd.DataFrame, cfg: Config,
               risk: RiskConfig, gates: GateConfig, bar_seconds: int,
               oos_sharpe: float | None = None) -> dict:
    items: dict[str, dict] = {}

    sig_t = causality_tests(lambda b: strategy.signal(b, **params), bars)
    exp_t = causality_tests(lambda b: strategy.exposure(b, params, risk), bars)
    shift_ok = engine_shift_check(cfg)
    la = sig_t["lookahead_at"] + exp_t["lookahead_at"]
    items["1_lookahead"] = {
        "status": "PASS" if shift_ok and not la else "FAIL",
        "evidence": {"engine_shifts_signal": shift_ok, "future_perturbation_violations": la[:5],
                     "cuts_tested": sig_t["cuts_tested"]},
    }

    items["2_survivorship"] = {
        "status": "N/A",
        "evidence": "single instrument; survivorship applies when selecting a universe. "
                    "If you pick symbols by today's listings, include delisted pairs.",
    }

    rp = sig_t["repaint_at"] + exp_t["repaint_at"]
    items["3_repainting"] = {
        "status": "FAIL" if rp else "PASS",
        "evidence": {"signal_changes_when_history_truncated_at": rp[:5]},
    }

    items["4_costs"] = {
        "status": "PASS" if cfg.fee_bps > 0 and cfg.slippage_bps > 0 else "FAIL",
        "evidence": {"fee_bps": cfg.fee_bps, "slippage_bps": cfg.slippage_bps,
                     "applied_on": "turnover = |Δposition|"},
    }

    # Engine assumes a fill at the signal bar's close. Real fills land near the next
    # open, so slippage must at least cover the typical close→next-open gap.
    gap_bps = float(((bars["open"].shift(-1) - bars["close"]).abs() / bars["close"]).dropna().median() * 1e4)
    items["5_fill_assumption"] = {
        "status": "PASS" if cfg.slippage_bps >= gap_bps else "FAIL",
        "evidence": {"assumed_fill": "close of signal bar + slippage",
                     "median_close_to_next_open_gap_bps": round(gap_bps, 2),
                     "slippage_bps": cfg.slippage_bps},
    }

    n_params = len(strategy.param_grid)
    n_combos = len(strategy.param_combos())
    items["6_parameter_fitting"] = {
        "status": "PASS" if n_combos <= 50 else "FAIL",
        "evidence": {"n_params": n_params, "grid_size": n_combos,
                     "chosen_on": "train windows only (walk-forward); full-sample runs are "
                                  "recorded in the trial ledger, never used for selection"},
    }

    rf = regime_fractions(bars["close"])
    items["7_sample_regimes"] = {
        "status": "PASS" if rf["bull"] >= gates.min_regime_frac and rf["bear"] >= gates.min_regime_frac else "FAIL",
        "evidence": {"regime_fractions_200ma": rf, "min_required": gates.min_regime_frac},
    }

    da = data_alignment(bars, bar_seconds)
    items["8_data_alignment"] = {"status": "PASS" if da["ok"] else "FAIL", "evidence": da}

    if oos_sharpe is not None:
        items["9_sharpe_sanity"] = {
            "status": "PASS" if oos_sharpe <= gates.max_plausible_sharpe else "FAIL",
            "evidence": {"oos_sharpe": oos_sharpe, "max_plausible": gates.max_plausible_sharpe,
                         "note": "above this on crypto means leakage until proven otherwise"},
        }

    failed = [k for k, v in items.items() if v["status"] == "FAIL"]
    return {"items": items, "failed": failed, "passed": not failed}
