"""Strategies defined by a JSON spec instead of code.

New hypotheses (from the built-in generator or from Claude) are expressed as a
spec: a family, bounded parameters and optional filters. Nothing is ever
executed from text: specs only select and parameterize building blocks written
here, each causal by construction (trailing, time-based windows). All windows
are in HOURS so a spec means the same thing on 15m, 1h or 1d bars.
"""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pandas as pd

from .base import Strategy

# family -> {param: allowed values}. The generator samples these; Claude is
# constrained to them by schema and every spec is validated against them.
FAMILIES: dict[str, dict[str, list]] = {
    "trend": {"lookback_h": [6, 12, 24, 48, 96, 168], "min_z": [0.0, 0.5, 1.0]},
    "breakout": {"entry_h": [4, 8, 24, 48, 96], "exit_h": [2, 4, 12, 24]},
    "reversion": {"window_h": [4, 12, 24, 48], "z_entry": [1.5, 2.0, 2.5, 3.0], "z_exit": [0.0, 0.5]},
    "session": {"range_end": [4, 6, 8, 10], "exit_hour": [14, 16, 20, 23]},
    "volatility": {"window_h": [6, 24, 48], "pct": [0.1, 0.2, 0.3]},
    "funding": {"window_h": [8, 24, 72], "threshold": [-0.0001, 0.0, 0.00005]},
}
FILTERS: dict[str, dict[str, list]] = {
    "trend_filter": {"days": [20, 50, 100, 200]},
    "hours": {"start": [0, 4, 8, 12, 16], "end": [8, 12, 16, 20, 24]},
    "funding_cap": {"max": [0.0002, 0.0003, 0.0005]},
}
MECHANISMS = {
    "trend": "Slow-reacting and forced traders extend moves; the trend follower is paid for "
             "providing liquidity in the direction of the move.",
    "breakout": "Stops and breakout orders cluster beyond recent highs; crossing them forces "
                "buying (short covering, liquidations) that extends the move.",
    "reversion": "Short-horizon overshoots come from forced or impatient sellers paying for "
                 "immediacy; a patient buyer is paid as price reverts.",
    "session": "Thin-session ranges break when a new session's liquidity arrives, triggering "
               "stops clustered at the range edge.",
    "volatility": "Quiet markets let leverage build; a breakout from low volatility forces it "
                  "out and produces an outsized move.",
    "funding": "Negative funding means crowded shorts paying longs; rallies force them to cover.",
}


def validate_spec(spec: dict) -> dict:
    """Reject anything outside the menu. Returns a normalized copy."""
    fam = spec.get("family")
    if fam not in FAMILIES:
        raise ValueError(f"unknown family {fam!r}")
    params = {}
    for k, allowed in FAMILIES[fam].items():
        v = spec.get("params", {}).get(k, allowed[0])
        if v not in allowed:
            raise ValueError(f"{fam}.{k}={v!r} not in {allowed}")
        params[k] = allowed[allowed.index(v)]        # 24.0 -> 24: stable ids
    extra = set(spec.get("params", {})) - set(FAMILIES[fam])
    if extra:
        raise ValueError(f"unknown params for {fam}: {sorted(extra)}")
    filters = []
    for f in spec.get("filters", []) or []:
        t = f.get("type")
        if t not in FILTERS:
            raise ValueError(f"unknown filter {t!r}")
        fp = {}
        for k, allowed in FILTERS[t].items():
            v = f.get(k, allowed[0])
            if v not in allowed:
                raise ValueError(f"filter {t}.{k}={v!r} not in {allowed}")
            fp[k] = allowed[allowed.index(v)]
        if t == "hours" and fp["start"] >= fp["end"]:
            raise ValueError("hours filter needs start < end")
        filters.append({"type": t, **fp})
    if len(filters) > 2:
        raise ValueError("at most 2 filters")
    mech = str(spec.get("mechanism") or MECHANISMS[fam]).strip()[:600]
    return {"family": fam, "params": params, "filters": filters, "mechanism": mech,
            "source": spec.get("source", "generator")}


def spec_id(spec: dict) -> str:
    core = {k: spec[k] for k in ("family", "params", "filters")}
    return "gen_" + hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8]


def random_spec(rng: random.Random) -> dict:
    fam = rng.choice(sorted(FAMILIES))
    params = {k: rng.choice(v) for k, v in FAMILIES[fam].items()}
    filters = []
    for t in rng.sample(sorted(FILTERS), k=rng.choice([0, 1, 1, 2])):
        fp = {k: rng.choice(v) for k, v in FILTERS[t].items()}
        if t == "hours" and fp["start"] >= fp["end"]:
            fp["start"], fp["end"] = 0, 24
        filters.append({"type": t, **fp})
    return validate_spec({"family": fam, "params": params, "filters": filters,
                          "mechanism": MECHANISMS[fam], "source": "generator"})


def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    ev = pd.Series(np.nan, index=entry.index)
    ev[entry] = 1.0
    ev[exit_] = 0.0
    return ev.ffill().fillna(0.0)


def _h(hours: float) -> str:
    return f"{int(hours * 60)}min"


class GeneratedStrategy(Strategy):
    param_grid = {"_": [0]}          # fixed spec: one trial per spec

    def __init__(self, spec: dict):
        self.spec = validate_spec(spec)
        self.name = spec_id(self.spec)
        self.mechanism = self.spec["mechanism"]
        self.requires = ("funding",) if (self.spec["family"] == "funding" or any(
            f["type"] == "funding_cap" for f in self.spec["filters"])) else ()

    def signal(self, bars: pd.DataFrame, **_) -> pd.Series:
        s, p = self.spec, self.spec["params"]
        c, idx = bars["close"], bars.index
        lr = np.log(c)
        fam = s["family"]
        if fam == "trend":
            w = _h(p["lookback_h"])
            # Log price as of `lookback_h` hours ago (last value at or before that time).
            past = lr.shift(freq=pd.Timedelta(hours=p["lookback_h"])).reindex(idx, method="ffill")
            ret = lr - past
            vol = lr.diff().rolling(w, min_periods=4).std() * np.sqrt(
                lr.rolling(w).count().clip(lower=1))
            sig = ((ret > p["min_z"] * vol) & past.notna()).astype(float)
        elif fam == "breakout":
            hi = bars["high"].rolling(_h(p["entry_h"]), min_periods=2).max().shift(1)
            lo = bars["low"].rolling(_h(p["exit_h"]), min_periods=2).min().shift(1)
            sig = _hold(c > hi, c < lo)
        elif fam == "reversion":
            w = _h(p["window_h"])
            m, sd = c.rolling(w, min_periods=4).mean(), c.rolling(w, min_periods=4).std()
            z = (c - m) / sd
            sig = _hold(z < -p["z_entry"], z > -p["z_exit"])
        elif fam == "session":
            day, hour = idx.floor("1D"), idx.hour + idx.minute / 60
            in_range = hour < p["range_end"]
            hi = bars["high"].where(in_range).groupby(day).cummax().groupby(day).ffill()
            lo = bars["low"].where(in_range).groupby(day).cummin().groupby(day).ffill()
            active = (hour >= p["range_end"]) & (hour < p["exit_hour"])
            sig = _hold(active & (c > hi), ~active | (c < lo))
        elif fam == "volatility":
            w = _h(p["window_h"])
            vol = lr.diff().rolling(w, min_periods=4).std()
            rank = vol.rolling("30D", min_periods=20).rank(pct=True)
            hi = bars["high"].rolling(w, min_periods=2).max().shift(1)
            ma = c.rolling(w, min_periods=2).mean()
            sig = _hold((rank.shift(1) <= p["pct"]) & (c > hi), c < ma)
        else:  # funding
            if "funding" not in bars:
                return pd.Series(0.0, index=idx)
            f = bars["funding"].rolling(_h(p["window_h"]), min_periods=1).mean()
            sig = (f < p["threshold"]).astype(float)

        for flt in s["filters"]:
            if flt["type"] == "trend_filter":
                daily = c.resample("1D").last().dropna()
                ma = daily.rolling(flt["days"], min_periods=flt["days"]).mean().shift(1)
                ma = ma.reindex(idx.floor("1D")).set_axis(idx)
                sig = sig.where(c > ma, 0.0)
            elif flt["type"] == "hours":
                hour = idx.hour
                sig = sig.where((hour >= flt["start"]) & (hour < flt["end"]), 0.0)
            elif flt["type"] == "funding_cap" and "funding" in bars:
                f3 = bars["funding"].rolling("3D", min_periods=1).mean()
                sig = sig.where(~(f3 > flt["max"]), 0.0)
        return sig.fillna(0.0)

    def fit(self, train, cfg, risk) -> dict:
        return {"_": 0}                # nothing to fit: the spec IS the hypothesis
