"""Honest accounting. Every parameter tweak is a trial, and the ledger counts them.

Append-only JSONL. Identical (strategy, params, dataset) re-runs are not double
counted; anything else — a new grid value, a new strategy, a new lookback —
is a new trial, and it raises the bar every strategy has to clear.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def _key(strategy: str, params: dict, universe: str) -> str:
    blob = json.dumps({"s": strategy, "p": params, "u": universe}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TrialLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self._entries[e["key"]] = e

    def record(self, strategy: str, params: dict, universe: str, sharpe_per_period: float,
               n_obs: int) -> bool:
        """Returns True if this was a new trial."""
        key = _key(strategy, params, universe)
        if key in self._entries:
            return False
        e = {
            "key": key, "ts": time.time(), "strategy": strategy, "params": params,
            "universe": universe, "sharpe_pp": sharpe_per_period, "n_obs": n_obs,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e, default=str) + "\n")
        self._entries[key] = e
        return True

    def n_trials(self, universe: str | None = None) -> int:
        """Counts across ALL strategies: choosing between strategies is also selection."""
        return len(self.sharpes(universe))

    def sharpes(self, universe: str | None = None) -> list[float]:
        return [e["sharpe_pp"] for e in self._entries.values()
                if universe is None or e["universe"] == universe]
