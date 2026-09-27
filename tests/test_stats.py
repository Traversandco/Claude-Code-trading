import numpy as np
import pandas as pd

from quantstack.stats import deflated_sharpe, deflated_sharpe_from_returns, expected_max_sharpe
from quantstack.trials import TrialLedger


def test_expected_max_grows_with_trials():
    v = 1 / 999
    assert expected_max_sharpe(1, v) == 0
    assert expected_max_sharpe(10, v) < expected_max_sharpe(100, v) < expected_max_sharpe(1000, v)


def test_best_of_many_noise_strategies_is_rejected():
    rng = np.random.default_rng(42)
    trials = [pd.Series(rng.normal(0, 0.02, 1000)) for _ in range(80)]
    srs = [t.mean() / t.std() for t in trials]
    best = trials[int(np.argmax(srs))]
    naive_ann_sharpe = max(srs) * np.sqrt(365)
    assert naive_ann_sharpe > 1.0                       # looks brilliant...
    out = deflated_sharpe_from_returns(best, 80, srs)
    assert out["verdict"] == "REJECT"                   # ...and is noise


def test_genuine_edge_passes_even_after_deflation():
    # True annual Sharpe ~2.9. (A true 1.9 realizes anywhere from ~0.7 to ~2.6 over
    # four years depending on the seed — which is exactly why the bar is hard.)
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.003, 0.02, 1500))
    assert deflated_sharpe_from_returns(r, 20)["verdict"] == "PASS"


def test_more_trials_lowers_dsr():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0012, 0.02, 1000))
    d = [deflated_sharpe_from_returns(r, n)["deflated_sharpe"] for n in (1, 10, 100, 1000)]
    assert d == sorted(d, reverse=True)


def test_article_signature_uses_annualized_input_consistently():
    assert deflated_sharpe(3.0, 10, 1500)["verdict"] == "PASS"
    assert deflated_sharpe(0.8, 80, 1000)["verdict"] == "REJECT"


def test_ledger_counts_distinct_trials_and_persists(tmp_path):
    p = tmp_path / "t.jsonl"
    L = TrialLedger(p)
    assert L.record("s", {"a": 1}, "u", 0.01, 100)
    assert not L.record("s", {"a": 1}, "u", 0.01, 100)      # identical re-run
    assert L.record("s", {"a": 2}, "u", 0.02, 100)          # a tweak is a trial
    assert L.record("s2", {"a": 1}, "u", 0.0, 100)          # another strategy too
    assert L.record("s", {"a": 1}, "other", 0.0, 100)
    assert TrialLedger(p).n_trials("u") == 3
