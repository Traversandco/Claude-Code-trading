from quantstack.cli import search
from quantstack.data import attach_funding
from tests.test_funding import synthetic_funding


def test_search_ranks_and_counts_every_trial(cfg, noise_bars):
    bars = attach_funding(noise_bars, synthetic_funding(noise_bars), 86_400)
    names = ["donchian_breakout", "vol_squeeze", "funding_squeeze", "bull_dip"]
    rows = search(cfg, names, None, True, loader=lambda c: bars)
    assert [r["strategy"] for r in rows] == names          # nothing passes on noise: all run
    assert all(r["verdict"].startswith("fail") for r in rows)
    # trial count only ever grows as the search proceeds
    trials = [r["n_trials"] for r in rows]
    assert trials == sorted(trials) and trials[-1] == 4 + 2 + 4 + 2


def test_search_stops_at_first_pass(cfg, edge_bars):
    rows = search(cfg, ["ts_momentum", "rsi_reversion"], None, True, loader=lambda c: edge_bars)
    assert len(rows) == 1 and rows[0]["verdict"] == "PASS"


def test_target_above_cap_is_flagged_not_passed(cfg, edge_bars):
    rows = search(cfg, ["ts_momentum"], 2.5, True, loader=lambda c: edge_bars)
    assert rows[0]["verdict"] != "PASS"
