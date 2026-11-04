import numpy as np
import pandas as pd
import pytest

from pairs.config import load_config
from pairs.engine import Rule, rolling_z, run, simulate, stats
from synthetic import make_universe


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_rolling_z_no_lookahead():
    rng = np.random.default_rng(1)
    a = pd.Series(np.cumsum(rng.standard_normal(2000)))
    b = pd.Series(np.cumsum(rng.standard_normal(2000)))
    full = rolling_z(a, b, 120)
    part = rolling_z(a.iloc[:1500], b.iloc[:1500], 120)
    pd.testing.assert_frame_equal(part, full.iloc[:1500])


def test_simulate_pnl_and_costs():
    # spread che scende a z=-3 e poi torna a 0: un trade long spread
    times = pd.date_range("2025-01-06", periods=6, freq="1h", tz="UTC")
    z = np.array([0.0, -3.0, -1.0, 0.5, 0.0, 0.0])
    beta = np.ones(6)
    oa = np.log(np.array([1.0, 1.0, 1.0, 1.001, 1.002, 1.002]))   # A sale di 10 bps tra ingresso e uscita
    ob = np.zeros(6)
    r = Rule(lookback=10, z_entry=2.0, z_exit=0.0, z_stop_add=2.0, max_hold=10)
    tr = simulate(times, oa, ob, np.ones(6), np.ones(6), z, beta, np.ones(6, bool), r,
                  np.full(6, 1.0), np.full(6, 2.0), 0.0)
    assert len(tr) == 1
    t = tr[0]
    assert t["side"] == 1 and t["reason"] == "mean"
    assert t["entry_time"] == times[2] and t["exit_time"] == times[4]
    assert t["gross_bps"] == pytest.approx(np.log(1.002) * 1e4)
    assert t["cost_bps"] == pytest.approx(3.0)


def test_finds_planted_cointegration(cfg):
    r = run(make_universe(24), cfg)
    sel = r["selection"]
    share = sel[sel.pair == "AUDUSD/NZDUSD"].selected.mean()
    assert share > 0.8
    tr = r["trades"]
    planted = tr[tr.pair == "AUDUSD/NZDUSD"]
    assert stats(planted)["gross_bps"] > 5


def test_no_false_positive_without_cointegration(cfg):
    from pairs.engine import per_pair_verdict
    v = cfg["verdict"]
    r = run(make_universe(36, seed=9, coint=False), cfg)
    pv = per_pair_verdict(r["trades"], v["pair_min_trades"], v["pair_t_min"])
    assert not pv.candidate.any()                     # random walk: nessuna coppia candidata
    assert r["selection"].selected.mean() < 0.15      # solo falsi positivi occasionali in selezione


def test_per_pair_verdict_finds_planted(cfg):
    from pairs.engine import per_pair_verdict
    v = cfg["verdict"]
    pv = per_pair_verdict(run(make_universe(36), cfg)["trades"], v["pair_min_trades"], v["pair_t_min"])
    assert set(pv[pv.candidate].pair) == {"AUDUSD/NZDUSD"}
