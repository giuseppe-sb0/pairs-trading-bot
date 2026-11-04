"""Universo sintetico: random walk indipendenti + AUDUSD/NZDUSD cointegrate (per i test)."""
import numpy as np
import pandas as pd

BASE = {"EURUSD": 1.10, "GBPUSD": 1.27, "AUDUSD": 0.66, "NZDUSD": 0.60, "USDCAD": 1.36,
        "USDCHF": 0.88, "USDJPY": 150.0, "EURGBP": 0.86, "EURCHF": 0.95}


def make_universe(months=24, seed=5, coint=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=int(months * 30.5 * 24), freq="1h", tz="UTC")
    idx = idx[idx.weekday < 5]
    n = len(idx)
    out = {}
    logs = {s: np.log(p) + np.cumsum(rng.standard_normal(n)) * 0.0012 for s, p in BASE.items()}
    if coint:  # NZD = AUD + processo OU (half-life ~ 35 ore)
        ou = np.zeros(n)
        for i in range(1, n):
            ou[i] = 0.98 * ou[i - 1] + rng.standard_normal() * 0.0008
        logs["NZDUSD"] = logs["AUDUSD"] - np.log(BASE["AUDUSD"]) + np.log(BASE["NZDUSD"]) + ou
    for s, lx in logs.items():
        c = np.exp(lx)
        o = np.r_[c[0], c[:-1]]
        w = np.abs(rng.standard_normal(n)) * 0.0004 * c
        out[s] = pd.DataFrame({"open": o, "high": np.maximum(o, c) + w, "low": np.minimum(o, c) - w,
                               "close": c, "volume": 100.0}, index=idx)
    return out
