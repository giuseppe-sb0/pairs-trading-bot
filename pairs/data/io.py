"""Lettura/scrittura barre (CSV gz, indice UTC = apertura barra, prezzi BID)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def clean_m1(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    flat = (df["volume"] <= 0) & (df["high"] == df["low"])  # weekend / festivi
    return df[~flat]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg(AGG).dropna(subset=["open"])


def save_bars(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.index = out.index.tz_convert("UTC")
    out.index.name = "time"
    out.to_csv(path, float_format="%.6f")


def load_bars(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001
