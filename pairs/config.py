"""Caricamento configurazione."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import ROOT

DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = set(cfg["universe"]) - set(cfg["costs_pips"])
    if missing:
        raise ValueError(f"costs_pips mancanti per: {sorted(missing)}")
    return cfg
