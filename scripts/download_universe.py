#!/usr/bin/env python3
"""Scarica da Dukascopy le candele M1 BID delle 9 coppie e salva barre H1 in data/h1/.

    python scripts/download_universe.py --months 36
    # riusa la cache del progetto asia-meanrev-bot (EURUSD gia' scaricato):
    python scripts/download_universe.py --months 36 --cache-dir ../asia-meanrev-bot/data/raw/dukascopy
"""
import _bootstrap  # noqa: F401

import argparse
import logging
from datetime import date, timedelta

import pandas as pd

from pairs import ROOT
from pairs.config import load_config
from pairs.data.dukascopy import download_range
from pairs.data.io import clean_m1, resample, save_bars

# intervalli plausibili del prezzo: un valore fuori indica un errore di scala nei dati
PLAUSIBLE = {"USDJPY": (80, 250)}
DEFAULT_RANGE = (0.4, 2.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--end", help="esclusa (default: ieri, cioe' dati fino all'altro ieri)")
    ap.add_argument("--symbols", nargs="*", help="default: universo in config.yaml")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "raw" / "dukascopy"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    # default: fino a ieri escluso. Gli ultimi giorni Dukascopy li pubblica in ritardo (errori 503)
    # e per una diagnosi su anni di dati non servono.
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    start = (pd.Timestamp(end) - pd.DateOffset(months=a.months)).date()
    problems, incomplete = [], {}
    for sym in a.symbols or cfg["universe"]:
        logging.info("=== %s %s -> %s ===", sym, start, end)
        raw = download_range(sym, start, end, a.cache_dir, workers=a.workers)
        failed = raw.attrs.get("failed_days", [])
        if failed:
            incomplete[sym] = len(failed)
            logging.error("%s: %d giorni non scaricati, file H1 NON salvato (dati incompleti)", sym, len(failed))
            continue
        m1 = clean_m1(raw)
        h1 = resample(m1, cfg["timeframe"])
        save_bars(h1, ROOT / "data" / "h1" / f"{sym}.csv.gz")
        med = float(h1["close"].median())
        lo, hi = PLAUSIBLE.get(sym, DEFAULT_RANGE)
        flag = "OK" if lo <= med <= hi else "SOSPETTO"
        if flag != "OK":
            problems.append(sym)
        logging.info("%s: %d barre H1, prezzo mediano %.5f [%s]", sym, len(h1), med, flag)
    if incomplete:
        raise SystemExit(f"Download incompleto per {incomplete} (giorni mancanti). Il server Dukascopy era "
                         "sovraccarico: aspetta qualche minuto e rilancia lo STESSO comando, magari con "
                         "--workers 2. I giorni gia' scaricati sono in cache e non vengono riscaricati.")
    if problems:
        raise SystemExit(f"Prezzi fuori scala per {problems}: non usare questi dati")
    logging.info("Fatto. Ora: python scripts/diagnose_pairs.py")


if __name__ == "__main__":
    main()
