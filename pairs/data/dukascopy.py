"""Download gratuito di candele M1 da Dukascopy (datafeed pubblico .bi5).

URL giornaliero: https://datafeed.dukascopy.com/datafeed/EURUSD/2025/00/15/BID_candles_min_1.bi5
(mese 0-based). Ogni file e' LZMA; record da 24 byte big-endian:
  uint32 secondi dall'inizio del giorno UTC, uint32 open, close, low, high, float32 volume
I prezzi sono interi da dividere per 10^5 (10^3 per le coppie JPY).
"""
from __future__ import annotations

import logging
import lzma
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
DTYPE = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])


def _divisor(symbol: str) -> float:
    return 1e3 if "JPY" in symbol.upper() else 1e5


class DownloadError(RuntimeError):
    pass


def _fetch(session: requests.Session, url: str, retries: int = 8) -> bytes:
    """GET con retry: attese crescenti (fino a 60 s) con jitter; rispetta Retry-After su 429/503."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            wait = min(60.0, 2.0 ** attempt) + random.uniform(0, 1.5)
            resp = getattr(exc, "response", None)
            if resp is not None and resp.headers.get("Retry-After", "").isdigit():
                wait = max(wait, float(resp.headers["Retry-After"]))
            if attempt < retries - 1:
                log.warning("errore %s (%s), retry %d/%d tra %.0fs", url, exc, attempt + 1, retries - 1, wait)
                time.sleep(wait)
    raise DownloadError(f"Download fallito dopo {retries} tentativi: {url}")


def parse_bi5(content: bytes, day: date, symbol: str) -> pd.DataFrame:
    if not content:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    raw = lzma.decompress(content)
    arr = np.frombuffer(raw[: len(raw) // DTYPE.itemsize * DTYPE.itemsize], dtype=DTYPE)
    div = _divisor(symbol)
    base = pd.Timestamp(day, tz="UTC")
    idx = base + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    return pd.DataFrame(
        {
            "open": arr["o"] / div,
            "high": arr["h"] / div,
            "low": arr["l"] / div,
            "close": arr["c"] / div,
            "volume": arr["v"].astype(float),
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )


def download_range(symbol: str, start: date, end: date, cache_dir: Path,
                   side: str = "BID", workers: int = 4) -> pd.DataFrame:
    """Scarica i giorni [start, end) con cache locale dei file .bi5."""
    cache_dir = Path(cache_dir) / symbol / side
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = [start + timedelta(days=i) for i in range((end - start).days)]
    sess = requests.Session()
    sess.headers["User-Agent"] = "asiamr-backtester/1.0"

    def download(d: date, f: Path) -> bytes:
        url = URL.format(sym=symbol, y=d.year, m=d.month - 1, d=d.day, side=side)
        content = _fetch(sess, url)
        if d < date.today() - timedelta(days=1):   # non mettere in cache il giorno corrente
            tmp = f.with_name(f.name + ".part")    # scrittura atomica: un'interruzione non lascia file a meta'
            tmp.write_bytes(content)
            tmp.replace(f)
        return content

    def job(d: date) -> pd.DataFrame:
        f = cache_dir / f"{d.isoformat()}.bi5"
        if f.exists():
            try:
                return parse_bi5(f.read_bytes(), d, symbol)
            except (lzma.LZMAError, EOFError, ValueError):
                log.warning("file in cache danneggiato, lo riscarico: %s", f)
                f.unlink(missing_ok=True)
        return parse_bi5(download(d, f), d, symbol)

    frames, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, d): d for d in days}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                frames.append(fut.result())
            except DownloadError as exc:      # un giorno non scaricabile non ferma tutto il resto
                failed.append(futs[fut])
                log.error("%s", exc)
            if n % 25 == 0 or n == len(days):
                log.info("%s: elaborati %d/%d giorni (%d falliti)", symbol, n, len(days), len(failed))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("Nessun dato scaricato")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.attrs["failed_days"] = sorted(failed)    # da riscaricare: rilanciare lo stesso comando
    return df
