"""Motore pairs trading: spread rolling, simulazione a due gambe, selezione e walk-forward.

Spread (causale): sui log-prezzi H1 si stima a ogni barra t, con le ultime L barre
(t inclusa), la regressione  logA = alpha + beta * logB. Il residuo corrente diviso
per la deviazione standard dei residui della finestra e' lo z-score.

Trade:
  z >= +z_entry -> vendi lo spread (short A, long beta*B)
  z <= -z_entry -> compra lo spread (long A, short beta*B)
  ingresso all'apertura della barra successiva; beta CONGELATO all'ingresso.
  uscita all'apertura della barra successiva a: ritorno alla media (z oltre z_exit),
  stop (|z| >= z_entry + z_stop_add dalla parte sbagliata) o tempo (max_hold barre).

P&L in punti base (bps) del nozionale della gamba A:
  side * [(dlogA) - beta * (dlogB)] * 1e4
Costi: costo round-trip di A + |beta| * costo di B (in bps) + swap stimato per giorno.

Nota: lo stato di posizione e' simulato in continuo; il walk-forward conta solo i trade
APERTI nei periodi di test in cui la coppia era stata selezionata sul train.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.io import pip_size

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    lookback: int
    z_entry: float
    z_exit: float
    z_stop_add: float
    max_hold: int

    @property
    def name(self) -> str:
        return f"L{self.lookback}_z{self.z_entry:g}"


def rules_from_cfg(cfg: dict) -> list[Rule]:
    r = cfg["rules"]
    return [Rule(L, z, r["z_exit"], r["z_stop_add"], int(L * r["max_hold_mult"]))
            for L, z in itertools.product(r["lookback_hours"], r["z_entry"])]


def rolling_z(la: pd.Series, lb: pd.Series, L: int) -> pd.DataFrame:
    """beta, alpha e z-score della regressione rolling logA ~ logB (finestra L, t inclusa)."""
    ma, mb = la.rolling(L).mean(), lb.rolling(L).mean()
    va, vb = la.rolling(L).var(), lb.rolling(L).var()
    cov = la.rolling(L).cov(lb)
    beta = cov / vb
    alpha = ma - beta * mb
    resid = la - alpha - beta * lb
    var_e = (va - beta ** 2 * vb).clip(lower=0)
    sd = np.sqrt(var_e)
    z = resid / sd.where(sd > 0)
    return pd.DataFrame({"beta": beta, "z": z})


def simulate(times: pd.DatetimeIndex, oa: np.ndarray, ob: np.ndarray, pa: np.ndarray, pb: np.ndarray,
             z: np.ndarray, beta: np.ndarray, entry_ok: np.ndarray, rule: Rule,
             cost_a_bps: np.ndarray, cost_b_bps: np.ndarray, swap_bps_leg_day: float) -> list[dict]:
    """oa/ob = log open, pa/pb = prezzo open (per i costi in bps)."""
    n = len(z)
    trades = []
    pos = 0
    tns = times.values.astype("datetime64[ns]").astype(np.int64)
    stop_lvl = rule.z_entry + rule.z_stop_add
    for t in range(n - 1):
        zt = z[t]
        if np.isnan(zt):
            continue
        if pos == 0:
            if not entry_ok[t] or np.isnan(beta[t]):
                continue
            if zt >= rule.z_entry:
                side = -1
            elif zt <= -rule.z_entry:
                side = 1
            else:
                continue
            pos, i_in, b = side, t + 1, beta[t]
            z_in = zt
            continue
        held = t + 1 - i_in
        reason = None
        if (pos == 1 and zt >= rule.z_exit) or (pos == -1 and zt <= rule.z_exit):
            reason = "mean"
        elif (pos == 1 and zt <= -stop_lvl) or (pos == -1 and zt >= stop_lvl):
            reason = "stop"
        elif held >= rule.max_hold:
            reason = "time"
        if reason is None:
            continue
        i_out = t + 1
        gross = pos * ((oa[i_out] - oa[i_in]) - b * (ob[i_out] - ob[i_in])) * 1e4
        days = (tns[i_out] - tns[i_in]) / 8.64e13
        cost = cost_a_bps[i_in] + abs(b) * cost_b_bps[i_in] + swap_bps_leg_day * days * (1 + abs(b))
        trades.append({"entry_time": times[i_in], "exit_time": times[i_out], "side": pos, "beta": b,
                       "z_in": z_in, "reason": reason, "hold_h": (tns[i_out] - tns[i_in]) / 3.6e12,
                       "gross_bps": gross, "cost_bps": cost, "net_bps": gross - cost})
        pos = 0
    return trades


def select_pair(la: pd.Series, lb: pd.Series, sel: dict) -> dict:
    """Engle-Granger + half-life sul periodo di train (log-prezzi)."""
    from statsmodels.tsa.stattools import coint
    ok = la.notna() & lb.notna()
    a, b = la[ok].to_numpy(), lb[ok].to_numpy()
    if len(a) < 500:
        return {"pvalue": np.nan, "half_life": np.nan, "selected": False}
    _, p, _ = coint(a, b, trend="c", autolag=None, maxlag=1)
    X = np.vstack([np.ones_like(b), b]).T
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    e = a - X @ coef
    de, e1 = np.diff(e), e[:-1] - e[:-1].mean()
    phi = float(np.dot(e1, de) / np.dot(e1, e1))
    hl = float(-np.log(2) / np.log(1 + phi)) if -1 < phi < 0 else np.inf
    selected = bool(p < sel["coint_pvalue"] and sel["half_life_min"] <= hl <= sel["half_life_max"])
    return {"pvalue": float(p), "half_life": hl, "selected": selected}


def make_folds(index: pd.DatetimeIndex, wf: dict) -> list[tuple]:
    first = index[0].normalize().replace(day=1)
    if index[0] > first:
        first += pd.DateOffset(months=1)
    folds, cur = [], first
    while True:
        tr_end = cur + pd.DateOffset(months=wf["train_months"])
        te_end = tr_end + pd.DateOffset(months=wf["test_months"])
        if te_end > index[-1] + pd.Timedelta(days=3):
            break
        folds.append((cur, tr_end, tr_end, te_end))
        cur += pd.DateOffset(months=wf["step_months"])
    return folds


def aligned_panel(h1: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open e close allineati (solo ore presenti per tutti i simboli)."""
    opens = pd.concat({s: d["open"] for s, d in h1.items()}, axis=1, join="inner")
    closes = pd.concat({s: d["close"] for s, d in h1.items()}, axis=1, join="inner")
    return opens, closes


def run(h1: dict[str, pd.DataFrame], cfg: dict) -> dict:
    opens, closes = aligned_panel(h1)
    times = closes.index
    local = times.tz_convert(cfg["timezone"])
    # il segnale e' valutato alla chiusura della barra t; l'ingresso avviene all'ora t+1
    entry_hour = (times + pd.Timedelta(hours=1)).tz_convert(cfg["timezone"]).hour
    entry_ok = ~np.isin(entry_hour, cfg["no_entry_hours"])
    lc, lo = np.log(closes), np.log(opens)
    cost_bps = {s: (cfg["costs_pips"][s] * pip_size(s) / opens[s] * 1e4).to_numpy() for s in closes}
    rules = rules_from_cfg(cfg)
    folds = make_folds(times, cfg["walkforward"])
    if not folds:
        raise ValueError("Dati insufficienti per il walk-forward")
    pairs = list(itertools.combinations(cfg["universe"], 2))
    log.info("%d coppie x %d regole, %d fold, %d barre H1", len(pairs), len(rules), len(folds), len(times))

    # 1) selezione su ogni train
    sel_rows = []
    for fi, (tr0, tr1, te0, te1) in enumerate(folds, 1):
        m = (times >= tr0) & (times < tr1)
        for a, b in pairs:
            r = select_pair(lc[a][m], lc[b][m], cfg["selection"])
            sel_rows.append({"fold": fi, "pair": f"{a}/{b}", **r})
    sel = pd.DataFrame(sel_rows)

    # 2) simulazione continua per coppia e regola, poi assegnazione ai fold di test
    all_tr = []
    for a, b in pairs:
        for rule in rules:
            rz = rolling_z(lc[a], lc[b], rule.lookback)
            tr = simulate(times, lo[a].to_numpy(), lo[b].to_numpy(), opens[a].to_numpy(), opens[b].to_numpy(),
                          rz["z"].to_numpy(), rz["beta"].to_numpy(), entry_ok, rule,
                          cost_bps[a], cost_bps[b], cfg["swap_bps_per_leg_day"])
            for t in tr:
                t.update(pair=f"{a}/{b}", rule=rule.name)
            all_tr.extend(tr)
    trades = pd.DataFrame(all_tr)
    if trades.empty:
        return {"trades": trades, "selection": sel, "folds": folds}
    trades["fold"] = 0
    for fi, (_, _, te0, te1) in enumerate(folds, 1):
        trades.loc[(trades.entry_time >= te0) & (trades.entry_time < te1), "fold"] = fi
    trades = trades[trades.fold > 0]
    trades = trades.merge(sel[["fold", "pair", "selected"]], on=["fold", "pair"], how="left")
    return {"trades": trades.reset_index(drop=True), "selection": sel, "folds": folds}


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    net = t["net_bps"].to_numpy()
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    exit_t = pd.to_datetime(t["exit_time"])
    daily = t.groupby(exit_t.dt.floor("D"))["net_bps"].sum()
    # settimane di calendario, incluse quelle senza trade (P&L 0): assorbe la correlazione tra trade
    wk = t.groupby(exit_t.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W"))["net_bps"].sum()
    wk = wk.reindex(pd.period_range(wk.index.min(), wk.index.max(), freq="W"), fill_value=0.0)
    order = np.argsort(t["entry_time"].to_numpy())
    half = np.zeros(len(net), bool)
    half[order[: len(net) // 2]] = True
    return {
        "n": len(t),
        "pairs": t["pair"].nunique(),
        "hold_h": round(float(t["hold_h"].mean()), 1),
        "win_%": round(float((net > 0).mean() * 100), 1),
        "gross_bps": round(float(t["gross_bps"].mean()), 2),
        "cost_bps": round(float(t["cost_bps"].mean()), 2),
        "net_bps": round(float(net.mean()), 2),
        "net_pf": round(float(gains / losses), 2) if losses > 0 else np.inf,
        "t_trades": round(float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))), 2) if len(net) > 2 else 0.0,
        "t_daily": round(float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))), 2) if len(daily) > 2 else 0.0,
        "t_weekly": round(float(wk.mean() / (wk.std(ddof=1) / np.sqrt(len(wk)))), 2) if len(wk) > 2 and wk.std() > 0 else 0.0,
        "net_1st_half": round(float(net[half].mean()), 2) if half.any() else np.nan,
        "net_2nd_half": round(float(net[~half].mean()), 2) if (~half).any() else np.nan,
        "exits": t["reason"].value_counts().to_dict(),
    }


def per_pair_verdict(trades: pd.DataFrame, min_trades: int, t_min: float) -> pd.DataFrame:
    """Statistiche OOS per (coppia, regola) sui soli periodi selezionati e candidati che superano i criteri."""
    sel = trades[trades.selected == True]  # noqa: E712
    rows = []
    for (pair, rule), g in sel.groupby(["pair", "rule"]):
        s = stats(g)
        s.pop("exits", None)
        s.pop("pairs", None)
        rows.append({"pair": pair, "rule": rule, **s})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["candidate"] = ((df.n >= min_trades) & (df.net_bps > 0) & (df.t_weekly >= t_min)
                       & (df.net_1st_half > 0) & (df.net_2nd_half > 0))
    return df
