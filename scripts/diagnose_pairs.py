#!/usr/bin/env python3
"""Diagnosi walk-forward del pairs trading forex (regole in config/config.yaml).

    python scripts/diagnose_pairs.py

Per ogni fold: seleziona le coppie cointegrate sugli 8 mesi di train, poi conta solo i
trade aperti nei 2 mesi di test successivi. Confronta coppie selezionate e non selezionate.
"""
import _bootstrap  # noqa: F401

import argparse
import logging
from pathlib import Path

import pandas as pd

from pairs import ROOT
from pairs.config import load_config
from pairs.data.io import load_bars
from pairs.engine import per_pair_verdict, run, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "h1"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "diagnose_pairs"))
    ap.add_argument("--config")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    h1 = {}
    for s in cfg["universe"]:
        f = Path(a.data_dir) / f"{s}.csv.gz"
        if not f.exists():
            raise SystemExit(f"Manca {f}: esegui prima scripts/download_universe.py")
        h1[s] = load_bars(f)
    logging.info("Calcolo in corso (1-3 minuti)...")
    r = run(h1, cfg)
    tr, sel = r["trades"], r["selection"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "oos_trades.csv", index=False)
    sel.to_csv(out / "selection.csv", index=False)

    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 60)

    print("\n=== 1) SELEZIONE PER FOLD (cointegrazione sul train) ===")
    rows = []
    for fi, (tr0, tr1, te0, te1) in enumerate(r["folds"], 1):
        s = sel[(sel.fold == fi) & sel.selected]
        rows.append({"fold": fi, "train": f"{tr0.date()}..{tr1.date()}", "test": f"{te0.date()}..{te1.date()}",
                     "n_sel": len(s), "coppie": ", ".join(s.pair.str.replace("/", "-"))})
    print(pd.DataFrame(rows).to_string(index=False))
    freq = sel[sel.selected].groupby("pair").size().sort_values(ascending=False)
    print(f"\nFrequenza di selezione (su {len(r['folds'])} fold):",
          ", ".join(f"{p} {n}" for p, n in freq.items()) or "nessuna")

    print("\n=== 2) RISULTATI OUT-OF-SAMPLE PER REGOLA (bps del nozionale; 1 bps EURUSD ~ 1.1 pip) ===")
    res = []
    for rule in sorted(tr.rule.unique()) if not tr.empty else []:
        for grp, mask in (("SELEZIONATE", tr.selected == True), ("non selez.", tr.selected != True),  # noqa: E712
                          ("tutte", tr.rule == tr.rule)):
            s = stats(tr[(tr.rule == rule) & mask])
            res.append({"rule": rule, "gruppo": grp, **s})
    res = pd.DataFrame(res)
    if not res.empty:
        print(res.drop(columns=["exits"]).to_string(index=False))
        res.to_csv(out / "summary.csv", index=False)

    print("\n=== 3) PER COPPIA (solo periodi selezionati, tutte le regole) ===")
    s_tr = tr[tr.selected == True]  # noqa: E712
    if not s_tr.empty:
        g = s_tr.groupby("pair")
        pp = pd.DataFrame({"n": g.size(), "gross_bps": g.gross_bps.mean().round(2),
                           "cost_bps": g.cost_bps.mean().round(2), "net_bps": g.net_bps.mean().round(2),
                           "win_%": (g.net_bps.apply(lambda x: (x > 0).mean() * 100)).round(1)})
        print(pp.sort_values("n", ascending=False).to_string())

    print("\n=== 4) VERDETTO A: portafoglio di tutte le coppie selezionate ===")
    v = cfg["verdict"]
    sres = res[res.gruppo == "SELEZIONATE"] if not res.empty else res
    cand = sres[(sres.n >= v["min_trades"]) & (sres.net_bps > 0) & (sres.t_weekly >= v["t_min"])
                & (sres.net_1st_half > 0) & (sres.net_2nd_half > 0)] if not sres.empty else sres
    cols = ["rule", "n", "pairs", "hold_h", "win_%", "gross_bps", "cost_bps", "net_bps", "net_pf",
            "t_weekly", "net_1st_half", "net_2nd_half"]
    if cand.empty:
        print(f"  NESSUN VANTAGGIO SFRUTTABILE (criteri: >= {v['min_trades']} trade OOS, netto > 0,")
        print(f"  t settimanale >= {v['t_min']}, positivo in entrambe le meta').")
        if not sres.empty:
            print("  Regole sulle coppie selezionate:")
            print(sres.sort_values("t_weekly", ascending=False)[cols].to_string(index=False))
    else:
        print("  CANDIDATI: da trasformare in strategia e validare in demo")
        print(cand.sort_values("t_weekly", ascending=False)[cols].to_string(index=False))
    if not res.empty:
        ctl = res[res.gruppo == "non selez."]
        print(f"\n  Controllo: netto medio coppie NON selezionate {ctl.net_bps.mean():+.2f} bps vs selezionate "
              f"{sres.net_bps.mean():+.2f} bps -> se simili, la selezione per cointegrazione non aggiunge valore.")

    print(f"\n=== 5) VERDETTO B: singola coppia (>= {v['pair_min_trades']} trade OOS, t settimanale >= {v['pair_t_min']}) ===")
    pv = per_pair_verdict(tr, v["pair_min_trades"], v["pair_t_min"]) if not tr.empty else pd.DataFrame()
    if not pv.empty:
        pv.to_csv(out / "per_pair.csv", index=False)
        pcols = ["pair", "rule", "n", "hold_h", "win_%", "gross_bps", "cost_bps", "net_bps", "net_pf", "t_weekly",
                 "net_1st_half", "net_2nd_half"]
        c = pv[pv.candidate]
        if c.empty:
            print("  NESSUNA COPPIA supera i criteri. Le 10 migliori (comunque insufficienti):")
            print(pv[pv.n >= v["pair_min_trades"]].sort_values("t_weekly", ascending=False).head(10)[pcols]
                  .to_string(index=False))
        else:
            print("  COPPIE CANDIDATE: da trasformare in strategia e validare in demo")
            print(c.sort_values("t_weekly", ascending=False)[pcols].to_string(index=False))
    print(f"\nFile completi in {out}")


if __name__ == "__main__":
    main()
