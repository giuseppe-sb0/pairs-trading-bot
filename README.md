# Pairs trading forex: diagnosi con walk-forward

Verifica se esiste un vantaggio **sfruttabile al netto dei costi** nel pairs trading fra le 9 principali
coppie forex (36 accoppiamenti), su barre H1 Dukascopy. Questa è la **fase di diagnosi**: il bot per
cTrader verrà costruito solo se il verdetto trova qualcosa.

## Ipotesi e regole (pre-registrate in `config/config.yaml`)

**Spread.** A ogni ora si stima, sulle ultime L ore, la regressione `log(A) = α + β·log(B)`. Lo z-score
è il residuo dell'ora corrente diviso per la deviazione standard dei residui della finestra. Il calcolo
è causale: usa solo dati passati.

| Elemento | Regola |
|---|---|
| Ingresso | z ≥ +soglia → vendi lo spread (short A, long β·B); z ≤ −soglia → compralo. Ingresso all'apertura dell'ora successiva, β congelato |
| Uscita | z torna a 0 (media), oppure stop a \|z\| ≥ soglia + 2, oppure dopo L ore |
| Varianti | L ∈ {120, 480} ore × soglia ∈ {2.0, 2.5} = **4 regole** |
| Nessun ingresso | 22:00–00:59 ora di Roma (rollover) |
| Costi | round trip per gamba in pip (spread + commissione Razor + slippage), più uno swap stimato di 0.5 bps per gamba al giorno |

**Selezione delle coppie (walk-forward).** Su ogni periodo di train di 8 mesi si selezionano le coppie con
cointegrazione Engle-Granger (p < 0.05) e half-life fra 5 e 500 ore. Poi si contano **solo** i trade aperti
nei 2 mesi di test successivi. Si scorre di 2 mesi, per circa 13 fold su 36 mesi di dati.

**Verdetti** (t-stat calcolato sui P&L **settimanali**: i trade di coppie con valute in comune sono correlati):
- **A, portafoglio:** tutte le coppie selezionate, per regola. Servono almeno 100 trade, netto > 0,
  t ≥ 2.5 e un risultato positivo in entrambe le metà del periodo.
- **B, singola coppia:** servono almeno 30 trade e t ≥ 3.4 (correzione di Bonferroni su 144 test),
  netto > 0 e positivo in entrambe le metà.

Le soglie sono tarate su dati sintetici: **0 falsi positivi su 30 universi casuali**, e una coppia
realmente cointegrata trovata **10 volte su 10**.

## Uso

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
```

```bash
python scripts/download_universe.py --months 36 --cache-dir ../asia-meanrev-bot/data/raw/dukascopy
```
Scarica le 9 coppie (circa 20-40 minuti la prima volta; EURUSD è già in cache) e salva le barre H1 in
`data/h1/`. Al termine controlla che i prezzi siano nella scala giusta.

```bash
python scripts/diagnose_pairs.py
```
Richiede 1-3 minuti. I report vengono salvati in `reports/diagnose_pairs/`.

## Come leggere l'output

1. **Selezione per fold:** quali coppie risultano cointegrate in ogni train. Coppie selezionate di rado
   e a caso indicano falsi positivi del test. Una coppia selezionata quasi sempre ha una relazione stabile.
2. **Risultati per regola:** il confronto `SELEZIONATE` contro `non selez.` dice se la cointegrazione
   aiuta davvero. Le unità sono bps del nozionale (1 bps su EURUSD ≈ 1.1 pip).
3. **Per coppia:** il risultato OOS di ogni coppia nei periodi in cui era stata selezionata.
4. **Verdetto A** e **5. Verdetto B.**

## Limiti noti

- Il P&L è in bps del nozionale della gamba A; le conversioni di valuta sono approssimate con i log-rendimenti.
- Costi e swap sono stime: vanno verificati in demo.
- Molte coppie forex condividono l'USD, quindi lo spread fra due coppie con l'USD equivale circa a un cross
  (ad esempio AUDUSD/NZDUSD ≈ AUDNZD). Operarlo su due gambe costa il doppio rispetto al cross diretto.
