"""
PIPELINE — ricalcola dai file salvati ogni numero riportato.

Verifica - ricalcola dai file salvati ogni numero riportato.

Serve a non dover credere sulla parola a nessuno, incluse le tabelle nelle
slide: ogni riga qui sotto viene ricalcolata dai runs/results_*.json e dai
dati, non trascritta.

Uso:
    python verify_claims.py
    python verify_claims.py --criterio      # solo il confronto PAI5 vs migliore
"""

import argparse
import json
import os

import numpy as np

from utils import leggi_righe_risultati
from globals import NUM_CLASSES, OUT_DIR, PAI_GRADES

def carica(variant="vit_small", layers=None):
    """
    Carica i risultati della griglia.

    ATTENZIONE al suffisso: da quando l'estrazione puo' concatenare piu'
    profondita', run_grid salva in results_{variant}_L2-7-11.json.
    Cercando solo il nome senza suffisso, questo verificatore leggeva i
    risultati VECCHI - quelli misurati coi crop che annullavano la scala -
    e li presentava come correnti. Un fallimento silenzioso: nessun errore,
    solo numeri superati spacciati per attuali.

    Senza `layers` si prende il file piu' RECENTE fra quelli disponibili,
    e si dichiara quale.
    """
    import glob
    if layers is not None:
        suff = "_L" + "-".join(map(str, layers))
        p = os.path.join(OUT_DIR, f"results_{variant}{suff}.json")
        if not os.path.isfile(p):
            return None
    else:
        cand = glob.glob(os.path.join(OUT_DIR, f"results_{variant}*.json"))
        if not cand:
            return None
        p = max(cand, key=os.path.getmtime)
    rows = leggi_righe_risultati(p)
    for r in rows:
        r["_file"] = os.path.basename(p)
    return rows


def pavimento_macro_f1(quota_maggioritaria, k=NUM_CLASSES):
    """
    Macro-F1 di un classificatore che predice SEMPRE la maggioritaria.

    F1 della maggioritaria = 2*prec*rec/(prec+rec) con rec=1 e prec=q,
    cioe' 2q/(1+q). Le altre K-1 classi hanno F1=0. Quindi la macro-F1
    e' 2q/(1+q)/K. E' il pavimento onesto, NON l'accuracy q: quest'ultima
    e' proprio la metrica che il brief vieta come criterio.
    """
    return (2 * quota_maggioritaria / (1 + quota_maggioritaria)) / k


def riga(r):
    return (r["method"], r["head"], r["macro_f1_mean"], r["macro_f1_std"],
            r.get("f1_pai3_mean"), r.get("f1_pai4_mean"), r.get("f1_pai5_mean"),
            r["recall_pai5_mean"], r.get("precision_pai5_mean"),
            r["quadratic_kappa_mean"])


def tabella(variant="vit_small"):
    # METRICA PRIMARIA: PR-AUC su PAI 5, non la macro-F1.
    # Il Task chiede "performance on the MINORITY CLASS using
    # THRESHOLD-AGNOSTIC metrics". La macro-F1 e' nell'elenco del brief ma
    # media le tre classi con lo stesso peso, quindi non e' specifica per la
    # minoritaria; e dipende dall'argmax, quindi non e' threshold-agnostic.
    # La PR-AUC su PAI 5 (average precision) e' entrambe le cose, ed e' la
    # seconda metrica nominata dal brief: e' quella su cui si ordina.
    print("=" * 116)
    print("TABELLA COMPLETA - ricalcolata da runs/results_*.json")
    print("  metrica PRIMARIA (criterio del Task): PR-AUC su PAI 5, la minoritaria")
    print("=" * 116)
    print(f"{'metodo':16s} {'testa':8s} {'PR-AUC5':>9s} {'macroF1':>16s} "
          f"{'F1 PAI5':>8s} {'rec5':>7s} {'prec5':>7s} {'kappa':>7s}")
    tutte = []
    rows = carica(variant)
    if not rows:
        print("(nessun risultato: lanciate train_downstream.py --grid)")
        return tutte
    print(f"letto da: {rows[0].get('_file','?')}")
    for r in sorted(rows, key=lambda x: -x.get("pr_auc_pai5_mean", 0)):
        m, h, f1, sd, a, b, c, r5, p5, kp = riga(r)
        pa = r.get("pr_auc_pai5_mean")
        tutte.append(r)
        fc = "   n/d" if c is None else f"{c:8.3f}"
        print(f"{m:16s} {h:8s} "
              f"{'    n/d' if pa is None else f'{pa:9.4f}'} "
              f"{f1:8.4f}+-{sd:.4f} {fc} {r5:7.4f} "
              f"{'  n/d' if p5 is None else f'{p5:7.3f}'} {kp:7.4f}")

    # classifica per la metrica primaria, sopra tutti i bracci
    con_pa = [r for r in tutte if r.get("pr_auc_pai5_mean") is not None]
    if con_pa:
        print("\n" + "-" * 116)
        print("CLASSIFICA per PR-AUC su PAI 5 (le prime 6):")
        for r in sorted(con_pa, key=lambda x: -x["pr_auc_pai5_mean"])[:6]:
            print(f"  {r['pr_auc_pai5_mean']:.4f} +-{r.get('pr_auc_pai5_std',0):.4f}"
                  f"   {r['method']}/{r['head']}"
                  f"   (macroF1 {r['macro_f1_mean']:.4f}, recall5 {r['recall_pai5_mean']:.4f})")
    return tutte


def criterio(tutte):
    """
    Criterio operativo dichiarato: PAI 3 e PAI 4 non peggiorano, PAI 5 migliora,
    rispetto alla configurazione migliore attuale.
    """
    if not tutte:
        print("\nNessun risultato da confrontare.")
        return

    # Due confronti, e sono domande diverse:
    #
    #  (a) contro `none` dello stesso braccio: "la gestione dello
    #      sbilanciamento migliora PAI 5?" - e' la domanda scientifica, ed e'
    #      quella che l'obiettivo 4 chiede di ablare.
    #  (b) contro la riga migliore in assoluto: "esiste di meglio?" - contro
    #      il massimo non puo' esistere nulla per costruzione, quindi da solo
    #      questo confronto dice sempre NESSUNO e non informa.
    base = [r for r in tutte if r["method"] == "none" and r["head"] == "flat"]
    if base and base[0].get("f1_pai3_mean") is not None:
        _criterio_contro(base[0], tutte, "contro CE semplice (none/flat)")

    # "migliore in assoluto" ora per PR-AUC su PAI 5, la metrica primaria,
    # non piu' per macro-F1.
    chiave = ("pr_auc_pai5_mean" if all(r.get("pr_auc_pai5_mean") is not None
                                        for r in tutte) else "macro_f1_mean")
    best = max(tutte, key=lambda r: r[chiave])
    _criterio_contro(best, tutte,
                     f"contro la riga migliore per {chiave.replace('_mean','')}")


def _criterio_contro(best, candidati, titolo):
    print("\n" + "=" * 108)
    print(f"CRITERIO: stessa resa su PAI 3 e 4, migliore su PAI 5 - {titolo}")
    print("=" * 108)
    print(f"Riferimento: {best['method']} / {best['head']}")
    print(f"  macroF1={best['macro_f1_mean']:.4f}  F1 PAI3={best.get('f1_pai3_mean')}  "
          f"F1 PAI4={best.get('f1_pai4_mean')}  F1 PAI5={best.get('f1_pai5_mean')}  "
          f"recall PAI5={best['recall_pai5_mean']:.4f}")

    if best.get("f1_pai3_mean") is None:
        print("\n  ATTENZIONE: il riferimento e' stato prodotto prima che la griglia")
        print("  salvasse le F1 per classe. Rilanciate quel braccio per confrontare")
        print("  su PAI 3 e 4: senza quei numeri il criterio NON e' verificabile.")
        return

    print("\nCandidati che soddisfano il criterio (tolleranza 1 dev.std sul riferimento):")
    tol3 = best["f1_pai3_std"]
    tol4 = best["f1_pai4_std"]
    trovati = 0
    for r in candidati:
        if r.get("f1_pai5_mean") is None:
            continue
        if r is best:
            continue
        ok3 = r["f1_pai3_mean"] >= best["f1_pai3_mean"] - tol3
        ok4 = r["f1_pai4_mean"] >= best["f1_pai4_mean"] - tol4
        su5 = r["recall_pai5_mean"] > best["recall_pai5_mean"]
        if ok3 and ok4 and su5:
            trovati += 1
            d = r["recall_pai5_mean"] - best["recall_pai5_mean"]
            # Differenza significativa? Con n seed per parte, errore standard
            # della differenza ~ sqrt(s1^2/n + s2^2/n).
            n = r["n_seeds"]
            se = float(np.sqrt(r["recall_pai5_std"] ** 2 / n
                               + best["recall_pai5_std"] ** 2 / n))
            z = d / se if se > 0 else float("inf")
            print(f"  {r['method']}/{r['head']}: recall PAI5 "
                  f"{r['recall_pai5_mean']:.4f} (+{d:.4f}, ~{z:.1f} err.std)  "
                  f"F1 3/4 = {r['f1_pai3_mean']:.3f}/{r['f1_pai4_mean']:.3f}")
    if not trovati:
        print("  NESSUNO. Nessuna configurazione alza PAI 5 senza perdere su PAI 3 o 4.")
        print("  Non arrotondate questo in 'risultati promettenti': e' un no.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--criterio", action="store_true")
    a = ap.parse_args()

    tutte = tabella(a.variant)

    # Pavimenti, ricalcolati e non citati a memoria.
    from data import load_splits, parse_annotations
    recs = parse_annotations(verbose=False)
    sp = load_splits()
    by = {r["image_id"]: r for r in recs}
    y = [l["grade"] for i in sp["test"] for l in by[i]["lesions"]]
    q = max(y.count(g) for g in PAI_GRADES) / len(y)
    print("\n" + "=" * 108)
    print("PAVIMENTI (ricalcolati sul test split)")
    print("=" * 108)
    print(f"  lesioni nel test        : {len(y)}")
    print(f"  quota maggioritaria (q) : {q:.4f}   <- e' l'ACCURACY di un modello costante")
    print(f"  macro-F1 di quel modello: {pavimento_macro_f1(q):.4f}   <- 2q/(1+q)/{NUM_CLASSES}")
    print("  Confrontare la macro-F1 con q invece che con questo valore e' l'errore")
    print("  che faceva dichiarare 'al livello del caso' un encoder che imparava.")

    criterio(tutte)
