"""
CONFRONTO — PR-AUC PAI 5 al variare di alpha, il parametro della novita'.

Ablation su alpha della novita' - obiettivo 4.

alpha regola quante viste riceve ogni classe: n_c = ceil((max/n_c)^alpha).
Con lo sbilanciamento reale del train (3017/1229/473) il quadro e':

    alpha 0.00  ->  [1,1,1]   4719 istanze   = identico a `none`
    alpha 0.25  ->  [1,2,2]   6421
    alpha 0.50  ->  [1,2,3]   6894           <- il default, mai ottimizzato
    alpha 0.75  ->  [1,2,5]   7840
    alpha 1.00  ->  [1,3,7]  10015           <- pareggio effettivo

Ad alpha 0.5 la novita' NON pareggia: per equalizzare servirebbero
[1, 2.5, 6.4] viste, e PAI 5 resta sotto-rappresentata di circa un fattore
due. E' ad alpha 1.0 che entra nello stesso regime di `oversample` (9051
istanze) - e li' il confronto diventa alla pari: stesso numero di istanze,
ma sottoinsiemi di token GENUINI invece di duplicati identici della stessa
lesione. E' l'argomento piu' forte per l'obiettivo 3, perche' isola
esattamente cio' che rende diversa la novita'.

PROTOCOLLO IN DUE FASI
  1. screening su 3 seed, tutte le alpha
  2. le 2 migliori rimisurate su 5 seed DISGIUNTI

I seed della fase 2 non riusano quelli della fase 1: selezionare su una
misura rumorosa e poi riportare quella stessa misura gonfia il vincitore.
Con 4 candidati e una deviazione di ~0.008, il massimo di 4 estrazioni e'
distorto verso l'alto di circa una deviazione - dello stesso ordine
dell'effetto cercato. Seed disgiunti eliminano la distorsione.

FRENO A CICLO DI LAVORO, non termostato. Il termostato controllava la
temperatura solo FRA un seed e l'altro, e un seed dura minuti: la GPU
arrivava a 88 C - la temperatura a cui questa macchina si e' spenta - senza
che il controllo scattasse. Il freno invece agisce a ogni singolo
addestramento, e a carico 70 ha retto quattro ore a 71-72 C.
"""

import json
import time

import numpy as np
import torch

from evaluation import evaluate_split
from train_downstream import load_latents, train_head
from utils import Freno, leggi_righe_risultati

CARICO = 70        # GPU attiva il 70% del tempo
TAG = "_casuale"   # l'encoder dove vive il picco della novita'
ALPHAS = [0.25, 0.50, 0.75, 1.00]
SEED_SCREENING = [0, 1, 2]
SEED_FINALI = [10, 11, 12, 13, 14]     # disgiunti da quelli sopra
HEAD = "flat"


def misura(cached, alpha, seeds, freno):
    """Media e deviazione su `seeds`, valutate sul TEST."""
    f1, pr, rec, prec = [], [], [], []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, "balanced_tokens", HEAD, seed=s,
                            bts_alpha=alpha)
        r = evaluate_split(clf, cached["data"]["test"], HEAD)
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
        f1.append(r["macro_f1"])
        pr.append(r["pr_auc_pai5"])
        rec.append(r["recall_pai5"])
        prec.append(r["precision_pai5"])
    return {k: (float(np.mean(v)), float(np.std(v)))
            for k, v in (("macro_f1", f1), ("pr_auc", pr),
                         ("recall5", rec), ("prec5", prec))}


def riga(a, m):
    return (f"{a:6.2f} {m['pr_auc'][0]:9.4f}+-{m['pr_auc'][1]:.4f} "
            f"{m['macro_f1'][0]:9.4f}+-{m['macro_f1'][1]:.4f} "
            f"{m['recall5'][0]:9.4f} {m['prec5'][0]:8.3f}")


if __name__ == "__main__":
    freno = Freno(CARICO)
    print(f"[freno] {freno}", flush=True)
    t0 = time.time()
    cached = load_latents("vit_small", layers=[2, 7, 11], tag=TAG)
    print(f"latenti caricati in {time.time()-t0:.0f}s", flush=True)

    intestazione = (f"{'alpha':>6s} {'PR-AUC5':>17s} {'macro-F1':>17s} "
                    f"{'recall5':>9s} {'prec5':>8s}")

    print(f"\nFASE 1 - screening, {len(SEED_SCREENING)} seed, encoder{TAG}")
    print(intestazione)
    print("-" * 62)
    scr = {}
    for a in ALPHAS:
        scr[a] = misura(cached, a, SEED_SCREENING, freno)
        print(riga(a, scr[a]), flush=True)

    migliori = sorted(ALPHAS, key=lambda a: -scr[a]["pr_auc"][0])[:2]
    print(f"\nFASE 2 - le due migliori {migliori}, "
          f"{len(SEED_FINALI)} seed DISGIUNTI da quelli sopra")
    print(intestazione)
    print("-" * 62)
    fin = {}
    for a in migliori:
        fin[a] = misura(cached, a, SEED_FINALI, freno)
        print(riga(a, fin[a]), flush=True)

    # riferimenti gia' misurati sullo stesso encoder, stessa testa, stesso test
    rif = {r["method"]: r for r in
           leggi_righe_risultati(f"runs/results_vit_small_L2-7-11{TAG}.json")
           if r["head"] == HEAD}
    print(f"\nCONFRONTO (encoder{TAG}, testa {HEAD}, test)")
    print(f"{'':30s} {'PR-AUC5':>10s} {'macro-F1':>10s} {'recall5':>9s} {'prec5':>8s}")
    print("-" * 70)
    for n in ("none", "oversample", "balanced_tokens"):
        r = rif[n]
        print(f"  {n + ' (5 seed, gia fatto)':28s} "
              f"{r['pr_auc_pai5_mean']:10.4f} {r['macro_f1_mean']:10.4f} "
              f"{r['recall_pai5_mean']:9.4f} {r['precision_pai5_mean']:8.3f}")
    for a in migliori:
        m = fin[a]
        print(f"  {'alpha ' + format(a, '.2f') + ' (5 seed nuovi)':28s} "
              f"{m['pr_auc'][0]:10.4f} {m['macro_f1'][0]:10.4f} "
              f"{m['recall5'][0]:9.4f} {m['prec5'][0]:8.3f}")

    with open(f"runs/sweep_alpha_vit_small_L2-7-11{TAG}.json", "w",
              encoding="utf-8") as f:
        json.dump({"screening": {str(k): v for k, v in scr.items()},
                   "finali": {str(k): v for k, v in fin.items()},
                   "seed_screening": SEED_SCREENING,
                   "seed_finali": SEED_FINALI}, f, indent=2)

    print(f"\n[freno] pausa accumulata: {freno.pausa_totale/60:.1f} min")
