"""
DIAGNOSI — origine di uno scarto di 0,0140 fra misure della stessa cella.

Da dove viene lo scarto di 0,0140 sul braccio casuale.

IL FATTO. La stessa misura - encoder casuale, protocollo del brief, testa
flat, metodo none, 5 seed, test - vale 0,7565 in tre file di agosto
(griglia, few-shot a frazione 1.0, exp_rumore) e 0,7705 misurata oggi da
exp_fixedk. In PR-AUC di PAI 5: 0,8676 contro 0,8758.

PERCHE' NON E' UN DETTAGLIO. Quello scarto era stato
diagnosticato come "tre celle stantie" e la griglia era stata rifatta.
Il valore vecchio si e' riprodotto alla quarta cifra da uno script
diverso: se si riproduce non era stantio, e la correzione del 27 potrebbe
aver sostituito numeri buoni con altri.

IPOTESI GIA' ESCLUSE
  versione di torch  falsificata. 2.12 e 2.13 danno risultati BIT-IDENTICI
                     su layer_norm, matmul, gelu, media mascherata, logit e
                     cross entropy: dodici cifre uguali.
  file dei latenti   `latents_..._casuale.pt` non cambia.

COSA DISTINGUE QUESTO ESPERIMENTO. Tre bracci, stesso interprete, stessi
semi, stesso file di latenti caricato UNA volta:

  A_diretto    train_head sui latenti come escono da load_latents.
               E' il percorso di exp_fewshot e della griglia.
  B_maschera   train_head sul dizionario ricostruito da con_maschera con
               protocollo P1_bbox, che NON cambia la maschera ma ricrea i
               dict. E' il percorso di exp_fixedk.
  A_ripetuto   di nuovo A, identico, piu' tardi.

LETTURA DEI RISULTATI
  A != A_ripetuto   il non-determinismo della GPU e' la causa, e allora
                    nessuno dei due valori e' "quello giusto": vanno
                    riportati con la loro dispersione fra ripetizioni.
  A == A_ripetuto e A != B    la causa e' con_maschera, cioe' la
                    ricostruzione del dizionario. Da capire cosa perde.
  A == B == 0,7705  la causa non e' in questo file: sta in run_grid, e
                    l'indagine si sposta li'.

Prima di misurare si verifica che i tensori dei due bracci siano gli
STESSI oggetti: se lo sono, qualunque differenza a valle non puo' venire
dai dati.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_scarto.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from exp_fixedk import con_maschera
from globals import OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CHIAVI = ("macro_f1", "pr_auc_pai5", "quadratic_kappa")


def identita(a, b):
    """I due dizionari puntano agli stessi tensori? Non 'uguali': gli stessi."""
    fuori = {}
    for split in ("train", "val", "test"):
        da, db = a["data"][split], b["data"][split]
        fuori[split] = {
            "tokens_stesso_oggetto": da["tokens"] is db["tokens"],
            "mask_stesso_oggetto": da["mask"] is db["mask"],
            "labels_uguali": bool(torch.equal(da["labels"], db["labels"])),
            "mask_uguale": bool(torch.equal(da["mask"], db["mask"])),
        }
    return fuori


def misura(cached, seeds, freno, etichetta):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, "none", "flat", seed=s)
        m = evaluate_split(clf, cached["data"]["test"], "flat")
        per_seed.append(m)
        print(f"    {etichetta} seme {s}: macro_f1={m['macro_f1']:.6f}  "
              f"pr_auc5={m['pr_auc_pai5']:.6f}", flush=True)
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}, \
           [float(m["macro_f1"]) for m in per_seed]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=100)
    ap.add_argument("--esito", default=None)
    a = ap.parse_args()

    freno = Freno(a.carico)
    percorso = a.esito or os.path.join(OUT_DIR, "scarto_vit_small.json")
    print(f"[freno] {freno}")
    print(f"tag {a.tag}, semi {a.seeds}, metodo none, testa flat, test\n")

    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)
    mascherato = con_maschera(cached, "P1_bbox")

    ident = identita(cached, mascherato)
    print("I DATI SONO GLI STESSI?")
    for split, v in ident.items():
        print(f"  {split:6s} {v}")
    tutti_stessi = all(v["tokens_stesso_oggetto"] and v["mask_uguale"]
                       for v in ident.values())
    print(f"  -> {'SI: i tensori coincidono' if tutti_stessi else 'NO: i dati differiscono, ed e la spiegazione'}\n")

    fuori = {"tag": a.tag, "seeds": a.seeds, "identita": ident,
             "atteso_agosto": {"macro_f1": 0.7565, "pr_auc_pai5": 0.8676},
             "atteso_fixedk_oggi": {"macro_f1": 0.7705, "pr_auc_pai5": 0.8758},
             "bracci": {}}

    for nome, dati in (("A_diretto", cached),
                       ("B_maschera", mascherato),
                       ("A_ripetuto", cached)):
        print(f"{nome}:")
        agg, singoli = misura(dati, a.seeds, freno, nome)
        fuori["bracci"][nome] = {**agg, "per_seme": singoli}
        print(f"  -> macro_f1 {agg['macro_f1'][0]:.6f} +- {agg['macro_f1'][1]:.6f}"
              f"   pr_auc5 {agg['pr_auc_pai5'][0]:.6f}\n", flush=True)
        save_json(fuori, percorso)

    # ------------------------------------------------------------ lettura
    A = fuori["bracci"]["A_diretto"]
    B = fuori["bracci"]["B_maschera"]
    R = fuori["bracci"]["A_ripetuto"]
    print("=" * 74)
    dAR = abs(A["macro_f1"][0] - R["macro_f1"][0])
    dAB = abs(A["macro_f1"][0] - B["macro_f1"][0])
    identici = all(x == y for x, y in zip(A["per_seme"], R["per_seme"]))
    print(f"A contro A_ripetuto : {dAR:.6f}   per-seme identici: {identici}")
    print(f"A contro B_maschera : {dAB:.6f}")
    print()
    if not identici:
        print("VERDETTO: il calcolo NON e' riproducibile fra due esecuzioni")
        print("identiche. La causa e' il non-determinismo della GPU, e nessuno")
        print("dei due valori storici e' 'quello giusto': vanno riportati con")
        print("la dispersione fra ripetizioni, non solo fra semi.")
    elif dAB > 1e-6:
        print("VERDETTO: con_maschera cambia il risultato pur non cambiando la")
        print("maschera. La differenza sta nella ricostruzione del dizionario.")
    else:
        print("VERDETTO: i due percorsi coincidono. Lo scarto non nasce qui:")
        print("l'indagine si sposta su run_grid, che e' l'unico percorso")
        print("rimasto fra quelli che hanno prodotto 0,7565.")
    print(f"\nvalore di agosto 0,7565   valore di oggi 0,7705")
    print(f"misurato ora: A={A['macro_f1'][0]:.4f}  B={B['macro_f1'][0]:.4f}")
    print(f"\nRisultati in {percorso}")
