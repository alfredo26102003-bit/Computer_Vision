"""
CONFRONTO — PR-AUC PAI 5, la novita' contro le baseline sotto conteggio fisso.

La novita' sotto il protocollo a K fisso - obiettivo 3 senza il canale.

PERCHE' RIMISURARLA. `balanced_token_sampling` e' stata validata nel
protocollo naif, dove la maschera della bbox seleziona 16 / 36 / 64 token
secondo la classe. In quel regime la sola maschera, one-hot e senza un
pixel, da' macro-F1 0.7708: il conteggio e' quasi la risposta, e un metodo
di ribilanciamento misurato li' potrebbe stare sfruttando quello.

Sotto K fisso il conteggio non dice piu' niente, e la novita' viene
giudicata su cio' che rivendica davvero.

E CAMBIA NATURA, in meglio. Nel protocollo naif campionava sottoinsiemi di
un insieme di dimensione VARIABILE - una PAI 5 ha 64 token da cui pescare,
una PAI 3 ne ha 16 - quindi la ricchezza del campionamento era essa stessa
correlata alla classe. Sotto K = 16 tutti pescano dallo stesso numero di
token: il numero di VISTE resta l'unica cosa che varia fra le classi, che
e' esattamente cio' che la novita' dichiara di fare.

L'IPOTESI, e non e' scontata. Con n_eff = 1.01 la novita' riassegna peso
senza aggiungere informazione. Riassegnare peso aiuta quando il
classificatore e' vicino al suo tetto e si gioca sul confine di decisione;
non aiuta quando l'informazione manca. Sotto K fisso il compito e' piu'
difficile - macro-F1 scende da 0.76 a 0.56 - quindi il vantaggio potrebbe
ridursi o sparire. Nel protocollo cieco alla dimensione infatti pareggiava
(z circa 0.5).

Se pareggia, si riporta cosi': il meccanismo lo prevedeva, ed e' una
previsione confermata, non una scusa.

SOLO SU I-JEPA. Il casuale e' il pavimento dichiarato: non si tara una
baseline, la si misura una volta e la si lascia. Il suo `none` sotto
P3_K16 e' gia' noto (0.5347) e serve da riferimento.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_novita_K.py
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
from globals import IMBALANCE_METHODS, OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CHIAVI = ("macro_f1", "pr_auc_pai5", "recall_pai5", "precision_pai5",
          "f1_pai5", "quadratic_kappa")


def misura(cached, metodo, head, seeds, freno, alpha=0.5):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, metodo, head, seed=s, bts_alpha=alpha)
        per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", default="_geo_completa")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--head", default="flat")
    ap.add_argument("--metodi", nargs="+", default=IMBALANCE_METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=100)
    a = ap.parse_args()

    prot = f"P3_K{a.K}"
    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    cached = con_maschera(load_latents(a.variant, layers=a.layers, tag=a.tag), prot)
    n = cached["data"]["test"]["mask"].sum(1).float().mean()
    print(f"encoder{a.tag}, protocollo {prot}: {n:.0f} token per lesione, "
          f"uguali per tutte le classi")
    print(f"testa {a.head}, {len(a.seeds)} seed, misure sul TEST\n", flush=True)

    percorso = os.path.join(OUT_DIR, f"novita_K{a.K}{a.tag}.json")
    F = {"tag": a.tag, "K": a.K, "head": a.head, "seeds": a.seeds, "metodi": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("seeds") == a.seeds and v.get("K") == a.K and v.get("head") == a.head:
            F = v
            print(f"Riprendo: {len(F['metodi'])} metodi su disco\n")

    print(f"{'metodo':18s} {'macro-F1':>17s} {'PR-AUC5':>17s} {'F1 PAI5':>8s} "
          f"{'rec5':>7s} {'prec5':>7s}")
    print("-" * 82)
    for m in a.metodi:
        if m not in F["metodi"]:
            F["metodi"][m] = misura(cached, m, a.head, a.seeds, freno)
            save_json(F, percorso)
        r = F["metodi"][m]
        print(f"{m:18s} {r['macro_f1'][0]:9.4f}+-{r['macro_f1'][1]:.4f} "
              f"{r['pr_auc_pai5'][0]:9.4f}+-{r['pr_auc_pai5'][1]:.4f} "
              f"{r['f1_pai5'][0]:8.3f} {r['recall_pai5'][0]:7.4f} "
              f"{r['precision_pai5'][0]:7.3f}", flush=True)

    # ---- lettura ----
    n_s = len(a.seeds)
    print(f"\n{'=' * 70}\nLA NOVITA' contro le baseline, sotto {prot}\n{'=' * 70}")
    if "balanced_tokens" in F["metodi"]:
        bt = F["metodi"]["balanced_tokens"]
        for altro in [x for x in a.metodi if x != "balanced_tokens"]:
            al = F["metodi"][altro]
            riga = []
            for k in ("macro_f1", "pr_auc_pai5"):
                d = bt[k][0] - al[k][0]
                se = math.sqrt(bt[k][1] ** 2 + al[k][1] ** 2) / math.sqrt(n_s)
                riga.append(f"{k.replace('_pai5','5'):11s} {d:+.4f} (z={d/se if se>0 else 0:+5.2f})")
            print(f"  vs {altro:16s} " + "   ".join(riga))
        print("\n  Riferimento: `none` sull'encoder CASUALE sotto lo stesso")
        print("  protocollo vale 0.5347 di macro-F1 e 0.4220 di PR-AUC5.")
        print("\n  ATTESA: con n_eff = 1.01 la novita' riassegna peso senza")
        print("  aggiungere informazione. Sotto K fisso il compito e' piu'")
        print("  difficile, quindi il vantaggio puo' ridursi. Se pareggia, il")
        print("  meccanismo lo prevedeva.")

    print(f"\nRisultati in {percorso}")
