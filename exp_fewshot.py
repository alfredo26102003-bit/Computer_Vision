"""
CONFRONTO — macro-F1 e PR-AUC5 fra due encoder al calare delle etichette.

Valutazione few-shot: quanto vale la rappresentazione quando le etichette
costano.

PERCHE' E' IL TEST CANONICO. "Questa rappresentazione e' buona" e "ho tanti
dati" producono lo stesso numero quando i dati abbondano. Li separa solo il
regime a poche etichette: una buona rappresentazione ha bisogno di MENO
esempi per essere letta. E' l'evaluation che riportano SimCLR, MoCo, DINO e
lo stesso paper I-JEPA, esattamente per questo.

E NON AZZOPPA NESSUNO. Togliere etichette le toglie a entrambi gli encoder
allo stesso modo: non e' un handicap alla baseline, e' un regime diverso.
La differenza con l'idea di "trovare un mascheramento che faccia andare
peggio il casuale" e' tutta qui, ed e' la differenza fra misurare e
truccare.

L'INDIZIO CHE LA RENDE PROMETTENTE. Nel protocollo cieco, dopo UNA SOLA
epoca di addestramento della testa, I-JEPA era gia' a 0.5457 di macro-F1 -
sopra il massimo di sempre dell'encoder casuale (0.5416) - e raggiungeva il
95% del proprio massimo all'epoca 4 contro le 54 del casuale. La sua
rappresentazione e' gia' linearmente separabile: la testa non deve
costruire niente, deve solo leggere. Con poche etichette quel vantaggio
dovrebbe aprirsi.

IL PROTOCOLLO. Si sottocampiona SOLO il train; validation e test restano
interi. E' lo standard delle valutazioni low-shot: si misura quanto serve
per IMPARARE a leggere la rappresentazione, non quanto serve per valutarla.

Il sottocampionamento e' STRATIFICATO per classe. All'1% sono 47 lesioni su
4.719: senza stratificazione una estrazione su tre non conterrebbe nessun
PAI 5, e la misura diventerebbe rumore. E dipende dal seme, cosi' la
dispersione fra seed include anche QUALI esempi sono capitati - che a
queste numerosita' e' la fonte di variabilita' dominante.

DUE PROTOCOLLI, e il confronto fra loro e' informativo:
  P1_bbox   quello del brief, dove la maschera regala il conteggio
  P3_K16    16 token per tutti, dove il canale non c'e'

Nel primo il conteggio e' disponibile a qualunque numero di etichette:
se il casuale regge anche a poche, e' perche' sta leggendo la bbox e non
l'immagine.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_fewshot.py
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
from globals import NUM_CLASSES, OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

FRAZIONI = [0.01, 0.05, 0.10, 0.25, 1.00]
CHIAVI = ("macro_f1", "pr_auc_pai5", "f1_pai5", "quadratic_kappa")


def sottocampiona(cached, frazione, seed):
    """
    Train ridotto, STRATIFICATO per classe. Validation e test intatti.

    La stratificazione non e' un dettaglio: all'1% sono 47 lesioni, e PAI 5
    e' il 10% del train. Senza stratificare, la varianza di quante ne
    capitano dominerebbe qualunque effetto si voglia misurare.
    """
    if frazione >= 1.0:
        return cached
    tr = cached["data"]["train"]
    y = tr["labels"]
    g = torch.Generator().manual_seed(seed * 1000 + int(frazione * 10000))
    tenuti = []
    for k in range(NUM_CLASSES):
        idx = (y == k).nonzero(as_tuple=True)[0]
        n = max(2, int(round(len(idx) * frazione)))     # almeno 2 per classe
        tenuti.append(idx[torch.randperm(len(idx), generator=g)[:n]])
    sel = torch.cat(tenuti)
    sel = sel[torch.randperm(len(sel), generator=g)]

    fuori = {k: v for k, v in cached.items() if k != "data"}
    fuori["data"] = dict(cached["data"])
    fuori["data"]["train"] = {k: (v[sel] if torch.is_tensor(v) else v)
                              for k, v in tr.items()}
    return fuori


def misura(cached, frazione, seeds, head, freno):
    per_seed, n_tr = [], None
    for s in seeds:
        t0 = time.perf_counter()
        sub = sottocampiona(cached, frazione, s)
        n_tr = len(sub["data"]["train"]["labels"])
        clf, _ = train_head(sub, "none", head, seed=s)
        per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        del clf, sub
        torch.cuda.empty_cache()
        freno.pausa(t0)
    out = {k: (float(np.mean([m[k] for m in per_seed])),
               float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}
    out["n_train"] = n_tr
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", nargs="+", default=["_casuale", "_geo_completa"])
    ap.add_argument("--protocolli", nargs="+", default=["P1_bbox", "P3_K16"])
    ap.add_argument("--frazioni", type=float, nargs="+", default=FRAZIONI)
    ap.add_argument("--head", default="flat")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=100)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}")
    print(f"testa {a.head}, {len(a.seeds)} seed, train sottocampionato "
          f"STRATIFICATO, validation e test interi\n", flush=True)

    percorso = os.path.join(OUT_DIR, "fewshot_vit_small.json")
    F = {"head": a.head, "seeds": a.seeds, "frazioni": a.frazioni, "ris": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            v = json.load(f)
        if v.get("seeds") == a.seeds and v.get("head") == a.head:
            F = v
            print(f"Riprendo: {len(F['ris'])} celle su disco\n")

    for prot in a.protocolli:
        print(f"{'=' * 78}\n{prot}\n{'=' * 78}")
        print(f"{'etichette':>10s} {'n train':>8s} " +
              " ".join(f"{t:>24s}" for t in a.tag) + f" {'divario':>9s} {'z':>7s}")
        print("-" * (20 + 25 * len(a.tag) + 18))
        for fr in a.frazioni:
            r = {}
            for tag in a.tag:
                ch = f"{prot}|{tag}|{fr}"
                if ch not in F["ris"]:
                    c = load_latents(a.variant, layers=a.layers, tag=tag)
                    if prot != "P1_bbox":
                        c = con_maschera(c, prot)
                    F["ris"][ch] = misura(c, fr, a.seeds, a.head, freno)
                    del c
                    save_json(F, percorso)
                r[tag] = F["ris"][ch]
            x, y = r[a.tag[0]], r[a.tag[-1]]
            d = y["macro_f1"][0] - x["macro_f1"][0]
            se = math.sqrt(x["macro_f1"][1] ** 2 + y["macro_f1"][1] ** 2) / math.sqrt(len(a.seeds))
            print(f"{fr:9.0%} {x['n_train']:8d} " +
                  " ".join(f"{r[t]['macro_f1'][0]:9.4f}+-{r[t]['macro_f1'][1]:.4f}"
                           f" ({r[t]['pr_auc_pai5'][0]:.3f})" for t in a.tag) +
                  f" {d:+9.4f} {d/se if se>0 else 0:+7.2f}", flush=True)
        print()

    print(f"{'=' * 78}\nIL DIVARIO AL CALARE DELLE ETICHETTE\n{'=' * 78}")
    for prot in a.protocolli:
        print(f"\n{prot}")
        for fr in a.frazioni:
            x = F["ris"].get(f"{prot}|{a.tag[0]}|{fr}")
            y = F["ris"].get(f"{prot}|{a.tag[-1]}|{fr}")
            if not (x and y):
                continue
            for k, lab in (("macro_f1", "macro-F1"), ("pr_auc_pai5", "PR-AUC5")):
                d = y[k][0] - x[k][0]
                rel = d / x[k][0] * 100 if x[k][0] else float("nan")
                se = math.sqrt(x[k][1] ** 2 + y[k][1] ** 2) / math.sqrt(len(a.seeds))
                print(f"  {fr:5.0%} {lab:10s} {d:+.4f} ({rel:+6.1f}%)  z = "
                      f"{d/se if se>0 else 0:+5.2f}")

    print(f"\nRisultati in {percorso}")
