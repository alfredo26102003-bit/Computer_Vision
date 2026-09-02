"""
CONFRONTO — macro-F1 e PR-AUC5 fra encoder, sui cinque protocolli di mascheramento.

Fixed-K nella pipeline vera: attention pooling addestrato piu' testa.

PERCHE' ESISTE, dato che exp_mascheramento.py misura gia' gli stessi
protocolli. Quello usa una sonda k-NN sulla media dei token: zero
parametri, ottima per dire cosa CONTIENE il vettore. Ma la pipeline del
progetto ha un attention pooling da 5,3M parametri e una testa addestrata,
e non e' scontato che si comportino allo stesso modo - una testa capace
puo' recuperare un segnale che il k-NN non vede.

Il risultato che va in presentazione dev'essere misurato dove il progetto
lavora davvero. Questo file lo fa.

L'IPOTESI. Nel protocollo del brief la maschera della bbox seleziona i
token da aggregare, e il loro numero dipende dalla dimensione della
lesione: 16 / 36 / 64 per PAI 3 / 4 / 5. La maschera da sola, one-hot e
senza un solo pixel, da' macro-F1 0,7708 con la sonda k-NN. Se quel canale
e' cio' che sostiene l'encoder casuale, toglierlo deve farlo crollare - e
lasciare in piedi I-JEPA, che l'ha in parte disimparato.

COSA CAMBIA E COSA NO. Cambia SOLO la maschera. Stesso crop 224 px, stessa
risoluzione, stessa scala apparente della lesione, stessi token
dell'encoder, stessa architettura di pooling e testa, stessi seed. La bbox
resta usata per LOCALIZZARE: nei protocolli a K fisso si prendono i K token
piu' vicini al suo centro.

Non stiamo togliendo la dimensione dall'immagine - sarebbe sbagliato,
perche' la dimensione della lesione e' informazione clinica vera. Stiamo
togliendo la dimensione come METADATO DEL PROTOCOLLO.

I PROTOCOLLI

    P1     maschera bbox              16/36/64   quello del brief
    P2b    griglia 6x6 fissa                36   stesse posizioni per tutti
    P3_K   i K piu' vicini al centro         K   K uguale per ogni classe

Le maschere si sostituiscono in memoria dentro il dizionario dei latenti:
i token non cambiano, quindi non serve riestrarre niente ne' scrivere
nuovi file da 3 GB.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_fixedk.py
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

GRID = 14
YY, XX = torch.meshgrid(torch.arange(GRID).float(),
                        torch.arange(GRID).float(), indexing="ij")
CHIAVI = ("macro_f1", "pr_auc_pai5", "recall_pai5", "precision_pai5",
          "f1_pai5", "quadratic_kappa")


def centri(mask):
    mk = mask.view(-1, GRID, GRID)
    out = []
    for i in range(mk.shape[0]):
        r = mk[i].any(1).nonzero().flatten()
        c = mk[i].any(0).nonzero().flatten()
        out.append(((r[0] + r[-1]) / 2.0, (c[0] + c[-1]) / 2.0))
    return out


def maschera(nome, d):
    """La maschera del protocollo. I token non vengono toccati."""
    if nome == "P1_bbox":
        return d["mask"]
    if nome == "P2b_griglia_fissa":
        sel = torch.arange(GRID)[::2][:6]
        m = torch.zeros(GRID, GRID, dtype=torch.bool)
        m[sel[:, None], sel[None, :]] = True
        return m.flatten()[None].expand(d["mask"].shape[0], -1).clone()
    if nome.startswith("P3_K"):
        K = int(nome.split("K")[1])
        out = torch.zeros(d["mask"].shape[0], GRID * GRID, dtype=torch.bool)
        for i, (cy, cx) in enumerate(centri(d["mask"])):
            dd = ((YY - cy) ** 2 + (XX - cx) ** 2).flatten()
            out[i, dd.argsort()[:K]] = True
        return out
    raise ValueError(nome)


def con_maschera(cached, nome):
    """Copia superficiale con la maschera sostituita: i token restano gli stessi."""
    fuori = {"embed_dim": cached["embed_dim"], "grid": cached["grid"],
             "layers": cached.get("layers"), "data": {}}
    for split, d in cached["data"].items():
        fuori["data"][split] = {**d, "mask": maschera(nome, d)}
    return fuori


def misura(cached, seeds, freno, head="flat", metodo="none", split="test"):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, metodo, head, seed=s)
        per_seed.append(evaluate_split(clf, cached["data"][split], head))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", nargs="+", default=["_casuale", "_geo_completa"])
    ap.add_argument("--protocolli", nargs="+",
                    default=["P1_bbox", "P2b_griglia_fissa",
                             "P3_K16", "P3_K36", "P3_K64"])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["val", "test"],
                    help="dove si valuta. 'val' per DECIDERE, 'test' per RIPORTARE")
    ap.add_argument("--esito", default=None,
                    help="file dei risultati; se assente, fixedk_<variant>.json")
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    print(f"testa {a.head}, metodo none, {len(a.seeds)} seed, "
          f"misure su {a.split.upper()}"
          + ("   <- split di SCELTA, non si riporta" if a.split == "val" else ""))
    print("cambia SOLO la maschera: stesso crop, stessi token, stessa "
          "architettura\n")

    percorso = a.esito or os.path.join(OUT_DIR, "fixedk_vit_small.json")
    fuori = {"head": a.head, "seeds": a.seeds, "split": a.split, "risultati": {}}
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            v = json.load(f)
        if (v.get("seeds") == a.seeds and v.get("head") == a.head
                and v.get("split", "test") == a.split):
            fuori = v
            print(f"Riprendo: {len(fuori['risultati'])} celle su disco\n")

    print(f"{'protocollo':22s} {'token':>6s} " +
          " ".join(f"{t:>26s}" for t in a.tag) + f" {'divario':>9s}")
    print("-" * (30 + 27 * len(a.tag) + 10))
    for prot in a.protocolli:
        r, ntok = {}, None
        for tag in a.tag:
            chiave = f"{prot}|{tag}"
            if chiave not in fuori["risultati"]:
                cached = load_latents(a.variant, layers=a.layers, tag=tag)
                mod = con_maschera(cached, prot)
                ntok = float(mod["data"][a.split]["mask"].sum(1).float().mean())
                fuori["risultati"][chiave] = {
                    "token_medi": ntok,
                    **misura(mod, a.seeds, freno, a.head, split=a.split)}
                del cached, mod
                save_json(fuori, percorso)
            r[tag] = fuori["risultati"][chiave]
            ntok = r[tag]["token_medi"]
        div = r[a.tag[-1]]["macro_f1"][0] - r[a.tag[0]]["macro_f1"][0]
        print(f"{prot:22s} {ntok:6.0f} " +
              " ".join(f"{r[t]['macro_f1'][0]:9.4f}+-{r[t]['macro_f1'][1]:.4f}"
                       f" ({r[t]['pr_auc_pai5'][0]:.4f})" for t in a.tag) +
              f" {div:+9.4f}", flush=True)

    # ---- lettura ----
    import math
    print(f"\n{'=' * 78}\nLA PREVISIONE: il casuale crolla, I-JEPA tiene\n{'=' * 78}")
    base = fuori["risultati"].get(f"P1_bbox|{a.tag[0]}")
    for prot in a.protocolli:
        x = fuori["risultati"].get(f"{prot}|{a.tag[0]}")
        y = fuori["risultati"].get(f"{prot}|{a.tag[-1]}")
        if not (x and y):
            continue
        d = y["macro_f1"][0] - x["macro_f1"][0]
        se = math.sqrt(x["macro_f1"][1] ** 2 + y["macro_f1"][1] ** 2) / math.sqrt(len(a.seeds))
        caduta = x["macro_f1"][0] - base["macro_f1"][0] if base else float("nan")
        print(f"  {prot:22s} I-JEPA - casuale {d:+.4f} (z = {d/se if se>0 else 0:+.2f})"
              f"   il casuale rispetto a P1: {caduta:+.4f}")

    print(f"\nRisultati in {percorso}")
