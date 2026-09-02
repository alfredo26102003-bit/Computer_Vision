"""
MISURAZIONE — macro-F1 di sonde k-NN senza parametri addestrati.

Le sonde senza parametri: cosa contiene un vettore, prima di qualunque testa.

PERCHE' SENZA PARAMETRI. La domanda "questo encoder e' migliore?" con una
testa addestrata in mezzo ha sempre una risposta ambigua: puo' essere
l'encoder, puo' essere che la testa abbia compensato. Una sonda k-NN sulla
media dei token dentro la bbox non ha niente da addestrare - se separa le
classi, e' perche' il vettore le separa.

E' anche il modo standard di valutare rappresentazioni self-supervised
quando si vuole misurare la rappresentazione e non la capacita' della
testa.

COSA MISURA, su tre assi incrociati:

  PROTOCOLLO   geometrico (finestra fissa, il compito del brief) contro
               cieco alla dimensione (finestra 3x la bbox, ridimensionata)
  ENCODER      casuale, e i checkpoint del pre-training
  BLOCCO       2, 7, 11 separatamente, e i tre concatenati

I tre assi insieme rispondono a domande che nessuno di loro risponde da
solo: DOVE vive l'informazione (blocco), QUALE informazione e' (protocollo),
e QUANDO compare (epoche del checkpoint).

Uso:
    python exp_sonde.py                    # tutto quello che trova
    python exp_sonde.py --k 20
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluation import confusion_matrix, macro_f1, quadratic_weighted_kappa
from globals import OUT_DIR
from train_downstream import load_latents, percorso_latenti
from utils import save_json

def _epoca_di(nome, fallback):
    """
    L'epoca del checkpoint letta DAL CHECKPOINT, non scritta a mano.

    Serve perche' il numero finisce sull'asse x della curva di
    apprendimento: se il run si ferma prima delle 300 epoche configurate -
    per il cancello o per uno spegnimento, come - una costante
    scritta qui direbbe una bugia proprio nella figura che dovrebbe
    dimostrare che I-JEPA impara.
    """
    import os
    from globals import CKPT_DIR
    try:
        d = torch.load(os.path.join(CKPT_DIR, nome + ".pt"),
                       map_location="cpu", weights_only=False)
        return int(d["epoch"])
    except Exception:
        return fallback


# nome leggibile -> (tag dei latenti, epoche di pre-training)
ENCODER = {
    "casuale":  ("_casuale",       0),
    "notte":    (None,             40),     # solo cieco
    "completa": ("_geo_completa", 179),
    "mask80":   (None,            208),     # solo cieco
    "finale":   ("_geo_finale",
                 _epoca_di("ijepa_vit_small_finale_best", 300)),
}
CIECO = {"casuale": "_cieco_casuale", "notte": "_cieco_notte",
         "completa": "_cieco_completa", "mask80": "_cieco_mask80",
         "finale": "_cieco_finale"}
BLOCCHI = {"b2": slice(0, 384), "b7": slice(384, 768),
           "b11": slice(768, 1152), "tutti": slice(0, 1152)}


def media_mascherata(d, sl):
    """Media dei token dentro la bbox, per un sottoinsieme di dimensioni."""
    t, m = d["tokens"][:, :, sl].float(), d["mask"].float()
    return (t * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def knn(tr, ytr, te, k=20):
    tr = torch.nn.functional.normalize(tr, dim=-1)
    te = torch.nn.functional.normalize(te, dim=-1)
    return torch.mode(ytr[(te @ tr.T).topk(k, dim=1).indices], dim=1).values.numpy()


def sonda(tag, variant, layers, k):
    """macro-F1 e kappa per ciascun blocco, o None se i latenti mancano."""
    if tag is None or not os.path.isfile(percorso_latenti(variant, layers, tag)):
        return None
    c = load_latents(variant, layers=layers, tag=tag)
    tr, te = c["data"]["train"], c["data"]["test"]
    y = te["labels"].numpy()
    out = {}
    for nome, sl in BLOCCHI.items():
        p = knn(media_mascherata(tr, sl), tr["labels"], media_mascherata(te, sl), k)
        cm = confusion_matrix(y, p)
        out[nome] = {"macro_f1": macro_f1(cm),
                     "kappa": quadratic_weighted_kappa(y, p)}
    del c
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()

    fuori = {"k": a.k, "blocchi": list(BLOCCHI), "sonde": {}}
    for prot, mappa in (("geometrico", {n: t for n, (t, _) in ENCODER.items()}),
                        ("cieco", CIECO)):
        fuori["sonde"][prot] = {}
        for nome, tag in mappa.items():
            r = sonda(tag, a.variant, a.layers, a.k)
            if r:
                fuori["sonde"][prot][nome] = {"epoche": ENCODER[nome][1], **r}

    for prot, d in fuori["sonde"].items():
        print(f"\n=== {prot.upper()} === macro-F1 della sonda k-NN, "
              f"nessun parametro addestrato")
        print(f"{'encoder':12s} {'epoche':>7s} " +
              " ".join(f"{b:>9s}" for b in BLOCCHI))
        print("-" * 58)
        for nome, r in sorted(d.items(), key=lambda x: x[1]["epoche"]):
            print(f"{nome:12s} {r['epoche']:7d} " +
                  " ".join(f"{r[b]['macro_f1']:9.4f}" for b in BLOCCHI))

    percorso = os.path.join(OUT_DIR, "sonde_vit_small.json")
    save_json(fuori, percorso)
    print(f"\nRisultati in {percorso}")
