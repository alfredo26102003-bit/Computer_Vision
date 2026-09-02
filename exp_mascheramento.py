"""
CONFRONTO — macro-F1 k-NN fra protocolli di mascheramento.

Il canale nascosto del protocollo: l'estensione della maschera.

LA SCOPERTA. Nel protocollo del brief la bounding box seleziona i token da
aggregare, e il loro NUMERO dipende dalla dimensione della lesione: 16 / 36
/ 64 token mediani per PAI 3 / 4 / 5. Quel numero non e' un dettaglio
implementativo, e' un canale informativo che arriva alla testa insieme ai
token - e siccome il grado PAI e' quasi tutto dimensione, e' quasi la
risposta.

Quanto quasi: la MASCHERA DA SOLA, trasformata in un vettore one-hot senza
un solo pixel dell'immagine, da' macro-F1 0.7708 - piu' del vettore
completo dell'encoder casuale (0.7638). Nel protocollo geometrico non e'
l'encoder a classificare: e' la bounding box.

PERCHE' QUESTO SPIEGA IL RISULTATO CHE SEMBRAVA NEGATIVO. L'encoder
casuale conserva quel canale intatto. Il pre-training, costruendo
invarianza, in parte lo scarta - e quindi appare PEGGIORE in un protocollo
dove il canale e' quasi tutta la risposta. Non e' che il pre-training non
impari: e' che disimpara una scorciatoia che il protocollo premia.

CIO' CHE NON FUNZIONA, ed e' istruttivo. Pareggiare il CONTEGGIO non basta:
RoIAlign (He et al. 2017) ricampiona la regione su una griglia fissa k x k,
quindi ogni lesione da' esattamente k^2 token, e il casuale resta a 0.7689.
Il motivo e' che RoIAlign campiona sempre DENTRO la bbox: il conteggio e'
pareggiato ma l'ESTENSIONE della regione aggregata no, e con essa la
dispersione dei positional embedding. Il conteggio era un correlato, non la
causa: la causa e' l'estensione.

CIO' CHE FUNZIONA. Rendere la maschera indipendente dalla dimensione della
bbox, tenendo tutto il resto identico - stesso crop 224 px, stessa
risoluzione, stessa scala apparente della lesione, nessun `geom`. La bbox
resta usata per LOCALIZZARE la lesione, non per dire quanto e' grande.

    P1   maschera bbox              16/36/64 token   il protocollo attuale
    P2a  K a caso su tutti i token  36 token         ignora del tutto la bbox
    P2b  griglia 6x6 fissa          36 token         stesse posizioni per tutti
    P2c  blocco centrale 6x6 fisso  36 token         centrato, esteso uguale
    P2d  K rimescolato fra classi   ~29 token        isola SOLO il conteggio
    P3   K fisso, i piu' vicini al centro della bbox, K uguale per tutti

Uso:
    python exp_mascheramento.py
"""

import argparse
import json
import os

import torch

from evaluation import confusion_matrix, macro_f1, quadratic_weighted_kappa
from globals import OUT_DIR
from train_downstream import load_latents
from utils import save_json

GRID = 14
YY, XX = torch.meshgrid(torch.arange(GRID).float(),
                        torch.arange(GRID).float(), indexing="ij")


def knn(tr, ytr, te, k=20):
    tr = torch.nn.functional.normalize(tr, dim=-1)
    te = torch.nn.functional.normalize(te, dim=-1)
    return torch.mode(ytr[(te @ tr.T).topk(k, dim=1).indices], dim=1).values.numpy()


def pool(t, m):
    m = m.float()
    return (t * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def centri(mask):
    """Centro della bbox di ogni campione, in coordinate della griglia."""
    mk = mask.view(-1, GRID, GRID)
    out = []
    for i in range(mk.shape[0]):
        r = mk[i].any(1).nonzero().flatten()
        c = mk[i].any(0).nonzero().flatten()
        out.append(((r[0] + r[-1]) / 2.0, (c[0] + c[-1]) / 2.0))
    return out


def piu_vicini(mask, K):
    """I K token piu' vicini al centro della bbox. K uguale per tutte le classi."""
    cen = centri(mask)
    out = torch.zeros(mask.shape[0], GRID * GRID, dtype=torch.bool)
    for i, (cy, cx) in enumerate(cen):
        d = ((YY - cy) ** 2 + (XX - cx) ** 2).flatten()
        out[i, d.argsort()[:K]] = True
    return out


def maschere(nome, d, g):
    if nome == "P1_bbox":
        return d["mask"]
    if nome == "P2a_casuale_K36":
        s = torch.rand(d["mask"].shape[0], GRID * GRID, generator=g)
        return s.argsort(1).argsort(1) < 36
    if nome == "P2b_griglia_fissa":
        sel = torch.arange(GRID)[::2][:6]
        m = torch.zeros(GRID, GRID, dtype=torch.bool)
        m[sel[:, None], sel[None, :]] = True
        return m.flatten()[None].expand(d["mask"].shape[0], -1)
    if nome == "P2c_centrale_fisso":
        m = torch.zeros(GRID, GRID, dtype=torch.bool)
        m[4:10, 4:10] = True
        return m.flatten()[None].expand(d["mask"].shape[0], -1)
    if nome == "P2d_conteggio_scorrelato":
        # Il conteggio viene rimescolato FRA i campioni, quindi la sua
        # distribuzione diventa identica per le tre classi. Isola il
        # conteggio da tutto il resto: centro, scala e contenuto restano.
        n = d["mask"].sum(1)
        K = n[torch.randperm(n.shape[0], generator=g)]
        cen = centri(d["mask"])
        out = torch.zeros(d["mask"].shape[0], GRID * GRID, dtype=torch.bool)
        for i, (cy, cx) in enumerate(cen):
            dd = ((YY - cy) ** 2 + (XX - cx) ** 2).flatten()
            out[i, dd.argsort()[:int(K[i])]] = True
        return out
    if nome.startswith("P3_K"):
        return piu_vicini(d["mask"], int(nome.split("K")[1]))
    raise ValueError(nome)


PROTOCOLLI = ["P1_bbox", "P2a_casuale_K36", "P2b_griglia_fissa",
              "P2c_centrale_fisso", "P2d_conteggio_scorrelato"] + \
             [f"P3_K{K}" for K in (9, 16, 36, 64, 100)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", nargs="+", default=["_casuale", "_geo_completa"])
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()

    D = {}
    for tag in a.tag:
        c = load_latents(a.variant, layers=a.layers, tag=tag)
        D[tag] = (c["data"]["train"], c["data"]["test"])

    # La maschera da sola, senza un solo pixel: e' il numero che spiega tutto.
    tr, te = D[a.tag[0]]
    y = te["labels"].numpy()
    def one_hot(d):
        m = d["mask"].float()
        return m / m.sum(1, keepdim=True).clamp(min=1)
    p = knn(one_hot(tr), tr["labels"], one_hot(te), a.k)
    solo_maschera = macro_f1(confusion_matrix(y, p))
    print(f"LA MASCHERA DA SOLA, one-hot, nessun pixel: macro-F1 "
          f"{solo_maschera:.4f}\n")

    fuori = {"solo_maschera_macro_f1": solo_maschera, "protocolli": {}}
    print(f"{'protocollo':28s} {'token':>7s} " +
          " ".join(f"{t:>11s}" for t in a.tag) + f" {'divario':>9s}")
    print("-" * (36 + 12 * len(a.tag) + 10))
    for prot in PROTOCOLLI:
        r, ntok = {}, None
        for tag in a.tag:
            tr, te = D[tag]
            g1 = torch.Generator().manual_seed(0)
            g2 = torch.Generator().manual_seed(1)
            mtr, mte = maschere(prot, tr, g1), maschere(prot, te, g2)
            ntok = float(mte.sum(1).float().mean())
            y = te["labels"].numpy()
            p = knn(pool(tr["tokens"].float(), mtr), tr["labels"],
                    pool(te["tokens"].float(), mte), a.k)
            cm = confusion_matrix(y, p)
            r[tag] = {"macro_f1": macro_f1(cm),
                      "kappa": quadratic_weighted_kappa(y, p)}
        fuori["protocolli"][prot] = {"token_medi": ntok, **r}
        div = r[a.tag[-1]]["macro_f1"] - r[a.tag[0]]["macro_f1"]
        print(f"{prot:28s} {ntok:7.0f} " +
              " ".join(f"{r[t]['macro_f1']:11.4f}" for t in a.tag) +
              f" {div:+9.4f}")

    percorso = os.path.join(OUT_DIR, "mascheramento_vit_small.json")
    save_json(fuori, percorso)
    print(f"\nRisultati in {percorso}")
