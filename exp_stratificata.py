"""
CONFRONTO — macro-F1 fra encoder, stratificata per dimensione della lesione.

Chi prende i casi in cui la dimensione inganna - analisi stratificata.

L'IDEA, che viene da una domanda giusta. Il grado PAI e' quasi tutto
dimensione della radiotrasparenza: due soglie sul lato della bbox danno
macro-F1 0.7567 senza alcuna rete. Ma "quasi tutto" non e' "tutto", e i
casi che restano fuori sono quelli dove la dimensione MENTE: una PAI 3 piu'
grande del normale, una PAI 5 piccola.

Su quei casi il misuratore d'area sbaglia per costruzione, e deve
intervenire l'aspetto - quanto e' scura, quanto e' netto il bordo, com'e'
l'osso attorno. Che e' esattamente cio' che il pre-training ha imparato:
sonda k-NN senza alcun parametro addestrato, protocollo cieco alla
dimensione, macro-F1 0.6092 per l'encoder addestrato contro 0.3173 per
quello casuale.

DEFINIZIONE DI "INGANNEVOLE", e non e' arbitraria. Si adatta sul TRAIN la
regola a due soglie sul lato della bbox - la stessa che da' il pavimento
geometrico - e si applica al test. I casi ingannevoli sono ESATTAMENTE
quelli che quella regola sbaglia. Non una soglia scelta a mano: la
partizione la decide il pavimento stesso.

    tipici      = la geometria da sola basta
    ingannevoli = la geometria da sola sbaglia

LA PREVISIONE, falsificabile. Sui tipici i due encoder pareggiano, perche'
li' basta la dimensione e ce l'hanno entrambi. Sugli ingannevoli l'encoder
addestrato tiene meglio, perche' ha l'aspetto e il casuale no.

Se non succede, la complementarita' fra i due encoder e' piu' debole di
quanto la sonda k-NN suggerisca, e va detto.

NOTA SUL PROTOCOLLO. Si misura sul protocollo GEOMETRICO, cioe' quello del
brief: la domanda e' quali casi si sbagliano nel compito vero, non in
un'ablation.

Uso:
    python sorveglia.py --tetto 95 --tetto-temp 86 -- python exp_stratificata.py
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from evaluation import confusion_matrix, macro_f1, quadratic_weighted_kappa
from globals import DEVICE, NUM_CLASSES, OUT_DIR, PAI_GRADES
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 100
SEEDS = [0, 1, 2, 3, 4]


def regola_due_soglie(lato_train, y_train, lato_test):
    """
    Le due soglie sul lato della bbox che massimizzano la macro-F1 sul TRAIN.

    E' il pavimento geometrico del progetto, rifatto qui per avere le
    predizioni per campione invece del solo aggregato. Adattata sul train e
    applicata al test: adattarla sul test renderebbe "ingannevole" una
    partizione scelta guardando le risposte.
    """
    cand = np.percentile(lato_train, np.arange(2, 99, 1))
    meglio, soglie = -1.0, (0.0, 0.0)
    for a, b in itertools.combinations(cand, 2):
        p = np.digitize(lato_train, [a, b])
        f = macro_f1(confusion_matrix(y_train, p))
        if f > meglio:
            meglio, soglie = f, (a, b)
    return soglie, meglio, np.digitize(lato_test, list(soglie))


@torch.no_grad()
def predizioni(clf, split, head_type="flat", batch=256):
    """Predizione per campione, senza aggregare."""
    from evaluation import e_ordinale
    clf.eval()
    out = []
    for i in range(0, len(split["labels"]), batch):
        tok = split["tokens"][i:i + batch].float().to(DEVICE)
        msk = split["mask"][i:i + batch].to(DEVICE)
        logits, _, _ = clf(tok, token_mask=msk)
        if e_ordinale(head_type):
            out.append((torch.sigmoid(logits) > 0.5).sum(1).cpu())
        else:
            out.append(logits.argmax(-1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--tag", nargs="+", default=["_casuale", "_geo_completa"])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--metodo", default="none")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--carico", type=int, default=CARICO)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}\n", flush=True)

    # --- la partizione, decisa dal pavimento geometrico ---
    c0 = load_latents(a.variant, layers=a.layers, tag=a.tag[0])
    tr, te = c0["data"]["train"], c0["data"]["test"]
    lato_tr = (tr["geom"][:, 2] * 200).numpy()      # lato max in pixel nativi
    lato_te = (te["geom"][:, 2] * 200).numpy()
    y_tr, y_te = tr["labels"].numpy(), te["labels"].numpy()

    soglie, f1_train, pred_geo = regola_due_soglie(lato_tr, y_tr, lato_te)
    geo_ok = pred_geo == y_te
    f1_test_geo = macro_f1(confusion_matrix(y_te, pred_geo))

    print(f"REGOLA A DUE SOGLIE sul lato della bbox, adattata sul train")
    print(f"  soglie: {soglie[0]:.0f} px e {soglie[1]:.0f} px")
    print(f"  macro-F1 train {f1_train:.4f}   test {f1_test_geo:.4f}")
    print(f"\nPARTIZIONE DEL TEST")
    print(f"  tipici      (la geometria basta)   {geo_ok.sum():4d}")
    print(f"  ingannevoli (la geometria sbaglia) {(~geo_ok).sum():4d}")
    print(f"  composizione degli ingannevoli per classe vera:")
    for k, g in enumerate(PAI_GRADES):
        n = int(((~geo_ok) & (y_te == k)).sum())
        tot = int((y_te == k).sum())
        lato_ing = lato_te[(~geo_ok) & (y_te == k)]
        m = f"lato mediano {np.median(lato_ing):.0f} px" if n else ""
        print(f"    PAI {g}: {n:3d} su {tot:3d} ({n/max(tot,1):.0%})   {m}")
    del c0

    # --- i due encoder sugli stessi casi ---
    fuori = {"soglie": [float(s) for s in soglie],
             "macro_f1_geometria_test": f1_test_geo,
             "n_tipici": int(geo_ok.sum()), "n_ingannevoli": int((~geo_ok).sum()),
             "encoder": {}}

    print(f"\n{'=' * 72}\nACCURATEZZA SUI DUE GRUPPI, media di "
          f"{len(a.seeds)} seed\n{'=' * 72}")
    print(f"{'encoder':16s} {'tipici':>10s} {'ingannevoli':>13s} {'divario':>9s}")
    print("-" * 72)
    for tag in a.tag:
        cached = load_latents(a.variant, layers=a.layers, tag=tag)
        acc_t, acc_i, f1_i = [], [], []
        for s in a.seeds:
            t0 = time.perf_counter()
            clf, _ = train_head(cached, a.metodo, a.head, seed=s)
            p = predizioni(clf, cached["data"]["test"], a.head)
            del clf
            torch.cuda.empty_cache()
            freno.pausa(t0)
            acc_t.append(float((p[geo_ok] == y_te[geo_ok]).mean()))
            acc_i.append(float((p[~geo_ok] == y_te[~geo_ok]).mean()))
            f1_i.append(macro_f1(confusion_matrix(y_te[~geo_ok], p[~geo_ok])))
        fuori["encoder"][tag] = {
            "acc_tipici": [float(np.mean(acc_t)), float(np.std(acc_t))],
            "acc_ingannevoli": [float(np.mean(acc_i)), float(np.std(acc_i))],
            "macro_f1_ingannevoli": [float(np.mean(f1_i)), float(np.std(f1_i))]}
        e = fuori["encoder"][tag]
        print(f"{tag:16s} {e['acc_tipici'][0]:6.4f}+-{e['acc_tipici'][1]:.3f} "
              f"{e['acc_ingannevoli'][0]:9.4f}+-{e['acc_ingannevoli'][1]:.3f} "
              f"{e['acc_tipici'][0] - e['acc_ingannevoli'][0]:9.4f}", flush=True)
        del cached

    if len(a.tag) == 2:
        import math
        x, y = (fuori["encoder"][t] for t in a.tag)
        print(f"\n{'=' * 72}\nLA PREVISIONE\n{'=' * 72}")
        for k, nome in (("acc_tipici", "sui TIPICI"),
                        ("acc_ingannevoli", "sugli INGANNEVOLI")):
            d = y[k][0] - x[k][0]
            se = math.sqrt(x[k][1] ** 2 + y[k][1] ** 2) / math.sqrt(len(a.seeds))
            z = d / se if se > 0 else float("nan")
            print(f"  {nome:20s} {a.tag[1]} - {a.tag[0]} = {d:+.4f}   z = {z:+.2f}")
        print("\n  attesa: pareggio sui tipici, vantaggio dell'addestrato")
        print("  sugli ingannevoli. Se non si vede, la complementarita' e'")
        print("  piu' debole di quanto la sonda k-NN suggerisca.")

    percorso = os.path.join(OUT_DIR, "stratificata_vit_small.json")
    save_json(fuori, percorso)
    print(f"\nRisultati in {percorso}")
