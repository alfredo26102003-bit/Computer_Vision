"""
PIPELINE — stadio 2 - estrazione latenti e addestramento della testa.

Train (stadio 2) - caching dei latenti + testa di classificazione PAI.

Sezione "Train" della struttura richiesta dal corso.

IL PUNTO CHIAVE DI QUESTO FILE: si estraggono i latenti dall'encoder
congelato UNA VOLTA e si salvano su disco. Da quel momento ogni esperimento
sullo sbilanciamento - novita', cinque baseline, sweep, cinque seed -
gira in secondi, anche su CPU. L'ablation dell'obiettivo 4 diventa
praticamente gratuito, ed e' la ragione per cui la seconda meta' del progetto
e' molto piu' tranquilla della prima.

Uso:
    python train_downstream.py --cache                  # una volta sola
    python train_downstream.py --method balanced_tokens --head ordinal
    python train_downstream.py --grid                   # tutti i metodi x teste x seed
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import LesionCropDataset, load_splits, make_loader, parse_annotations
from globals import (
    CACHE_DIR, DEFAULT_VARIANT, DEVICE, HEAD_BATCH_SIZE, HEAD_EPOCHS, HEAD_LR,
    HEAD_TYPES, HEAD_WEIGHT_DECAY, IMBALANCE_METHODS, N_SEEDS, NUM_CLASSES,
    OUT_DIR, SEED,
)
from imbalance import (
    balanced_sampler_weights, balanced_token_sampling, class_counts,
    compute_loss, n_views_per_class, n_views_uniform, random_token_sampling,
)
from network import (
    FrozenImageNetEncoder, LesionClassifier, bbox_to_token_mask, build_ijepa,
)
from utils import load_checkpoint, save_json, set_seed


# ==========================================================================
# 1. Caching dei latenti - si fa una volta
# ==========================================================================
@torch.no_grad()
def cache_latents(variant=DEFAULT_VARIANT, batch_size=64, layers=None,
                  ckpt_tag="", casuale=False, tag="", imagenet=False,
                  context_factor=None):
    """
    Estrae e salva i token dell'encoder congelato per tutte le lesioni.

    L'encoder e' quello pre-addestrato da train_ssl.py e resta CONGELATO:
    l'obiettivo 2 del brief chiede esplicitamente di valutare le
    rappresentazioni "frozen", quindi qui non si aggiorna nessun peso del
    backbone - si estraggono e basta.
    """
    # `layers` concatena piu' profondita' del ViT invece del solo ultimo
    # blocco. Misurato: le feature dell'ultimo blocco sono le piu'
    # COMPRESSE, e con una sonda lineare - che e' cio' che fa la testa - un
    # blocco intermedio rende molto di piu'. Il protocollo va tenuto IDENTICO
    # su tutte le configurazioni dell'ablation, altrimenti si confrontano i
    # protocolli di estrazione invece dei metodi.
    records = parse_annotations(verbose=False)
    splits = load_splits()

    # ckpt_tag sceglie QUALE run di pre-training usare: gli esperimenti
    # producono checkpoint distinti e il downstream deve puntare a quello
    # voluto, non al primo che c'e'.
    if imagenet:
        # CONTROLLO ESTERNO: ViT-B/16 pre-addestrato su ImageNet, congelato.
        # Stessa griglia 14x14, stesso ritaglio, stesso `layers`: cambia
        # l'encoder e nient'altro. E' l'unico modo di distinguere "il
        # dominio non ha niente da imparare" da "la nostra implementazione
        # non impara".
        model = FrozenImageNetEncoder().to(DEVICE)
        print("Encoder ImageNet ViT-B/16 congelato (torchvision, IMAGENET1K_V1)")
    elif casuale:
        # ENCODER CASUALE: stessa architettura, pesi non addestrati.
        # Non e' un "braccio di confronto" opzionale, e' il RIFERIMENTO
        # senza cui i numeri del pre-training non significano niente: dice
        # quanta della prestazione viene dall'addestramento e quanta e'
        # gia' data dall'architettura piu' le bbox. Il seme e' fisso, cosi'
        # il riferimento e' riproducibile e non cambia a ogni misura.
        set_seed(SEED)
        model = build_ijepa(variant).to(DEVICE)
        print(f"Encoder CASUALE (seme {SEED}), nessun peso addestrato")
    else:
        nome = f"ijepa_{variant}{('_' + ckpt_tag) if ckpt_tag else ''}"
        ckpt = load_checkpoint(nome, map_location=DEVICE)
        if ckpt is None:
            raise FileNotFoundError(f"Nessun checkpoint {nome}. Lanciate train_ssl.py")
        model = build_ijepa(ckpt.get("variant", variant)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"Encoder da {nome}, epoca {ckpt['epoch']}")

    model.eval()
    out = {}

    for split, ids in splits.items():
        ds = LesionCropDataset(records, ids, context_factor=context_factor)
        loader = make_loader(ds, batch_size=batch_size)
        toks, masks, labels, geoms = [], [], [], []

        for batch in loader:
            t = model.encode(batch["image"].to(DEVICE), return_layers=layers)
            m = bbox_to_token_mask(batch["bbox"].to(DEVICE), model.grid)
            toks.append(t.half().cpu())
            masks.append(m.cpu())
            labels.append(batch["label"])
            geoms.append(batch["geom"])

        out[split] = {
            "tokens": torch.cat(toks),
            "mask": torch.cat(masks),
            "labels": torch.cat(labels),
            "geom": torch.cat(geoms),
        }
        c = class_counts(out[split]["labels"])
        print(f"  {split:6s}: {len(ds):5d} lesioni  token={tuple(out[split]['tokens'].shape)}  "
              f"PAI3/4/5 = {c.int().tolist()}")

    dim = out["train"]["tokens"].shape[-1]
    suffisso = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(CACHE_DIR, f"latents_{variant}{suffisso}{tag}.pt")
    torch.save({"data": out, "embed_dim": dim, "grid": model.grid,
                "layers": layers}, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\nLatenti salvati: {path} ({size_mb:.0f} MB)")
    print("Da qui in poi ogni esperimento sullo sbilanciamento gira in secondi.")
    return path


def percorso_latenti(variant=DEFAULT_VARIANT, layers=None, tag=""):
    suffisso = "" if layers is None else "_L" + "-".join(map(str, layers))
    return os.path.join(CACHE_DIR, f"latents_{variant}{suffisso}{tag}.pt")


def impronta_latenti(variant=DEFAULT_VARIANT, layers=None, tag=""):
    """
    Identifica il file di latenti che ha prodotto un risultato.

    PERCHE' SERVE. `run_grid` riprende da file saltando le celle
    (metodo, testa) gia' presenti, e in precedenza non verificava NIENTE su
    quale encoder le avesse prodotte. La riga `none/flat`
    della griglia del 24 dava 0.8758 e non si e' riprodotta: rimisurata tre
    volte da' 0.8676, mentre `class_weighted` combacia alla quarta cifra.
    Una riga conservata da uno stato precedente non era escludibile perche'
    il file non portava alcuna provenienza.

    Dimensione e data di modifica bastano: i latenti si rigenerano solo con
    --cache, che riscrive il file per intero. Non e' un hash crittografico,
    e' un'impronta contro la ripartenza distratta - che e' il problema
    reale che si e' presentato.
    """
    path = percorso_latenti(variant, layers, tag)
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    return {"file": os.path.basename(path), "byte": st.st_size,
            "modificato": int(st.st_mtime)}


def load_latents(variant=DEFAULT_VARIANT, layers=None, tag=""):
    path = percorso_latenti(variant, layers, tag)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} mancante. Lanciate --cache")
    return torch.load(path, map_location="cpu", weights_only=False)


# ==========================================================================
# 2. Training della testa sui latenti cachati
# ==========================================================================
def istanze_per_epoca(train_labels, method, bts_alpha=0.5):
    """
    Quante istanze vede la testa in un'epoca con questo metodo.

    Non e' la stessa cosa per tutti, ed e' un confronto viziato finche' non
    lo si dice. `none`, `class_weighted`, `focal` e `oversample` fanno un
    passo per lesione: 4719. I metodi a viste ne fanno uno per VISTA: 6894
    ad alpha 0.5, cioe' il 46% di passi di gradiente in piu' a parita' di
    epoche. Parte del vantaggio della novita' potrebbe essere solo questo, e
    finche' non si pareggia il budget non si puo' escludere.
    """
    n = len(train_labels)
    if method in ("balanced_tokens", "random_tokens"):
        counts = class_counts(train_labels)
        v = (n_views_per_class(counts, bts_alpha) if method == "balanced_tokens"
             else n_views_uniform(counts, bts_alpha))
        return float((v.float() * counts).sum())
    return float(n)


def train_head(cached, method="none", head_type="flat", seed=0,
               epochs=HEAD_EPOCHS, verbose=False, bts_alpha=0.5, use_geom=False,
               pool_type="attn", budget_istanze=None, traccia_ogni=None):
    """
    Addestra attention pooling + testa sui latenti congelati.

    Gira in secondi: e' il motivo per cui potete permettervi N_SEEDS seed e
    intervalli di confidenza. Con sbilanciamento 7:1 i margini tra i metodi
    sono stretti e un singolo run non distingue niente.

    `budget_istanze` fissa il numero TOTALE di istanze di training invece
    del numero di epoche, e le epoche si ricavano di conseguenza. Serve al
    controllo a pari esempi visti: senza, i metodi a viste ricevono piu'
    passi di gradiente delle baseline e il confronto misura anche quello.

    `traccia_ogni` registra la validation ogni N epoche in `best["traiettoria"]`.
    E' SOLO registrazione: la selezione dell'epoca migliore continua a
    guardare la griglia ogni 10 epoche di sempre. Tenerle separate e'
    necessario - se la selezione vedesse piu' candidati sceglierebbe il
    massimo di piu' estrazioni, e la distorsione verso l'alto cambierebbe
    il numero misurato invece di limitarsi a descriverlo.
    """
    set_seed(seed)
    data, dim, grid = cached["data"], cached["embed_dim"], cached["grid"]

    tr, va = data["train"], data["val"]
    train_labels = tr["labels"]

    if budget_istanze is not None:
        per_ep = istanze_per_epoca(train_labels, method, bts_alpha)
        epochs = max(1, int(round(budget_istanze / per_ep)))

    gdim = tr["geom"].shape[1] if (use_geom and "geom" in tr) else 0
    clf = LesionClassifier(dim, grid, head_type, geom_dim=gdim,
                           pool_type=pool_type).to(DEVICE)
    opt = torch.optim.AdamW(clf.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)

    n = len(train_labels)
    if method == "oversample":
        w = balanced_sampler_weights(train_labels).double()
        sampler = torch.utils.data.WeightedRandomSampler(w, n, replacement=True)
        order_fn = lambda: torch.tensor(list(sampler))
    else:
        order_fn = lambda: torch.randperm(n)

    # SMOTE latente: si sintetizza UNA VOLTA prima del ciclo, e solo dal
    # train split. Interpolare campioni di validation nel train falsifica
    # tutto, ed e' un errore che non si vede nelle metriche.
    tr_tokens, tr_mask, tr_labels_ep = tr["tokens"], tr["mask"], train_labels
    tr_geom = tr.get("geom")

    # I token dello split stanno sulla GPU UNA VOLTA SOLA, non batch per
    # batch. La griglia dell'obiettivo 4 sono 6 metodi x 2 teste x N_SEEDS =
    # decine di addestramenti da migliaia di step ciascuno, e ricopiare ogni
    # batch domina il tempo totale. Con ~1.4 GB per lo split piu' grosso ci
    # sta; se non ci sta si continua dalla CPU senza cambiare i risultati.
    try:
        tr_tokens = tr_tokens.to(DEVICE)
        tr_mask = tr_mask.to(DEVICE)
        tr_labels_ep = tr_labels_ep.to(DEVICE)
        if gdim:
            tr_geom = tr_geom.to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # La novita' invece si applica PER BATCH: le viste devono cambiare a ogni
    # epoca, altrimenti sono duplicati e non aggiungono informazione.
    bts_counts = class_counts(train_labels).to(tr_labels_ep.device)
    bts_gen = torch.Generator(device=tr_tokens.device).manual_seed(seed)

    best = {"val_f1": -1.0}
    traiettoria = []
    for epoch in range(epochs):
        clf.train()
        order = order_fn().to(tr_tokens.device)
        for i in range(0, n, HEAD_BATCH_SIZE):
            idx = order[i:i + HEAD_BATCH_SIZE]
            tok = tr_tokens[idx].float()
            msk = tr_mask[idx]
            y = tr_labels_ep[idx]
            gm = tr_geom[idx] if gdim else None

            if method in ("balanced_tokens", "random_tokens"):
                campiona = (balanced_token_sampling if method == "balanced_tokens"
                            else random_token_sampling)
                tok, msk, y, orig = campiona(
                    tok, msk, y, bts_counts, generator=bts_gen, alpha=bts_alpha
                )
                # La geometria segue le viste tramite l'indice di origine:
                # ogni vista e' la STESSA lesione, quindi eredita la sua bbox.
                if gdim:
                    gm = gm[orig]

            tok, msk, y = tok.to(DEVICE), msk.to(DEVICE), y.to(DEVICE)
            if gdim:
                gm = gm.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits, _, _ = clf(tok, token_mask=msk, geom=gm)
            loss = compute_loss(logits, y, method, head_type, train_labels)
            loss.backward()
            opt.step()

        if traccia_ogni and ((epoch + 1) % traccia_ogni == 0 or epoch == 0):
            from evaluation import evaluate_split
            t = evaluate_split(clf, va, head_type, use_geom=bool(gdim))
            traiettoria.append({"epoca": epoch,
                                "val_macro_f1": t["macro_f1"],
                                "val_pr_auc_pai5": t["pr_auc_pai5"]})

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            from evaluation import evaluate_split
            m = evaluate_split(clf, va, head_type, use_geom=bool(gdim))
            if m["macro_f1"] > best["val_f1"]:
                best = {"val_f1": m["macro_f1"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in clf.state_dict().items()}}
            if verbose:
                print(f"    ep{epoch:03d} loss={loss.item():.4f} "
                      f"val_macroF1={m['macro_f1']:.4f}")

    if "state" in best:
        clf.load_state_dict(best["state"])
    if traccia_ogni:
        best["traiettoria"] = traiettoria
    return clf, best


def run_grid(variant=DEFAULT_VARIANT, methods=None, heads=None,
             seeds=None, layers=None, tag=""):
    """
    Griglia completa: metodo x tipo di testa x seed.

    Copre l'obiettivo 4 (ablation) e produce gli intervalli di confidenza
    senza cui, con 7:1 di sbilanciamento, i confronti non significano nulla.
    """
    from evaluation import evaluate_split

    cached = load_latents(variant, layers=layers, tag=tag)
    methods = methods or IMBALANCE_METHODS
    heads = heads or HEAD_TYPES
    seeds = seeds or list(range(N_SEEDS))

    # RIPARTENZA. La griglia dura ~40 minuti e questa macchina si e' spenta
    # da sola sette volte in cinque giorni. Senza ripartenza un crash a
    # meta' butta via tutto: e' gia' successo, 30 minuti persi perche' i
    # risultati stavano solo in memoria.
    suff = "" if layers is None else "_L" + "-".join(map(str, layers))
    percorso = os.path.join(OUT_DIR, f"results_{variant}{suff}{tag}.json")
    impronta = impronta_latenti(variant, layers, tag)

    rows = []
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            vecchio = json.load(f)
        righe_vecchie = vecchio.get("righe", vecchio) if isinstance(vecchio, dict) else vecchio
        imp_vecchia = vecchio.get("latenti") if isinstance(vecchio, dict) else None
        if imp_vecchia == impronta and righe_vecchie:
            rows = righe_vecchie
            print(f"Riprendo: {len(rows)} righe gia' su disco in "
                  f"{os.path.basename(percorso)}")
        elif righe_vecchie:
            # RIFIUTO DI RIPRENDERE. Un file senza impronta viene da prima
            # che questo controllo esistesse: non si sa con quali latenti
            # sia stato prodotto, e mescolare due encoder nella stessa
            # tabella e' peggio che rifare due ore di calcolo.
            motivo = ("non porta l'impronta dei latenti"
                      if imp_vecchia is None else
                      f"e' stato prodotto con {imp_vecchia.get('file')} "
                      f"({imp_vecchia.get('byte')} byte)")
            print(f"NON riprendo da {os.path.basename(percorso)}: {motivo}, "
                  f"mentre ora userei {impronta['file'] if impronta else '?'}.")
            print("Rifaccio tutte le righe da zero.")
    fatte = {(r["method"], r["head"]) for r in rows}

    for method, head in itertools.product(methods, heads):
        if (method, head) in fatte:
            print(f"  {method:16s} {head:8s} gia' fatta, salto")
            continue
        per_seed = []
        for s in seeds:
            try:
                clf, _ = train_head(cached, method, head, seed=s)
                per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
            except NotImplementedError as exc:
                print(f"  salto {method}/{head}: {exc}")
                per_seed = []
                break
        if not per_seed:
            continue

        agg = {"method": method, "head": head, "n_seeds": len(per_seed)}
        # Le F1 per classe servono per il criterio operativo: si accetta un
        # metodo se alza PAI 5 SENZA erodere PAI 3 e 4. Con la sola macro-F1
        # un guadagno sulla minoritaria pagato dalle altre due sembrerebbe
        # un miglioramento.
        for k in ("macro_f1", "balanced_acc", "recall_pai5", "pr_auc_pai5",
                  "quadratic_kappa", "f1_pai3", "f1_pai4", "f1_pai5",
                  "precision_pai5"):
            vals = [m[k] for m in per_seed]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        rows.append(agg)
        # Salvataggio incrementale: il costo di un crash scende da "tutta la
        # griglia" a "la riga in corso".
        save_json({"latenti": impronta, "quando": time.strftime("%Y-%m-%d %H:%M"),
                   "seeds": list(seeds), "righe": rows}, percorso)
        print(f"  {method:16s} {head:8s} macroF1={agg['macro_f1_mean']:.4f}"
              f"+-{agg['macro_f1_std']:.4f}  F1(3/4/5)="
              f"{agg['f1_pai3_mean']:.3f}/{agg['f1_pai4_mean']:.3f}/{agg['f1_pai5_mean']:.3f}"
              f"  recall5={agg['recall_pai5_mean']:.4f}"
              f"  prec5={agg['precision_pai5_mean']:.3f}"
              f"  kappa={agg['quadratic_kappa_mean']:.4f}")

    if rows:
        path = percorso
        save_json(rows, path)
        print(f"\nRisultati in {path}")
    return rows


def sweep_alpha(variant=DEFAULT_VARIANT, alphas=None, heads=None,
                seeds=None, layers=None, tag=""):
    """
    Ablation su alpha della novita' - richiesto dall'obiettivo 4.

    alpha regola quante viste riceve ogni classe: n_c = ceil((max/n_c)^alpha).
    Con lo sbilanciamento reale (3017/1229/473 nel train) alpha=0.5 da'
    [1,2,3] viste, cioe' PAI 5 resta sotto-rappresentata di circa un fattore
    2; alpha=1.0 da' [1,2,6], cioe' il pareggio effettivo. E' la ragione per
    cui la novita' perdeva contro `oversample`, che invece pareggia davvero.

    Il confronto interessante e' proprio con `oversample`: stesso numero di
    istanze per classe, ma li' sono duplicati identici, qui sono
    sottoinsiemi di token diversi della stessa lesione.
    """
    from evaluation import evaluate_split

    cached = load_latents(variant, layers=layers, tag=tag)
    alphas = alphas or [0.0, 0.25, 0.5, 0.75, 1.0]
    heads = heads or HEAD_TYPES
    seeds = seeds or list(range(N_SEEDS))

    counts = class_counts(cached["data"]["train"]["labels"])
    print(f"conteggi train PAI3/4/5: {counts.int().tolist()}")
    from imbalance import n_views_per_class
    for al in alphas:
        print(f"  alpha={al:.2f} -> viste per classe {n_views_per_class(counts, al).tolist()}")

    rows = []
    for al, head in itertools.product(alphas, heads):
        per_seed = []
        for s in seeds:
            clf, _ = train_head(cached, "balanced_tokens", head, seed=s, bts_alpha=al)
            per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        agg = {"method": f"balanced_tokens", "alpha": al, "head": head,
               "n_seeds": len(per_seed)}
        for k in ("macro_f1", "balanced_acc", "recall_pai5", "pr_auc_pai5",
                  "quadratic_kappa", "f1_pai3", "f1_pai4", "f1_pai5",
                  "precision_pai5"):
            vals = [m[k] for m in per_seed]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        rows.append(agg)
        print(f"  alpha={al:.2f} {head:8s} macroF1={agg['macro_f1_mean']:.4f}"
              f"+-{agg['macro_f1_std']:.4f}  F1(3/4/5)="
              f"{agg['f1_pai3_mean']:.3f}/{agg['f1_pai4_mean']:.3f}/{agg['f1_pai5_mean']:.3f}"
              f"  recall5={agg['recall_pai5_mean']:.4f}+-{agg['recall_pai5_std']:.4f}"
              f"  prec5={agg['precision_pai5_mean']:.3f}")

    suff = "" if layers is None else "_L" + "-".join(map(str, layers))
    path = os.path.join(OUT_DIR, f"sweep_alpha_{variant}{suff}{tag}.json")
    save_json(rows, path)
    print(f"\nRisultati in {path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--sweep-alpha", action="store_true")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="blocchi da concatenare, es. --layers 2 7 11")
    ap.add_argument("--context-factor", type=float, default=None,
                    help="ABLATION cieca alla dimensione: finestra pari a "
                         "questo multiplo del lato della bbox, ridimensionata. "
                         "Con 3.0 ogni lesione appare uguale e il segnale "
                         "geometrico sparisce. Senza, finestra fissa 224 px")
    ap.add_argument("--imagenet", action="store_true",
                    help="controllo esterno: ViT-B/16 ImageNet congelato, "
                         "stesso ritaglio e stesso --layers dei nostri")
    ap.add_argument("--random", action="store_true",
                    help="encoder con pesi CASUALI: il riferimento senza cui "
                         "i numeri del pre-training non significano nulla")
    ap.add_argument("--tag", default="",
                    help="suffisso dei file, per tenere i bracci separati")
    ap.add_argument("--ckpt-tag", default="",
                    help="quale run SSL usare per il braccio ijepa (es. pred48)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--method", default="none", choices=IMBALANCE_METHODS)
    ap.add_argument("--head", default="flat", choices=HEAD_TYPES)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--metodi", nargs="+", default=None,
                    help="sottoinsieme di IMBALANCE_METHODS per la griglia. "
                         "Serve ai bracci di controllo, dove la domanda "
                         "riguarda l'ENCODER e i metodi di sbilanciamento "
                         "moltiplicherebbero il costo senza aggiungere niente")
    ap.add_argument("--teste", nargs="+", default=None,
                    help="sottoinsieme di HEAD_TYPES per la griglia")
    a = ap.parse_args()

    if a.cache:
        cache_latents(a.variant, layers=a.layers, ckpt_tag=a.ckpt_tag,
                      casuale=a.random, tag=a.tag, imagenet=a.imagenet,
                      context_factor=a.context_factor)
    elif a.sweep_alpha:
        sweep_alpha(a.variant, layers=a.layers, tag=a.tag)
    elif a.grid:
        run_grid(a.variant, layers=a.layers, tag=a.tag,
                 methods=a.metodi, heads=a.teste)
    else:
        from evaluation import evaluate_split, print_report
        cached = load_latents(a.variant, layers=a.layers)
        clf, best = train_head(cached, a.method, a.head, verbose=True)
        print_report(evaluate_split(clf, cached["data"]["test"], a.head),
                     f"{a.method} / {a.head}")
