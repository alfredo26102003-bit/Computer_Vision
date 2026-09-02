"""
PIPELINE — stadio 1 - pre-training I-JEPA.

Train (stadio 1) - pre-training I-JEPA sui tile, con monitoraggio del collasso.

Sezione "Train" della struttura richiesta dal corso.

Uso:
    python train_ssl.py --variant vit_tiny --epochs 300
    python train_ssl.py --resume
    python train_ssl.py --smoke          # verifica che tutto girii, 20 step
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score

from data import (
    CropCacheDataset, LesionCropDataset, TileDataset, cache_crop, load_splits,
    make_loader, parse_annotations,
)
from globals import (
    AMP, CKPT_DIR, GRAD_CLIP, MONITOR_SAMPLES, NUM_CLASSES, amp_dtype, DEFAULT_VARIANT, DEVICE, FIG_DIR, KNN_PROBE_EVERY, KNN_SUBSET,
    DOWNSTREAM_SONDA_TRAIN, GATE_CROLLO, GATE_EPOCH, GATE_MARGINE,
    GATE_SONDE_SOTTO, LAYERS_DOWNSTREAM, OUT_DIR, RESOCONTO_OGNI, SSL_BATCH_SIZE, SSL_EMA_END, SSL_EMA_START,
    SSL_EPOCHS, SSL_LR, SSL_WARMUP_EPOCHS, SSL_WEIGHT_DECAY, TILE_SIZE,
)
from evaluation import confusion_matrix, macro_f1
import network
from network import bbox_to_token_mask, build_ijepa, count_params

# Sovrascrivibili da riga di comando: sono i parametri che si esplorano per
# uscire dal collasso, e vanno cambiati senza toccare globals.py, che
# descrive la configurazione di riferimento.
EMA_START = SSL_EMA_START
EMA_END = SSL_EMA_END
LR = SSL_LR
GATE_AT = GATE_EPOCH
from utils import (
    Freno, Termostato, effective_rank_centered,
    AverageMeter, CollapseMonitor, knn_probe, load_checkpoint, save_checkpoint,
    set_seed,
)


def ema_momentum(step, total_steps):
    """Momentum EMA con schedule da EMA_START a EMA_END (coseno).

    EMA_START e' la leva principale contro il collasso: piu' e' alto, piu'
    il target encoder e' lento, e piu' e' difficile per il context encoder
    inseguirlo fino alla soluzione costante.
    """
    p = min(step / max(total_steps, 1), 1.0)
    return EMA_END - (EMA_END - EMA_START) * (math.cos(math.pi * p) + 1) / 2


@torch.no_grad()
def extract_for_probe(model, loader, max_items=KNN_SUBSET):
    """
    Feature aggregate sui token DENTRO LA BBOX, piu' le etichette.

    PERCHE' LA MASCHERA. La versione precedente mediava tutti i 196 token
    dell'immagine. Funzionava per caso: col vecchio crop 'relative' la
    lesione occupava sempre un terzo esatto del riquadro, quindi la media
    portava comunque segnale. Col crop a finestra fissa la lesione e' 8-20
    token su 196 e la media e' dominata dallo sfondo: rimisurando i
    checkpoint esistenti, tutti crollavano a ridosso del pavimento
    (0.27-0.28 contro 0.2530) mentre gli stessi encoder rendono 0.7415 a
    valle, dove l'attention pooling usa la bbox.

    Non era l'encoder a essere peggiorato: era la sonda a misurare
    prevalentemente osso sano. Qui si aggrega come fa il downstream, cosi'
    la sonda torna a essere un anticipo di quel numero e non di altro.
    """
    model.eval()
    feats, labels = [], []
    n = 0
    for batch in loader:
        tokens = model.encode(batch["image"].to(DEVICE))
        msk = bbox_to_token_mask(batch["bbox"].to(DEVICE), model.grid)
        w = msk.float().unsqueeze(-1)
        pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1)
        feats.append(pooled.float().cpu())
        labels.append(batch["label"])
        n += tokens.shape[0]
        if n >= max_items:
            break
    model.train()
    return torch.cat(feats), torch.cat(labels)


def probe_lineare(ftr, ltr, fva, lva):
    """
    Sonda LINEARE: una direzione APPRESA nello spazio delle feature.

    E' la sonda che conta, e sostituisce il k-NN come criterio di giudizio.

    PERCHE' IL k-NN ERA LA MISURA SBAGLIATA. Il k-NN classifica per
    DISTANZA euclidea, quindi premia una geometria precisa: che i vicini
    piu' prossimi abbiano lo stesso grado. Un addestramento puo' conservare
    tutta l'informazione e cambiare la geometria - riallocando la varianza
    su direzioni che servono al compito di pre-training - e il k-NN crolla
    lo stesso.

    E' esattamente cio' che succede qui. Misurato Sul checkpoint
    dell'epoca 40, contro l'encoder casuale:

        informazione presente (R^2 di una regressione lineare)
          intensita' media nella lesione   0.991 -> 0.992
          log area della bbox              0.886 -> 0.884
        lettura della stessa rappresentazione
          k-NN     (distanza)              0.7299 -> 0.5422
          lineare  (appresa)               0.7212 -> 0.7329

    L'informazione e' intatta. Solo la distanza euclidea smette di
    rifletterla. E il downstream di questo progetto NON usa distanze: usa
    attention pooling piu' una testa addestrata, cioe' una lettura appresa.
    Il k-NN misurava quindi una proprieta' che al progetto non serve, e ha
    fatto interrompere run che stavano migliorando.
    """
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-8
    ztr, zva = ((ftr - mu) / sd).numpy(), ((fva - mu) / sd).numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(ztr, ltr.numpy())
    return macro_f1(confusion_matrix(lva.numpy(), clf.predict(zva)))


_CROP = {}


def _dati_sonda(records, splits):
    """
    Crop di train e val, ritagliati UNA volta e tenuti in memoria.

    Prima ogni sonda ricostruiva i crop da zero: apriva ~5700 panoramiche
    intere da disco e le ridimensionava, per un costo di oltre due minuti a
    sonda, quasi tutti spesi in decodifica JPEG e non nella rete. Con la
    cache la stessa operazione costa 0.02 s, quindi sondare spesso smette di
    essere un lusso: e' proprio la mancanza di misure frequenti che ha
    lasciato correre run sbagliati per ore.
    """
    if not _CROP:
        for split in ("train", "val"):
            d = cache_crop(records, splits[split], split)
            _CROP[split] = make_loader(CropCacheDataset(d), shuffle=False,
                                       batch_size=64, num_workers=0)
            # Grandezze misurate sull'IMMAGINE, indipendenti dalla rete:
            # servono a chiedersi non "quanto e' brava" ma "quale
            # informazione conserva".
            img, bb = d["image"].float() / 255.0, d["bbox"]
            inten, dim = [], []
            for k in range(img.shape[0]):
                x0, y0, x1, y1 = [int(v) for v in bb[k]]
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
                inten.append(float(img[k, y0:y1, x0:x1].mean()))
                dim.append(float(np.log((x1 - x0) * (y1 - y0))))
            _CROP[split + "_aux"] = np.array([inten, dim]).T
    return _CROP


def _r2(F, y):
    """R^2 di una regressione lineare (ridge) dall'embedding a `y`, in
    5-fold. Dice se l'informazione e' PRESENTE e leggibile linearmente."""
    Z = (F - F.mean(0)) / (F.std(0) + 1e-8)
    return float(cross_val_score(RidgeCV(alphas=np.logspace(-2, 4, 13)),
                                 Z.numpy(), y, cv=5, scoring="r2").mean())


@torch.no_grad()
def _token_multilayer(model, loader, cap):
    """Token concatenati sulle profondita' del downstream, piu' maschera
    bbox, etichette e geometria: lo stesso formato che cache_latents scrive
    su disco, ma tenuto in memoria e su un sottoinsieme."""
    T, M, Y, G = [], [], [], []
    n = 0
    for b in loader:
        t = model.encode(b["image"].to(DEVICE), return_layers=LAYERS_DOWNSTREAM)
        T.append(t.half().cpu())
        M.append(bbox_to_token_mask(b["bbox"].to(DEVICE), model.grid).cpu())
        Y.append(b["label"] if torch.is_tensor(b["label"])
                 else torch.tensor(b["label"]))
        G.append(b["geom"])
        n += t.shape[0]
        if n >= cap:
            break
    return {"tokens": torch.cat(T), "mask": torch.cat(M),
            "labels": torch.cat(Y), "geom": torch.cat(G)}


def sonda_downstream(model, records, splits, seed=0):
    """
    Il modello vero dell'obiettivo 2, ma letto DOVE si vede la
    rappresentazione: a conteggio di token fisso.

    ESISTE PERCHE' TUTTI I SURROGATI HANNO MENTITO. In quest'ordine:
      rango effettivo  segnava collasso anche sull'encoder casuale, che e'
                       il miglior estrattore che abbiamo
      k-NN             misura la geometria: e' crollato di 0.17 mentre
                       l'informazione restava intatta (R^2 0.99 / 0.88)
      sonda lineare    guarda solo l'ultimo blocco e usa media invece di
                       attention pooling. Sui due punti dove conosciamo
                       entrambi i numeri ha il SEGNO INVERTITO rispetto al
                       downstream: +0.0043 contro -0.0074.

    E POI HA MENTITO ANCHE IL NON-SURROGATO. Questa funzione leggeva la
    pipeline del brief tale e quale - maschera della bbox - e sembrava
    l'unica misura al riparo, perche' era la stessa lettura che finisce in
    presentazione. Abbiamo mostrato che quella lettura e'
    dominata dal CANALE DELLA MASCHERA: la bbox seleziona 16 / 36 / 64
    token secondo la classe, e la sola maschera one-hot, senza un solo
    pixel, da' macro-F1 0.7708 - piu' del vettore intero dell'encoder
    casuale. Il criterio guardava la bounding box, non l'encoder.

    La prova sta nel resoconto del run `completa`: fra l'epoca 154 e la 229
    il downstream cosi' misurato oscilla fra 0.7415 e 0.7639 SENZA TENDENZA,
    mentre nello stesso intervallo la rappresentazione stava cambiando. Era
    rumore attorno a un canale costante.

    COSA CAMBIA ORA. Si seleziona sul protocollo P3_K16 - i 16 token piu'
    vicini al centro della bbox, uguali per ogni classe. La bbox resta usata
    per LOCALIZZARE, non per contare. Cambia solo la maschera: stessi token,
    stessa testa, stessi seed.

    SI MISURA ANCHE P1_bbox, e non serve al criterio: serve come CONTROLLO.
    Se durante il pre-training K16 sale e P1 resta piatta, la diagnosi e'
    dimostrata sulla curva stessa invece che a posteriori. Costa un secondo
    train_head ogni RESOCONTO_OGNI epoche - i token, che sono la parte cara,
    si estraggono una volta sola.

    Si valuta su validation e MAI su test: scegliere il checkpoint guardando
    il test sarebbe barare.

    Ritorna (macroF1_K16, PR-AUC5_K16, macroF1_P1): il criterio e' il primo.
    """
    from train_downstream import train_head
    from evaluation import evaluate_split
    from exp_fixedk import maschera, GRID

    # maschera() ragiona su una griglia GRID x GRID fissa. Se il modello ne
    # avesse un'altra, gli indici dei token piu' vicini al centro sarebbero
    # sbagliati in silenzio - e la sonda mentirebbe di nuovo.
    assert model.grid == GRID, f"griglia {model.grid} != {GRID} di exp_fixedk"

    d = _dati_sonda(records, splits)
    cached = {"data": {"train": _token_multilayer(model, d["train"],
                                                  DOWNSTREAM_SONDA_TRAIN),
                       "val": _token_multilayer(model, d["val"], 10 ** 9)},
              "grid": model.grid}
    cached["embed_dim"] = cached["data"]["train"]["tokens"].shape[-1]

    out = {}
    for prot in ("P3_K16", "P1_bbox"):
        # Copia superficiale con la sola maschera sostituita: i token non si
        # toccano. E' esattamente cio' che fa exp_fixedk.con_maschera.
        c = {**cached, "data": {s: {**v, "mask": maschera(prot, v)}
                                for s, v in cached["data"].items()}}
        clf, _ = train_head(c, "none", "flat", seed=seed)
        r = evaluate_split(clf, c["data"]["val"], "flat")
        out[prot] = (r["macro_f1"], r.get("pr_auc_pai5", float("nan")))
        del clf, c
        torch.cuda.empty_cache()

    del cached
    torch.cuda.empty_cache()
    return out["P3_K16"][0], out["P3_K16"][1], out["P1_bbox"][0]


def stampa_resoconto(storico, rif, run_name, epoca, totale, rif_down=None):
    """
    Tabella riassuntiva di tutte le sonde fatte finora.

    Un run da 300 epoche produce centinaia di righe di log, e la domanda
    utile e' sempre la stessa: sta migliorando rispetto all'encoder casuale?
    Qui la risposta si legge in una tabella sola, con i delta gia' fatti e
    la TENDENZA delle ultime tre sonde - perche' il valore singolo oscilla
    di ~0.01 e da solo non dice niente.

    Si scrive anche su file, cosi' si puo' guardare senza aprire il log.
    """
    righe = []
    righe.append("=" * 78)
    righe.append(f"RESOCONTO - {run_name} - epoca {epoca}/{totale}")
    righe.append(f"DA BATTERE - encoder casuale su P3_K16 = "
                 f"{rif_down if rif_down is not None else float('nan'):.4f}"
                 f"   (sonda lineare {rif:.4f})")
    righe.append("CRITERIO: P3_K16, 16 token per tutti. La colonna P1 ctrl e' di"
                 " CONTROLLO e non giudica:")
    righe.append("se K16 sale mentre P1 resta piatta, il protocollo del brief"
                 " sta leggendo la bbox e non l'encoder.")
    righe.append("=" * 78)
    righe.append(f"{'epoca':>6s} {'K16 *crit*':>11s} {'PR-AUC5':>8s} "
                 f"{'P1 ctrl':>8s} {'lineare':>9s} {'k-NN':>8s} {'rango':>7s} "
                 f"{'R2 int':>7s} {'R2 dim':>7s}")
    for e in storico:
        righe.append(f"{e['epoch']:6d} "
                     f"{e.get('downstream', float('nan')):11.4f} "
                     f"{e.get('pr_auc5', float('nan')):8.4f} "
                     f"{e.get('p1_bbox', float('nan')):8.4f} "
                     f"{e['lineare']:9.4f} {e['knn']:8.4f} "
                     f"{e.get('rango_c', float('nan')):7.2f} "
                     f"{e.get('r2_int', float('nan')):7.3f} "
                     f"{e.get('r2_dim', float('nan')):7.3f}")

    # La tendenza si legge sul DOWNSTREAM quando c'e', non sul surrogato.
    con_d = [e for e in storico if "downstream" in e]
    chiave, serie = ("downstream", con_d) if con_d else ("lineare", storico)
    if len(serie) >= 3:
        ultime = [e[chiave] for e in serie[-3:]]
        pend = (ultime[-1] - ultime[0]) / 2
        verso = ("in miglioramento" if pend > 0.002 else
                 "in peggioramento" if pend < -0.002 else "stabile")
        righe.append("")
        righe.append(f"Tendenza sulle ultime 3 sonde ({chiave}): {verso} "
                     f"({pend:+.4f} per sonda)")

    migliore = max(serie, key=lambda e: e[chiave])
    rifer = rif_down if chiave == "downstream" and rif_down is not None else rif
    righe.append(f"Migliore finora ({chiave}): {migliore[chiave]:.4f} "
                 f"all'epoca {migliore['epoch']}  "
                 f"({migliore[chiave] - rifer:+.4f} vs casuale)")
    righe.append("=" * 78)

    testo = chr(10).join(righe)
    print(testo, flush=True)
    with open(os.path.join(OUT_DIR, f"resoconto_{run_name}.txt"), "w",
              encoding="utf-8") as f:
        f.write(testo + chr(10))


def run_probe(model, records, splits, completo=False):
    """
    Pannello diagnostico. Ritorna un dict, non un numero solo.

    Il progetto ha perso giorni perche' guardava una misura sola per volta,
    e ogni volta era quella sbagliata: prima il rango effettivo (che segna
    collasso anche sull'encoder casuale), poi il k-NN (che misura la
    geometria, non l'informazione). Qui si guardano insieme:

      lineare  macro-F1 di una direzione APPRESA. E' il criterio: e' la
               stessa lettura che fa il downstream (attention pooling piu'
               testa addestrata).
      knn      macro-F1 per distanza euclidea. Diagnostica della GEOMETRIA:
               se scende mentre `lineare` tiene, il modello ha riorganizzato
               lo spazio senza perdere informazione.
      r2_int   quanto e' leggibile l'intensita' media della lesione
      r2_dim   quanto e' leggibile la dimensione della bbox
               Sono le due grandezze che definiscono il grado PAI. Se
               restano alte, nessuna informazione utile e' andata persa,
               qualunque cosa dicano le misure geometriche.
      rango    rango effettivo centrato: quante direzioni sono usate.

    `completo=False` salta le due R^2, che costano una cross-validation.
    """
    d = _dati_sonda(records, splits)
    ftr, ltr = extract_for_probe(model, d["train"], max_items=10 ** 9)
    fva, lva = extract_for_probe(model, d["val"], max_items=10 ** 9)

    lin = probe_lineare(ftr, ltr, fva, lva)
    _, knn = knn_probe(ftr, ltr, fva, lva)
    quota = torch.bincount(lva.long()).max().item() / len(lva)
    pavimento = (2 * quota / (1 + quota)) / NUM_CLASSES

    p = {"lineare": lin, "knn": knn, "rango_c": effective_rank_centered(fva),
         "pavimento": pavimento}
    if completo:
        aux = d["val_aux"]
        p["r2_int"], p["r2_dim"] = _r2(fva, aux[:, 0]), _r2(fva, aux[:, 1])

    extra = (f"  R2 intensita={p['r2_int']:.3f} dimensione={p['r2_dim']:.3f}"
             if completo else "")
    stato = "OK" if lin > pavimento * 1.10 else "<-- AL LIVELLO DEL CASO"
    print(f"  [sonda] lineare={lin:.4f}  k-NN={knn:.4f}  rango={p['rango_c']:.2f}"
          f"{extra}  (pavimento {pavimento:.4f}) {stato}", flush=True)
    return p

def train(variant=DEFAULT_VARIANT, epochs=SSL_EPOCHS, batch_size=SSL_BATCH_SIZE,
          resume=False, smoke=False, tag="", carico=100, soglia_gpu=0):
    set_seed()
    # Il tag tiene separati i checkpoint di run paralleli: senza, due varianti
    # lanciate insieme si sovrascrivono a vicenda lo stesso file.
    run_name = f"ijepa_{variant}{('_' + tag) if tag else ''}"

    records = parse_annotations(verbose=False)
    splits = load_splits()

    # Pre-training SOLO sui tile delle immagini di train, e SOLO dagli
    # originali: la cartella di augmentation del dataset e' ignorata (il SSL
    # fa le proprie augmentation e quelle pre-generate creano solo occasioni
    # di leakage).
    train_ds = TileDataset(records, splits["train"], augment=True)
    # Il dataset rende k crop per item da una sola decodifica, quindi il
    # batch va chiesto in ITEM e non in tile, altrimenti il batch effettivo
    # sarebbe k volte quello voluto (e andrebbe in OOM).
    k = train_ds.k
    loader = make_loader(train_ds, shuffle=True, batch_size=max(batch_size // k, 1))
    n_tile = len(train_ds) * k
    print(f"Tile di pre-training: {n_tile} da {len(splits['train'])} immagini "
          f"({len(train_ds)} item x {k} crop per decodifica)")
    print(f"Step per epoca: {len(loader)}  (batch = {max(batch_size // k, 1)} img x {k} = {max(batch_size // k, 1) * k} tile)")

    model = build_ijepa(variant).to(DEVICE)
    print(f"{variant}: {count_params(model)/1e6:.2f}M parametri addestrabili")
    print(f"[iperparametri] lr={LR:.2e} ema={EMA_START}->{EMA_END} "
          f"predictor_dim={network.PREDICTOR_DIM}")
    if network.MASK_RATIO is not None:
        _c, _b = network.sample_masks(TILE_SIZE // 16)
        print(f"[mascheramento] richiesto {network.MASK_RATIO}, blocchi "
              f"{network.BLOCK_SCALE} -> contesto {_c.numel()} patch, "
              f"{len(_b)} blocchi (paper: 90 patch, 4 blocchi)")

    # In I-JEPA si ottimizzano context encoder e predictor: il target
    # encoder segue per EMA (obiettivo 1 del brief) e non riceve gradienti,
    # quindi si prendono solo i parametri che li richiedono.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=SSL_WEIGHT_DECAY)

    total_steps = max(epochs * len(loader), 1)
    warmup = SSL_WARMUP_EPOCHS * len(loader)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    # Il GradScaler serve SOLO con float16: il bfloat16 ha il range dinamico
    # del float32 e non ha bisogno di scalare la loss. Abilitarlo comunque
    # non romperebbe nulla, ma maschera i problemi numerici veri.
    dtype = amp_dtype()
    use_amp = AMP and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    print(f"Precisione: {dtype} (GradScaler {'attivo' if scaler.is_enabled() else 'non necessario'})")

    monitor = CollapseMonitor()
    brake = Freno(carico)
    if carico < 100:
        print(f"[freno] {brake}")
    term = Termostato(soglia=soglia_gpu, riparti=soglia_gpu - 8) if soglia_gpu else None
    if term:
        print(f"[termostato] {term}")

    # TensorBoard: curve dal vivo. Mentre un run gira, in un altro terminale:
    #     .venv\Scripts\tensorboard --logdir runs\tb
    # poi apri http://localhost:6006. Scrive per epoca loss, rango effettivo,
    # rapporto di rango, std, k-NN, durata e temperatura GPU. Costa nulla, e se
    # tensorboard non e' installato il run prosegue senza.
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(OUT_DIR, "tb", run_name))
    except Exception as e:
        print(f"  [tensorboard non attivo: {e}]")
        tb = None

    start_epoch, gstep = 0, 0
    knn_ref = None
    rif_down = None
    if resume:
        ckpt = load_checkpoint(run_name, map_location=DEVICE)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch, gstep = ckpt["epoch"] + 1, ckpt["gstep"]
            monitor.history = ckpt.get("monitor", [])
            knn_ref = ckpt.get("knn_ref")
            rif_down = ckpt.get("rif_down")
            print(f"Ripreso dall'epoca {start_epoch}")

            # Se il checkpoint e' stato scritto con iperparametri diversi da
            # quelli attivi ora, si RIFIUTA di proseguire: continuare
            # silenziosamente produrrebbe un run ibrido, impossibile da
            # descrivere in presentazione.
            atteso = {"lr": LR, "ema_start": EMA_START,
                      "predictor_dim": network.PREDICTOR_DIM}
            diverso = {k: (ckpt[k], v) for k, v in atteso.items()
                       if k in ckpt and ckpt[k] != v}
            if diverso:
                print("ERRORE: il checkpoint usa iperparametri diversi da questi.")
                for k, (era, ora) in diverso.items():
                    print(f"  {k}: checkpoint={era}  riga di comando={ora}")
                print("Rilanciate con gli stessi valori, oppure senza --resume")
                print("e con un --tag nuovo per iniziare un run separato.")
                raise SystemExit(1)

    # RIFERIMENTO: la sonda k-NN sull'encoder non ancora addestrato. E' il
    # "modello casuale" - la cosa da battere. Misurarlo QUI, con lo stesso
    # protocollo usato dopo, e' l'unico modo per sapere se il pre-training
    # aggiunge o toglie. Costa una manciata di secondi.
    if knn_ref is None or rif_down is None:
        # Il riferimento si misura su un encoder APPENA INIZIALIZZATO, non su
        # `model`. Al riavvio `model` ha gia' i pesi addestrati caricati:
        # misurarlo qui significherebbe confrontare il modello con se' stesso.
        # Successo Su un resume dall'epoca 149: il riferimento
        # usciva 0.7654 - il valore di quell'epoca - invece dei 0.7512
        # dell'encoder casuale, e il cancello avrebbe giudicato il run contro
        # la sua stessa versione precedente.
        print(""
              "Riferimento (encoder casuale, pesi non addestrati):")
        set_seed()
        casuale = build_ijepa(variant).to(DEVICE).eval()
        knn_ref = run_probe(casuale, records, splits, completo=True)['lineare']
        rif_down, rif_pra, rif_p1 = sonda_downstream(casuale, records, splits)
        del casuale
        torch.cuda.empty_cache()
        set_seed()
        print(f"  [K16 *criterio*] macroF1={rif_down:.4f}  PR-AUC5={rif_pra:.4f}"
              f"   <- E' QUESTO il numero da battere"
              f"     [P1_bbox di controllo {rif_p1:.4f}]", flush=True)
    print(f"Da battere: macro-F1 della sonda LINEARE = {knn_ref:.4f}"
          f"   (cancello: se a {GATE_AT} epoche non e' superato, ci si ferma)")
    knn_best = 0.0
    sotto = 0
    sonde = []

    for epoch in range(start_epoch, epochs):
        t_epoca = time.time()
        model.train()
        meter = AverageMeter()
        # Il monitor vuole abbastanza campioni: con un solo batch il rango
        # effettivo e' limitato dal batch, non dalla salute del modello.
        emb_epoca = []

        for step, batch in enumerate(loader):
            t_passo = time.perf_counter()
            if smoke and step >= 20:
                break
            images = batch["image"].to(DEVICE, non_blocking=True)
            if images.dim() == 5:      # (B, k, C, H, W) -> (B*k, C, H, W)
                images = images.flatten(0, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                loss, embeddings = model(images)

            scaler.scale(loss).backward()

            # Clipping del gradiente: I-JEPA lo usa, e qui serve davvero.
            # Un picco di gradiente all'inizio del training manda la loss a
            # NaN e brucia una nottata di GPU senza avvisare.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            model.update_target(ema_momentum(gstep, total_steps))
            gstep += 1
            meter.update(loss.item(), images.size(0))

            # Segnale di vita: senza, un'epoca da decine di secondi sembra
            # un blocco. Poche righe per epoca, non intasa il log.
            if step % 25 == 0:
                el = time.time() - t_epoca
                print(f"    ep{epoch:03d} step {step:4d}/{len(loader)}  "
                      f"loss={loss.item():.4f}  {el:.0f}s", flush=True)

            if sum(e.shape[0] for e in emb_epoca) < MONITOR_SAMPLES:
                emb_epoca.append(embeddings.detach().float().cpu())

            # Freno a fine passo: abbassa il ciclo di lavoro della GPU.
            brake.pausa(t_passo)
            if term:
                term.controlla()

        knn = None
        if (epoch + 1) % KNN_PROBE_EVERY == 0 or epoch == epochs - 1:
            pieno = (epoch + 1) % RESOCONTO_OGNI == 0 or epoch == epochs - 1
            knn = run_probe(model, records, splits, completo=pieno)
            # Il DOWNSTREAM VERO costa di piu', quindi si misura con la
            # cadenza del resoconto e non a ogni sonda. E' pero' lui il
            # criterio: vedi sonda_downstream().
            if (epoch + 1) % RESOCONTO_OGNI == 0 or epoch == epochs - 1:
                t0 = time.time()
                dwn, pra, p1 = sonda_downstream(model, records, splits)
                knn["downstream"], knn["pr_auc5"], knn["p1_bbox"] = dwn, pra, p1
                print(f"  [K16 *criterio*] macroF1={dwn:.4f}  PR-AUC5={pra:.4f}"
                      f"   (riferimento casuale {rif_down:.4f}, "
                      f"{dwn - rif_down:+.4f})   [{time.time()-t0:.0f}s]",
                      flush=True)
                print(f"  [P1_bbox controllo] {p1:.4f}   (casuale {rif_p1:.4f}, "
                      f"{p1 - rif_p1:+.4f})   -- NON entra nel criterio",
                      flush=True)
            sonde.append({**knn, "epoch": epoch})

        if not emb_epoca:
            print("  [monitor] nessun batch elaborato: dataset vuoto?")
            break
        entry = monitor.update(epoch, meter.avg, torch.cat(emb_epoca), knn)
        dt = time.time() - t_epoca
        rimanenti = epochs - epoch - 1
        print(f"            epoca in {dt:.0f}s   restano {rimanenti} epoche "
              f"(~{rimanenti * dt / 3600:.1f} h)", flush=True)

        if tb is not None:
            tb.add_scalar("loss", meter.avg, epoch)
            # Si logga qualunque scalare numerico ci sia nella voce del
            # monitor (std, eff_rank, rank_ratio...), senza assumerne i nomi.
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(f"monitor/{k}", v, epoch)
            if knn is not None:
                for k, v in knn.items():
                    if isinstance(v, (int, float)):
                        tb.add_scalar(f"sonda/{k}", v, epoch)
            tb.add_scalar("sistema/epoca_secondi", dt, epoch)
            tb.add_scalar("sistema/learning_rate", scheduler.get_last_lr()[0], epoch)
            tb.flush()

        if knn is not None:
            # Si giudica sul DOWNSTREAM quando c'e' - e' la lettura che
            # finisce in presentazione. Nelle sonde intermedie, dove il
            # downstream non e' stato misurato, si ripiega sulla lineare
            # senza mai mescolare le due scale: il riferimento cambia
            # insieme al criterio.
            # Cancello e checkpoint migliore agiscono SOLO sulle sonde in
            # cui il downstream e' stato misurato. Le sonde intermedie si
            # stampano e basta: confrontarle con lo stesso `knn_best`
            # mescolerebbe due scale diverse (~0.77 contro ~0.75) e
            # produrrebbe record falsi o mancati.
            nome_crit, criterio, riferimento = "downstream", None, rif_down
            if "downstream" not in knn:
                print(f"            [sonda intermedia] lineare "
                      f"{knn['lineare']:.4f}  k-NN {knn['knn']:.4f} "
                      f"(nessun giudizio: il criterio e' il downstream)")
            else:
                criterio = knn["downstream"]

                # CHECKPOINT MIGLIORE, tenuto a parte.
                # Il checkpoint normale viene sovrascritto a ogni epoca: se la
                # sonda peggiora - e in questo progetto peggiora sempre, da un
                # certo punto in poi - alla fine resta salvato l'encoder PEGGIORE
                # e quello buono e' perduto. E' successo: la sonda
                # migliore era all'epoca 10 (0.7067) ma sul disco e' rimasta
                # l'epoca 39 (0.4280).
                #
                # Il modello che si consegna e' questo, non l'ultimo: nulla nel
                # brief chiede di addestrare fino all'ultima epoca, e scegliere
                # sulla base di una sonda misurata sul VALIDATION e' un criterio
                # di selezione onesto, da dichiarare in presentazione.
                if criterio > knn_best:
                    save_checkpoint({
                        "model": model.state_dict(), "epoch": epoch,
                        "gstep": gstep, "variant": variant,
                        "downstream": knn.get("downstream"),
                        "probe_lineare": knn['lineare'], "probe_knn": knn['knn'],
                        "probe_ref": knn_ref, "rif_down": rif_down,
                        "lr": LR, "ema_start": EMA_START,
                        "predictor_dim": network.PREDICTOR_DIM,
                    }, run_name + "_best")
                    print(f"            [migliore] nuovo record {criterio:.4f} ({nome_crit}) "
                          f"all'epoca {epoch}: salvato {run_name}_best")
                knn_best = max(knn_best, criterio)
                delta = criterio - riferimento
                soglia = riferimento - GATE_MARGINE

                # Si giudica la sonda CORRENTE, non la migliore mai vista.
                # La versione precedente confrontava knn_best con la soglia: una
                # sola sonda fortunata all'inizio disarmava il cancello per
                # sempre. Successo - k-NN 0.7067 all'epoca 10, poi
                # 0.6520, 0.5548, 0.4280 - e il run e' proseguito verso altre
                # 4.7 ore di degrado con il cancello che taceva, perche' il
                # massimo storico restava sopra la soglia.
                sotto = sotto + 1 if criterio < soglia else 0
                print(f"            [cancello] {nome_crit} {criterio:.4f} vs casuale {riferimento:.4f}"
                      f"  -> {delta:+.4f}   miglior finora {knn_best:.4f}"
                      f"   sonde sotto di fila: {sotto}")

                crollo = criterio < riferimento - GATE_CROLLO * GATE_MARGINE
                if epoch + 1 >= GATE_AT and (sotto >= GATE_SONDE_SOTTO or crollo):
                    motivo = (f"crollo netto ({delta:+.4f}, oltre {GATE_CROLLO} margini)"
                              if crollo else
                              f"{sotto} sonde consecutive sotto il riferimento")
                    print(f"  CANCELLO all'epoca {epoch+1}: {motivo}.")
                    print(f"  {nome_crit} {criterio:.4f} contro encoder casuale {riferimento:.4f}.")
                    print("  Il pre-training sta DEGRADANDO le rappresentazioni: ci si")
                    print("  ferma qui invece di consumare le epoche restanti.")
                    break

            if (epoch + 1) % RESOCONTO_OGNI == 0 and sonde:
                stampa_resoconto(sonde, knn_ref, run_name, epoch + 1, epochs,
                             rif_down=rif_down)

        if monitor.is_collapsing():
            print("\n  COLLASSO RILEVATO. Non insistete: cambiate qualcosa.")
            print("  Ordine di intervento suggerito:")
            print("   1. abbassare il learning rate (fattore 3)")
            print("   2. alzare SSL_EMA_START verso 0.999")
            print("   3. ridurre la capacita' del predictor (e' troppo forte)")
            break

        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "gstep": gstep, "variant": variant,
            "monitor": monitor.history, "knn_ref": knn_ref,
            # Gli iperparametri passati da riga di comando vanno nel
            # checkpoint: senza, un --resume che dimentica --ema-start
            # ripartirebbe con un valore DIVERSO e il run cambierebbe
            # regime a meta' senza dirlo. E' lo stesso tipo di confusione
            # dello scheduler troncato di agosto, che aveva nascosto per
            # giorni il vero effetto di una configurazione.
            "lr": LR, "ema_start": EMA_START,
            "predictor_dim": network.PREDICTOR_DIM,
        }, run_name)

    if tb is not None:
        tb.close()
    monitor.save(os.path.join(FIG_DIR, f"{run_name}_monitor.json"))
    print(f"\nFigura monitoraggio: {monitor.plot(f'{run_name}_monitor')}")
    print(f"Checkpoint: {CKPT_DIR}/{run_name}.pt")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--epochs", type=int, default=SSL_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=SSL_BATCH_SIZE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # Override per gli esperimenti in parallelo. Restano fuori da globals.py
    # apposta: globals descrive la configurazione di riferimento, questi
    # servono a esplorarne le varianti senza toccarla.
    ap.add_argument("--tag", default="", help="suffisso del run (checkpoint separati)")
    ap.add_argument("--context-scale", type=float, nargs=2, default=None,
                    help="es. 0.4 0.7 - contesto piu' piccolo = compito piu' difficile")
    ap.add_argument("--target-scale", type=float, nargs=2, default=None)
    ap.add_argument("--predictor-dim", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema-start", type=float, default=None,
                    help="momentum EMA iniziale: piu' alto = target piu' lento = meno collasso")
    ap.add_argument("--soglia-gpu", type=int, default=0,
                    help="gradi oltre i quali fermarsi finche' la GPU non "
                         "scende di 8. 0 = disattivato. Alternativa a "
                         "--carico: non paga nulla finche' la scheda e' "
                         "fredda, invece di rallentare sempre")
    ap.add_argument("--carico", type=int, default=100,
                    help="percentuale di tempo in cui la GPU lavora. 80 = "
                         "lavora l'80%% e riposa il 20%%, run 1.25x piu' "
                         "lungo. Serve su hardware che si spegne sotto "
                         "carico sostenuto")
    ap.add_argument("--mask-ratio", type=float, nargs=2, default=None,
                    help="es. 0.75 0.85 - rapporto mascherato CONTROLLATO. "
                         "Senza, vale quello del paper: 53.8%%")
    ap.add_argument("--block-scale", type=float, nargs=2, default=None,
                    help="taglia dei blocchi. Grandi = meno blocchi = stesso "
                         "costo per passo a parita' di rapporto")
    ap.add_argument("--gate-epoch", type=int, default=None,
                    help="epoche dopo cui fermarsi se la sonda non batte l'encoder casuale")
    a = ap.parse_args()

    # Si scrivono nei moduli che li leggono a ogni chiamata, cosi' l'override
    # vale per l'intero run senza duplicare la configurazione.
    import network
    import data as data_mod
    if a.lr is not None:
        LR = a.lr
    if a.ema_start is not None:
        EMA_START = a.ema_start
    if a.gate_epoch is not None:
        GATE_AT = a.gate_epoch
    if a.context_scale:
        network.CONTEXT_SCALE = tuple(a.context_scale)
    if a.target_scale:
        network.TARGET_SCALE = tuple(a.target_scale)
    if a.mask_ratio:
        network.MASK_RATIO = tuple(a.mask_ratio)
    if a.block_scale:
        network.BLOCK_SCALE = tuple(a.block_scale)
    if a.predictor_dim is not None:
        network.PREDICTOR_DIM = a.predictor_dim
    if a.workers is not None:
        data_mod.NUM_WORKERS = a.workers

    print(f"[config] tag={a.tag or '-'} context={network.CONTEXT_SCALE} "
          f"target={network.TARGET_SCALE} predictor_dim={network.PREDICTOR_DIM} "
          f"workers={data_mod.NUM_WORKERS}")
    train(a.variant, a.epochs, a.batch_size, a.resume, a.smoke, a.tag,
          carico=a.carico, soglia_gpu=a.soglia_gpu)
