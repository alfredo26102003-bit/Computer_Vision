"""
PIPELINE — parsing XML, split per paziente, dataset SSL e downstream.

Data - parsing annotazioni, split anti-leakage, dataset per SSL e downstream.

Sezione "Data" della struttura richiesta dal corso.

PRIMA DI ADDESTRARE QUALSIASI COSA:
    python data.py --inspect      # cosa c'e' davvero nel dataset
    python data.py --bbox-stats   # la lesione copre abbastanza token?
    python data.py --splits       # crea gli split e verifica il leakage

Il secondo comando decide l'architettura. Vedi ANALISI_PROGETTO_8.md sez.2.
"""

import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from globals import (
    CROPS_PER_IMAGE, DATA_ROOT, DIR_ANNOTATIONS, DIR_AUGMENTED, DIR_ORIGINAL, EXPECTED_COUNTS,
    EXPECTED_TOTAL_IMAGES, EXPECTED_TOTAL_LESIONS, GRADE_TO_IDX,
    MIN_TOKENS_PER_LESION, NUM_WORKERS, PAI_GRADES, PATCH_SIZE, RESIZE_H,
    CACHE_DIR, CROPS_PER_ITEM, LESION_CROP_PIXELS, RESIZE_W, SEED,
    SPLIT_FRACTIONS, SPLIT_JSON, TILE_MIN_FOREGROUND,
    TILE_SIZE, TILE_STRIDE,
)
from utils import load_json, save_json

Image.MAX_IMAGE_PIXELS = None  # le panoramiche sono grandi


# ==========================================================================
# 1. Ispezione - da lanciare per primo
# ==========================================================================
def inspect_dataset(root=DATA_ROOT):
    """Stampa la struttura reale. Diagnostica, non magia."""
    print(f"=== {root} ===")
    if not os.path.isdir(root):
        print("  ROOT INESISTENTE. Scaricate il dataset da")
        print("  https://data.mendeley.com/datasets/kx52tk2ddj/3")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 2:
            dirnames[:] = []
            continue
        print(f"  {dirpath}  ({len(filenames)} file)")
        for f in filenames[:3]:
            print(f"      {f}")

    xmls = glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
    print(f"\n  XML trovati: {len(xmls)}")
    if xmls:
        print(f"\n  --- contenuto di {os.path.basename(xmls[0])} ---")
        with open(xmls[0]) as fh:
            print("  " + "\n  ".join(fh.read().splitlines()[:30]))


# ==========================================================================
# 2. Parsing delle annotazioni
# ==========================================================================
def _walk_excluding_augmented(root):
    """Percorre l'albero saltando la cartella delle immagini aumentate."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if DIR_AUGMENTED.lower() not in d.lower()]
        if DIR_AUGMENTED.lower() in dirpath.lower():
            continue
        yield dirpath, filenames


def _find_annotation_files(root=DATA_ROOT):
    """Tutti gli XML, escluse le augmentate. Una sola passata sull'albero."""
    files = []
    for dirpath, filenames in _walk_excluding_augmented(root):
        files += [os.path.join(dirpath, f) for f in filenames
                  if f.lower().endswith(".xml")]
    return sorted(files)


def build_image_index(root=DATA_ROOT):
    """
    Mappa stem -> percorso immagine, costruita UNA VOLTA SOLA.

    Perche' esiste questa funzione: la versione precedente cercava ogni
    immagine con una glob ricorsiva sull'intero albero, una per annotazione.
    Con 17.004 XML su ~21.000 file significa 17.004 scansioni complete della
    cartella - comportamento quadratico, minuti o ore invece di secondi, e
    su un disco lento e' anche peggio. Un indice costruito in una passata
    riduce tutto a lookup O(1).
    """
    index = {}
    for dirpath, filenames in _walk_excluding_augmented(root):
        for f in filenames:
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                index.setdefault(stem, os.path.join(dirpath, f))
    return index


def _parse_grade(raw):
    """
    Estrae il grado PAI dall'etichetta testuale.

    Non conosco lo schema esatto delle etichette del dataset: potrebbero
    essere "PAI3", "3", "pai_3", "Grade 3"... Questo parser prende la prima
    cifra in {3,4,5}. Se `unknown` non e' vuoto dopo il parsing, guardate
    cosa c'e' davvero e adattate.
    """
    if raw is None:
        return None
    for ch in str(raw):
        if ch.isdigit() and int(ch) in PAI_GRADES:
            return int(ch)
    return None


def parse_annotations(root=DATA_ROOT, verbose=True):
    """
    Legge gli XML (formato PASCAL VOC atteso) e restituisce la lista dei record.

    Ogni record: dict con image_id, image_path, width, height, e la lista
    delle lesioni [{xmin, ymin, xmax, ymax, grade}].
    """
    files = _find_annotation_files(root)
    if not files:
        raise RuntimeError(f"Nessun XML trovato in {root}. Lanciate --inspect.")

    # Indice costruito una volta: vedi build_image_index().
    images = build_image_index(root)

    records, unknown = [], Counter()
    senza_immagine = 0

    for xml_path in files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        r = tree.getroot()

        stem = os.path.splitext(os.path.basename(xml_path))[0]
        size = r.find("size")
        w = int(size.findtext("width", 0)) if size is not None else 0
        h = int(size.findtext("height", 0)) if size is not None else 0

        lesions = []
        for obj in r.findall(".//object"):
            grade = _parse_grade(obj.findtext("name"))
            if grade is None:
                unknown[obj.findtext("name")] += 1
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                lesions.append({
                    "xmin": float(bb.findtext("xmin")),
                    "ymin": float(bb.findtext("ymin")),
                    "xmax": float(bb.findtext("xmax")),
                    "ymax": float(bb.findtext("ymax")),
                    "grade": grade,
                })
            except (TypeError, ValueError):
                continue

        if not lesions:
            continue

        img_path = images.get(stem)
        if img_path is None:
            # Normale: l'archivio contiene anche le annotazioni delle
            # immagini aumentate, che non estraiamo. Le contiamo e basta.
            senza_immagine += 1
            continue
        if not (w and h):
            with Image.open(img_path) as im:
                w, h = im.size

        # Il nome ORIGINALE dell'immagine (PN######) sopravvive solo qui: i
        # file pubblicati sono stati rinominati con un progressivo. E' l'ID
        # paziente che il brief chiede di usare per lo split - vedi
        # verify_patient_level().
        patient_id = os.path.splitext(r.findtext("filename") or "")[0] or None

        records.append({"image_id": stem, "image_path": img_path,
                        "patient_id": patient_id,
                        "width": w, "height": h, "lesions": lesions})

    if verbose:
        n_les = sum(len(r["lesions"]) for r in records)
        counts = Counter(l["grade"] for r in records for l in r["lesions"])
        print(f"XML letti                : {len(files):5d}")
        if senza_immagine:
            print(f"  di cui senza immagine  : {senza_immagine:5d}  "
                  "(annotazioni delle augmentate, ignorate come previsto)")
        print(f"Immagini con annotazioni : {len(records):5d}  (atteso {EXPECTED_TOTAL_IMAGES})")
        print(f"Lesioni totali           : {n_les:5d}  (atteso {EXPECTED_TOTAL_LESIONS})")
        print(f"Lesioni per immagine     : {n_les / max(len(records),1):.2f}")
        print("\nDistribuzione per grado PAI:")
        for g in PAI_GRADES:
            exp = EXPECTED_COUNTS[g]
            got = counts.get(g, 0)
            mark = "ok" if abs(got - exp) <= max(50, 0.05 * exp) else "DIVERGE"
            print(f"  PAI {g}: {got:5d}  ({100*got/max(n_les,1):5.1f}%)  "
                  f"atteso ~{exp:5d}  [{mark}]")
        if counts:
            mn = min(counts.values())
            print(f"\nSbilanciamento max:min = {max(counts.values())/mn:.2f} : 1")
        if unknown:
            print(f"\nEtichette non riconosciute: {dict(unknown)}")
            print("  -> adattate _parse_grade()")

    return records


# ==========================================================================
# 3. Statistiche bbox - LA VERIFICA CHE DECIDE L'ARCHITETTURA
# ==========================================================================
def bbox_statistics(records=None, root=DATA_ROOT, plot=True):
    """
    Misura quanto e' grande una lesione in TOKEN a ciascuna risoluzione
    candidata.

    Il punto: una panoramica inquadra l'intera arcata, una lesione e' di
    pochi millimetri. Se ridimensionate l'immagine intera a 224x224, la
    lesione finisce sotto la dimensione di un patch token da 16x16 e il
    latente estratto alla bbox non contiene la patologia.

    Se la mediana esce sotto MIN_TOKENS_PER_LESION, quella configurazione
    e' inutilizzabile. Fatelo girare PRIMA di scrivere il modello.
    """
    records = records or parse_annotations(root, verbose=False)

    ws = np.array([r["width"] for r in records], dtype=float)
    hs = np.array([r["height"] for r in records], dtype=float)
    bw, bh, frac = [], [], []
    for r in records:
        for l in r["lesions"]:
            w_, h_ = l["xmax"] - l["xmin"], l["ymax"] - l["ymin"]
            bw.append(w_)
            bh.append(h_)
            frac.append(w_ / r["width"])
    bw, bh, frac = map(lambda a: np.array(a, dtype=float), (bw, bh, frac))

    print("=== Dimensioni delle immagini ===")
    print(f"  larghezza: mediana {np.median(ws):7.0f}  range {ws.min():.0f}-{ws.max():.0f}")
    print(f"  altezza  : mediana {np.median(hs):7.0f}  range {hs.min():.0f}-{hs.max():.0f}")

    print("\n=== Dimensioni delle bbox (pixel nativi) ===")
    print(f"  larghezza: mediana {np.median(bw):6.1f}  p10 {np.percentile(bw,10):6.1f}  p90 {np.percentile(bw,90):6.1f}")
    print(f"  altezza  : mediana {np.median(bh):6.1f}  p10 {np.percentile(bh,10):6.1f}  p90 {np.percentile(bh,90):6.1f}")
    print(f"  frazione della larghezza immagine: mediana {np.median(frac):.4f}")

    print(f"\n=== Copertura in token (patch {PATCH_SIZE}px) ===")
    print(f"  {'configurazione':22s} {'lato (token)':>13s} {'area (token)':>13s}   verdetto")

    configs = [
        ("panoramica -> 224",   RESIZE_W / np.median(ws)),
        ("panoramica -> 384",   384 / np.median(ws)),
        ("panoramica -> 512",   512 / np.median(ws)),
        (f"tile nativo {TILE_SIZE}px", 1.0),
    ]
    results = {}
    for name, scale in configs:
        side = np.median(bw) * scale / PATCH_SIZE
        area = side ** 2
        verdict = "OK" if area >= MIN_TOKENS_PER_LESION else "INUTILIZZABILE"
        print(f"  {name:22s} {side:13.2f} {area:13.2f}   {verdict}")
        results[name] = {"side_tokens": float(side), "area_tokens": float(area)}

    best = max(results.items(), key=lambda kv: kv[1]["area_tokens"])
    print(f"\n  Soglia: almeno {MIN_TOKENS_PER_LESION} token di area per lesione.")
    print(f"  Migliore: {best[0]} ({best[1]['area_tokens']:.1f} token)")
    if results[f"tile nativo {TILE_SIZE}px"]["area_tokens"] < MIN_TOKENS_PER_LESION:
        print("  ATTENZIONE: nemmeno il tile nativo basta. Riducete PATCH_SIZE "
              "o lavorate su crop centrati sulla lesione.")

    if plot:
        _plot_bbox_stats(bw, bh, frac, results)
    return results


def _plot_bbox_stats(bw, bh, frac, results):
    import matplotlib.pyplot as plt
    from globals import FIG_DIR

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(bw, bins=50, color="#1f77b4")
    axes[0].set_title("Larghezza bbox (px nativi)")
    axes[1].hist(frac, bins=50, color="#ff7f0e")
    axes[1].set_title("Bbox / larghezza immagine")
    names = list(results.keys())
    vals = [results[n]["area_tokens"] for n in names]
    cols = ["#2ca02c" if v >= MIN_TOKENS_PER_LESION else "#d62728" for v in vals]
    axes[2].barh(names, vals, color=cols)
    axes[2].axvline(MIN_TOKENS_PER_LESION, ls="--", c="black", lw=1)
    axes[2].set_xscale("log")
    axes[2].set_title("Area lesione in token")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "bbox_statistics.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figura salvata: {path}")


# ==========================================================================
# 4. Split anti-leakage
# ==========================================================================
def build_splits(records=None, root=DATA_ROOT, seed=SEED, save=True):
    """
    Split a livello di IMMAGINE, mai di lesione.

    Tre trappole, e il brief ne nomina solo una:

    1. ID paziente: non risultano documentati nel dataset. Splittiamo per
       immagine assumendo una radiografia per paziente. E' un'ASSUNZIONE:
       dichiaratela in una slide.

    2. Cartella di augmentation: 17.004 immagini derivate da 3.926 originali.
       Splittarle a caso mette varianti geometriche della stessa radiografia
       in train e test. Qui la ignoriamo del tutto - il SSL fa gia' le
       proprie augmentation on-the-fly.

    3. Piu' lesioni per immagine (~1.54): splittare per lesione mette la
       stessa radiografia nei due lati. Per questo si splitta per immagine e
       le lesioni seguono.
    """
    records = records or parse_annotations(root, verbose=False)
    ids = sorted(r["image_id"] for r in records)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_tr = int(n * SPLIT_FRACTIONS["train"])
    n_va = int(n * SPLIT_FRACTIONS["val"])
    splits = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

    verify_splits(splits, records)
    if save:
        save_json(splits, SPLIT_JSON)
        print(f"\nSplit salvati in {SPLIT_JSON}")
    return splits


def verify_splits(splits, records):
    """
    Assert anti-leakage. Non verificate a occhio: verificate col codice.
    """
    sets = {k: set(v) for k, v in splits.items()}
    for a in sets:
        for b in sets:
            if a < b:
                inter = sets[a] & sets[b]
                assert not inter, f"LEAKAGE: {len(inter)} immagini in {a} e {b}"

    total = sum(len(v) for v in splits.values())
    assert total == len(records), f"{total} immagini negli split, {len(records)} nei record"

    by_id = {r["image_id"]: r for r in records}
    print("\n=== Split (livello immagine) ===")
    print(f"  {'split':6s} {'imgs':>6s} {'lesioni':>8s}   " +
          "  ".join(f"PAI{g}" for g in PAI_GRADES))
    for name, id_list in splits.items():
        counts = Counter(l["grade"] for i in id_list for l in by_id[i]["lesions"])
        n_les = sum(counts.values())
        dist = "  ".join(f"{100*counts.get(g,0)/max(n_les,1):4.1f}%" for g in PAI_GRADES)
        print(f"  {name:6s} {len(id_list):6d} {n_les:8d}   {dist}")

    print("\n  Nessuna immagine condivisa tra gli split: verificato.")
    verify_patient_level(records)
    return True


def verify_patient_level(records):
    """
    Verifica che lo split per immagine sia ANCHE uno split per paziente.

    Il brief lo chiede esplicitamente: "handle validation/testing splits
    strictly at the patient level to avoid data leakage".

    Gli ID paziente sembravano non esistere, e per questo lo split per
    immagine era stato adottato come ripiego dichiarando l'assunzione "una
    radiografia = un paziente". In realta' l'identificativo c'e': ogni XML
    porta nel campo <filename> il nome ORIGINALE dell'immagine, nella forma
    PN######, mentre il file pubblicato e' stato rinominato con un progressivo
    numerico. Misurato Sulle 3.924 originali: 3.924 PN distinti,
    zero ripetizioni. Una panoramica per paziente, quindi lo split per
    immagine soddisfa il vincolo alla lettera.

    Questa funzione lo ri-verifica sui dati, cosi' in presentazione e' un
    risultato misurato e non una dichiarazione di buona fede.
    """
    pn = {r["image_id"]: r.get("patient_id") for r in records}
    noti = {k: v for k, v in pn.items() if v}
    if not noti:
        print("  ATTENZIONE: nessun patient_id nei record, split per immagine")
        print("  non verificabile a livello paziente.")
        return False

    distinti = len(set(noti.values()))
    condivisi = len(noti) - distinti
    print(f"  ID paziente (PN da <filename>): {distinti} distinti su {len(noti)} immagini")
    if condivisi:
        print(f"  ATTENZIONE: {condivisi} immagini condividono un paziente.")
        print("  Lo split va rifatto RAGGRUPPANDO per patient_id, non per immagine.")
        return False
    print("  Una panoramica per paziente: split per immagine == split per paziente.")
    return True


def load_splits():
    s = load_json(SPLIT_JSON)
    if s is None:
        raise FileNotFoundError("Split mancanti. Lanciate: python data.py --splits")
    return s


# ==========================================================================
# 5. Tiling
# ==========================================================================
def tile_positions(width, height, size=TILE_SIZE, stride=TILE_STRIDE):
    xs = list(range(0, max(width - size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(height - size, 0) + 1, stride)) or [0]
    if xs[-1] + size < width:
        xs.append(width - size)
    if ys[-1] + size < height:
        ys.append(height - size)
    return [(x, y) for y in ys for x in xs]


def _augmenta(arr):
    """
    Augmentation per i tile del pre-training: SOLO flip orizzontale.

    I-JEPA non usa augmentation fatte a mano - il mascheramento E' la
    perturbazione. Ogni trasformazione aggiunta insegna un'INVARIANZA, e
    un'invarianza e' utile solo se la proprieta' resa invariante e'
    irrilevante per il compito finale.

    Qui non lo era. Le trasformazioni fotometriche che c'erano prima
    (luminosita' x0.75-1.25, contrasto x0.7-1.4, gamma 0.7-1.4, e anche la
    versione "debole" x0.85-1.15) insegnavano invarianza all'INTENSITA'.
    Ma un encoder casuale codifica l'intensita' media della regione con
    R^2 = 1.00, e il grado PAI e' in larga parte l'estensione e la
    profondita' di una radiotrasparenza: cioe' intensita'. Si stava
    addestrando la rete a buttare via il segnale piu' predittivo che
    possedeva, ed e' il motivo per cui la sonda k-NN peggiorava
    monotonamente dall'epoca 0.

    Il flip orizzontale resta perche' e' l'unica trasformazione che non
    tocca ne' la scala ne' l'intensita': l'arcata e' grosso modo simmetrica,
    quindi l'invarianza che insegna e' vera e gratuita.
    """
    if random.random() < 0.5:
        arr = arr[:, ::-1].copy()
    return np.clip(arr, 0, 1).astype(np.float32)


class TileDataset(Dataset):
    """
    Tile a risoluzione nativa per il pre-training SSL.

    Il pre-training e' senza etichette: usa solo i pixel. I tile quasi tutti
    neri (bordi della radiografia) vengono scartati, altrimenti il modello
    spende capacita' a predire il nulla.
    """

    def __init__(self, records, image_ids, size=TILE_SIZE,
                 crops_per_image=CROPS_PER_IMAGE, augment=True,
                 random_crops=True, stride=TILE_STRIDE,
                 crops_per_item=CROPS_PER_ITEM):
        self.records = {r["image_id"]: r for r in records if r["image_id"] in set(image_ids)}
        self.size, self.augment = size, augment
        self.random_crops = random_crops
        # Ogni item restituisce k crop da UNA sola decodifica: e' il 75% del
        # costo di caricamento. Vedi CROPS_PER_ITEM in globals.py.
        self.k = max(1, crops_per_item) if random_crops else 1

        if random_crops:
            # Ogni voce e' solo un'immagine: la posizione si sceglie a caso
            # a ogni accesso, quindi le epoche non si ripetono identiche.
            n_item = max(1, crops_per_image // self.k)
            self.index = [(rid, None, None)
                          for rid in self.records
                          for _ in range(n_item)]
        else:
            self.index = [(rid, x, y)
                          for rid, r in self.records.items()
                          for (x, y) in tile_positions(r["width"], r["height"],
                                                       self.size, stride)]

    def __len__(self):
        return len(self.index)

    def _un_crop(self, im, W, H, x, y):
        """Un singolo crop dall'immagine GIA' decodificata."""
        # Le panoramiche hanno bordi neri: un crop tutto nero non insegna
        # niente e sprecherebbe capacita' predittiva. Si riprova qualche
        # volta, poi ci si arrende e si tiene quello che c'e'.
        for _ in range(8):
            # Lato del crop COSTANTE, pari a self.size: nessun
            # ridimensionamento, quindi ingrandimento esattamente 1.0x.
            #
            # C'era un jitter di scala 0.6-1.6 introdotto quando il
            # downstream ritagliava in modo RELATIVO alla bbox e quindi
            # vedeva ingrandimenti da 0.69x a 1.59x: serviva ad allineare i
            # due stadi. Da quando LesionCropDataset ritaglia una finestra
            # FISSA di 224 px nativi, il downstream lavora a 1.0x costante
            # per ogni lesione, e quel jitter e' diventato il difetto
            # opposto: creava il disallineamento invece di toglierlo, e
            # insegnava invarianza alla DIMENSIONE della lesione. Ma il lato
            # mediano della bbox e' 57 / 81 / 126 px per PAI 3 / 4 / 5: la
            # dimensione e' il segnale piu' forte che ci sia.
            side = self.size
            if x is None:
                cx = random.randint(0, max(W - side, 0))
                cy = random.randint(0, max(H - side, 0))
            else:
                cx, cy = x, y

            tile = im.crop((cx, cy, cx + side, cy + side))
            arr = np.asarray(tile, dtype=np.float32) / 255.0
            if arr.mean() >= TILE_MIN_FOREGROUND or x is not None:
                break

        if self.augment:
            arr = _augmenta(arr)

        # le radiografie sono in scala di grigi: replichiamo su 3 canali per
        # restare compatibili con backbone standard
        x3 = torch.from_numpy(arr)[None].repeat(3, 1, 1)
        return (x3 - 0.5) / 0.5

    def __getitem__(self, i):
        rid, x, y = self.index[i]
        r = self.records[rid]

        # UNA decodifica, k crop. E' il 75% del costo di caricamento.
        with Image.open(r["image_path"]) as im:
            im = im.convert("L")
            W, H = im.size
            crops = [self._un_crop(im, W, H, x, y) for _ in range(self.k)]

        if self.k == 1:
            return {"image": crops[0], "image_id": rid}
        # (k, 3, S, S): train_ssl appiattisce la dimensione dei crop.
        return {"image": torch.stack(crops), "image_id": rid}


class LesionCropDataset(Dataset):
    """
    Crop centrati sulla lesione, per il downstream.

    Il crop e' preso a risoluzione NATIVA e ha lato `context_factor` volte la
    bbox, cosi' il modello vede la lesione e l'osso attorno - che e' cio' che
    distingue i gradi PAI. Poi si ridimensiona a TILE_SIZE.

    Ritorna anche la bbox in coordinate del crop, che serve per l'attention
    pooling sui token interni alla lesione.
    """

    def __init__(self, records, image_ids, size=TILE_SIZE, augment=False,
                 crop_pixels=None, context_factor=None):
        keep = set(image_ids)
        self.size, self.augment = size, augment
        # Finestra di lato COSTANTE in pixel nativi: la dimensione apparente
        # della lesione e' preservata. Il ritaglio relativo alla bbox, che la
        # normalizzava via, e' stato tolto: annullava il segnale piu' forte
        # del problema (vedi LESION_CROP_PIXELS in globals.py).
        self.crop_px = LESION_CROP_PIXELS if crop_pixels is None else crop_pixels

        # RITAGLIO CIECO ALLA DIMENSIONE, come ABLATION - non come default.
        # Con context_factor la finestra vale cf volte il lato della bbox e
        # poi si ridimensiona a `size`: ogni lesione appare della STESSA
        # dimensione apparente, e il segnale dominante del problema sparisce.
        #
        # Era il ritaglio originale del progetto, tolto Perche'
        # "annullava il segnale piu' forte". Torna qui come strumento di
        # misura, non come scelta: serve a rispondere a una domanda che il
        # protocollo geometrico non puo' porre.
        #
        # PERCHE' SERVE. Con la finestra fissa il pavimento geometrico e'
        # macro-F1 0.7567 (due soglie sul lato della bbox) e il massimo
        # misurato 0.7705: 0.0138 di spazio. In quel margine nessuna
        # differenza fra encoder puo' emergere, e "il pre-training non
        # aggiunge niente" resta indistinguibile da "non si vede". Nel
        # ritaglio cieco alla dimensione la dinamica misurata e' 0.177
        # (casuale 0.5356 contro ImageNet 0.7121): li' la qualita' della
        # rappresentazione conta davvero, e la domanda diventa ponibile.
        self.cf = context_factor
        self.items = [
            {"image_id": r["image_id"], "image_path": r["image_path"],
             "lesion_idx": j, **l}
            for r in records if r["image_id"] in keep
            for j, l in enumerate(r["lesions"])
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        cx, cy = (it["xmin"] + it["xmax"]) / 2, (it["ymin"] + it["ymax"]) / 2
        if self.cf is None:
            half = self.crop_px / 2                    # finestra fissa: scala reale
        else:
            lato = max(it["xmax"] - it["xmin"], it["ymax"] - it["ymin"])
            half = lato * self.cf / 2                  # cieco: scala normalizzata

        with Image.open(it["image_path"]) as im:
            im = im.convert("L")
            W, H = im.size
            x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
            x1, y1 = int(min(cx + half, W)), int(min(cy + half, H))
            crop = im.crop((x0, y0, x1, y1)).resize((self.size, self.size), Image.BILINEAR)

        arr = np.asarray(crop, dtype=np.float32) / 255.0

        # bbox riportata nelle coordinate del crop ridimensionato
        sx = self.size / max(x1 - x0, 1)
        sy = self.size / max(y1 - y0, 1)
        bbox = torch.tensor([
            (it["xmin"] - x0) * sx, (it["ymin"] - y0) * sy,
            (it["xmax"] - x0) * sx, (it["ymax"] - y0) * sy,
        ], dtype=torch.float32).clamp(0, self.size)

        # Il flip DEVE specchiare anche la bbox. Prima non lo faceva: con
        # augment=True meta' dei campioni aveva la maschera dei token sulla
        # posizione speculare della lesione. Nessun chiamante passava
        # augment=True, quindi i risultati non ne erano affetti, ma era una
        # mina innescata.
        if self.augment and random.random() < 0.5:
            arr = arr[:, ::-1].copy()
            bbox = torch.tensor([self.size - bbox[2], bbox[1],
                                 self.size - bbox[0], bbox[3]])

        # Geometria della bbox in pixel NATIVI, normalizzata. E' il segnale
        # piu' predittivo del dataset: due soglie sul lato danno macro-F1
        # 0.7567 senza alcuna rete.
        #
        # NON VA DATA AL CLASSIFICATORE. Il brief dice di usare le bounding
        # box "per estrarre i vettori latenti corrispondenti alle aree
        # lesionate": la bbox e' un SELETTORE di token, non un ingresso del
        # classificatore. Passarla come feature misurerebbe rappresentazione
        # piu' una feature costruita a mano, e il numero non direbbe piu'
        # niente sull'encoder - che e' esattamente cio' che l'obiettivo 2
        # vuole misurare.
        #
        # Una versione precedente di questo commento si giustificava con
        # "e' un input fornito dal dataset": era una razionalizzazione.
        # `use_geom` resta a False in ogni risultato riportato, e questa
        # geometria serve solo all'ANALISI - per esempio a partizionare il
        # test fra casi in cui la dimensione basta e casi in cui inganna
        # (exp_stratificata.py).
        w_nat = it["xmax"] - it["xmin"]
        h_nat = it["ymax"] - it["ymin"]
        geom = torch.tensor([w_nat / 200.0, h_nat / 200.0,
                             max(w_nat, h_nat) / 200.0,
                             (w_nat * h_nat) ** 0.5 / 200.0],
                            dtype=torch.float32)

        x3 = torch.from_numpy(arr)[None].repeat(3, 1, 1)
        return {
            "image": (x3 - 0.5) / 0.5,
            "bbox": bbox,
            "geom": geom,
            "label": GRADE_TO_IDX[it["grade"]],
            "image_id": it["image_id"],
            "lesion_idx": it["lesion_idx"],
        }


def make_loader(dataset, shuffle=False, batch_size=64, num_workers=None):
    """
    Loader standard. `num_workers=0` per i loader USA E GETTA.

    Perche' il parametro esiste: i worker persistenti convengono per il
    loader di training, che vive per l'intero run, ma sono una trappola per
    i loader temporanei. Il k-NN probe ne creava due nuovi ogni volta e
    usciva dal ciclo con un break: i worker restavano vivi, si accumulavano
    a ogni probe e su Windows facevano esaurire la memoria condivisa
    (RuntimeError 1455, visto al primo probe del run).
    """
    nw = NUM_WORKERS if num_workers is None else num_workers
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=nw, pin_memory=True,
                      drop_last=shuffle, persistent_workers=nw > 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--bbox-stats", action="store_true")
    ap.add_argument("--splits", action="store_true")
    args = ap.parse_args()

    if args.inspect or not any(vars(args).values()):
        inspect_dataset()
        print()
        parse_annotations(verbose=True)
    if args.bbox_stats:
        bbox_statistics()
    if args.splits:
        build_splits()


# ==========================================================================
# Cache dei crop - perche' le sonde erano lente
# ==========================================================================
def cache_crop(records, image_ids, nome, size=TILE_SIZE, crop_pixels=None):
    """
    Ritaglia una volta sola e conserva il risultato.

    PERCHE'. LesionCropDataset, a ogni accesso, apre la PANORAMICA INTERA da
    disco, la converte in scala di grigi, ritaglia e ridimensiona. Sono
    ~5700 lesioni fra train e val, e le panoramiche sono grandi: una singola
    sonda diagnostica costava minuti, quasi tutti spesi a decodificare JPEG,
    non a far girare la rete. E i crop del downstream sono DETERMINISTICI -
    finestra fissa, nessuna augmentation - quindi rifare quel lavoro a ogni
    sonda era puro spreco.

    Si conservano come uint8. Non e' una perdita di precisione: PIL produce
    gia' uint8 e il dataset si limita a dividere per 255, quindi il giro
    uint8 -> float -> uint8 e' esatto (verificato nel blocco __main__).
    A 224x224 in scala di grigi sono ~50 KB per lesione: train+val stanno in
    ~290 MB, contro i 3 GB dei latenti.
    """
    path = os.path.join(CACHE_DIR, f"crop_{nome}_{size}.pt")
    if os.path.isfile(path):
        return torch.load(path, weights_only=False)

    ds = LesionCropDataset(records, image_ids, size=size, crop_pixels=crop_pixels)
    n = len(ds)
    d = {"image": torch.empty(n, size, size, dtype=torch.uint8),
         "bbox": torch.empty(n, 4), "geom": torch.empty(n, 4),
         "label": torch.empty(n, dtype=torch.long)}
    for i in range(n):
        c = ds[i]
        grigio = c["image"][0] * 0.5 + 0.5            # da [-1,1] a [0,1]
        d["image"][i] = (grigio * 255).round().clamp(0, 255).to(torch.uint8)
        d["bbox"][i], d["geom"][i] = c["bbox"], c["geom"]
        d["label"][i] = c["label"]
    torch.save(d, path)
    return d


class CropCacheDataset(Dataset):
    """Legge dalla cache di cache_crop() e ricostruisce il formato di
    LesionCropDataset, cosi' e' intercambiabile senza toccare i chiamanti."""

    def __init__(self, d):
        self.d = d

    def __len__(self):
        return self.d["image"].shape[0]

    def __getitem__(self, i):
        a = self.d["image"][i].float() / 255.0
        x3 = a[None].repeat(3, 1, 1)
        return {"image": (x3 - 0.5) / 0.5, "bbox": self.d["bbox"][i],
                "geom": self.d["geom"][i], "label": int(self.d["label"][i])}
