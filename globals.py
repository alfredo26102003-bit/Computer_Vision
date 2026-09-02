"""
PIPELINE — iperparametri e percorsi condivisi.

Globals - iperparametri e path condivisi.

Sezione "Globals" della struttura richiesta dal corso.
Ogni costante sta QUI: se un numero magico compare altrove, spostatelo.
"""

import os
import torch

# ----------------------------------------------------------------- ambiente
#
# ATTENZIONE: `os.path.isdir("/kaggle/input")` NON basta per riconoscere
# Kaggle. Su Google Colab quella cartella esiste comunque - c'e'
# un'integrazione Kaggle preinstallata - ma e' VUOTA. Con il solo isdir, il
# codice su Colab cercava il dataset in /kaggle/input/panoramic-periapical-
# lesions e riportava "Nessun XML trovato", mentre sul disco ce n'erano
# 17.004. Un fallimento che indica il posto sbagliato invece della causa.
#
# Si richiede quindi che la cartella esista E contenga qualcosa. In piu'
# PERIAPICAL_DATA e PERIAPICAL_OUT permettono di imporre i percorsi
# dall'esterno, senza toccare il codice: e' la via d'uscita quando il
# rilevamento automatico sbaglia comunque.
_kaggle_input = "/kaggle/input"
ON_KAGGLE = os.path.isdir(_kaggle_input) and bool(os.listdir(_kaggle_input))

if ON_KAGGLE:
    # Caricate il dataset Mendeley come Kaggle Dataset e aggiustate il nome.
    DATA_ROOT = "/kaggle/input/panoramic-periapical-lesions"
    OUT_DIR = "/kaggle/working"
else:
    DATA_ROOT = "./data/periapical"
    OUT_DIR = "./runs"

DATA_ROOT = os.environ.get("PERIAPICAL_DATA", DATA_ROOT)
OUT_DIR = os.environ.get("PERIAPICAL_OUT", OUT_DIR)

# Le tre cartelle del dataset Mendeley (DOI 10.17632/kx52tk2ddj.3).
# ATTENZIONE ai nomi: l'archivio Mendeley usa "Image Annots", non
# "Image Annotations" come lascerebbe pensare la descrizione del dataset.
# La struttura reale dentro lo zip e':
#   Periapical Dataset/Periapical Lesions/Original JPG Images/   3924 jpg
#   Periapical Dataset/Periapical Lesions/Image Annots/         17004 xml
#   Periapical Dataset/Periapical Lesions/Augmentation JPG Images/ (non usata)
DIR_ORIGINAL = "Original JPG Images"
DIR_AUGMENTED = "Augmentation JPG Images"   # NON usarla: vedi data.py
DIR_ANNOTATIONS = "Image Annots"

CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUT_DIR, "figures")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
SPLIT_JSON = os.path.join(OUT_DIR, "splits.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Su Windows ogni worker e' un processo nuovo che re-importa il modulo, quindi
# l'avvio costa; con persistent_workers=True lo si paga una volta sola.
# Se DataLoader da' problemi su Windows, scendete a 0.
# 12 e non 16, ed e' una scelta HARDWARE, non di prestazioni.
#
# La macchina e' un portatile (i9-13980HX + RTX 4080 Laptop, alimentatore
# 330 W) e Ha subito TRE spegnimenti improvvisi - eventi
# Kernel-Power 41 alle 16:46, 20:34 e 23:55 - tutti durante nostri
# addestramenti. Con 16 worker che decodificano JPEG su 24 core piu' la GPU
# sotto carico, i picchi di assorbimento e le temperature superano quello che
# il sistema regge in modo continuativo.
#
# 12 worker costano circa il 30% di tempo in piu' per epoca, ma un run di 8
# ore che si spegne a meta' costa molto di piu'. Se lanciate piu' run in
# parallelo, dividete questo numero fra loro.
NUM_WORKERS = 12


def amp_dtype():
    """
    Precisione mista da usare.

    bfloat16 su GPU con compute capability >= 8.0 (Ampere in poi): stesso
    range dinamico del float32, quindi niente overflow del gradiente e
    nessun bisogno del GradScaler. Per il SSL conta: I-JEPA con EMA e
    smooth-L1 e' sensibile agli scalamenti della loss, e il bf16 toglie di
    mezzo un'intera classe di instabilita'.

    Sulle GPU piu' vecchie (Turing, es. la T4 di Kaggle) il bf16 non esiste
    e si ripiega su float16 con GradScaler.
    """
    if DEVICE.type != "cuda":
        return torch.float32
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16

# ------------------------------------------------------------------ classi
# Il PAI (Periapical Index) e' una scala ORDINALE: 3 < 4 < 5.
# Trattarlo come 3 classi senza relazione butta via informazione, ed e' il
# motivo per cui in evaluation.py c'e' il kappa quadratico pesato.
PAI_GRADES = [3, 4, 5]
NUM_CLASSES = len(PAI_GRADES)
GRADE_TO_IDX = {g: i for i, g in enumerate(PAI_GRADES)}

# ATTENZIONE: i numeri del brief NON corrispondono al dataset scaricato.
#
#              brief      misurato sulla v3      differenza
#   totale      6029          ~6735                 +706
#   PAI 3       3691          ~4274                 +583
#   PAI 4       1817(*)       ~1753                  -64
#   PAI 5        521           ~708                 +187
#   immagini    3926           3924                   -2
#   (*) valore derivato per sottrazione, non dichiarato nel brief
#
# Lo sbilanciamento reale e' quindi ~6.0:1, non 7.08:1, e ci sono 187 casi
# PAI 5 in piu' del previsto: buona notizia, perche' la classe rara e' il
# collo di bottiglia statistico dell'intero progetto.
# Vale la pena dichiararlo in presentazione: sono numeri misurati, non
# assunti dalla descrizione del dataset.
EXPECTED_COUNTS = {3: 4274, 4: 1753, 5: 708}
EXPECTED_TOTAL_LESIONS = 6735
EXPECTED_TOTAL_IMAGES = 3924

# Numeri dichiarati nel brief, tenuti per confronto.
BRIEF_COUNTS = {3: 3691, 4: 1817, 5: 521}

# --------------------------------------------------------------- risoluzione
# LA DECISIONE PIU' IMPORTANTE DEL PROGETTO.
#
# Una panoramica inquadra l'intera arcata (~2900 px); una lesione periapicale
# e' di pochi millimetri (~50 px). Ridimensionando la panoramica a 224x224 la
# lesione diventa ~4 px, cioe' MENO DI UN PATCH TOKEN da 16x16: il latente
# estratto alla bbox descrive mandibola generica, non patologia.
#
# Per questo si lavora a TILE su risoluzione nativa. Lanciate
# `python data.py --bbox-stats` PRIMA di addestrare qualsiasi cosa e
# verificate che la bbox mediana copra almeno ~4 token.
USE_TILES = True
TILE_SIZE = 224

# I crop per il pre-training si campionano a CASO, non su una griglia fissa.
# Motivo, misurato sui dati veri: le panoramiche sono 2473x1252 px, quindi una
# griglia con stride 168 produce 15x8 = 120 tile per immagine, cioe' 470.880
# tile per epoca e 3.678 step a batch 128. Un'epoca cosi' e' ingestibile, e
# la maggior parte dei tile sarebbe comunque quasi identica a quella accanto.
# Con 4 crop casuali per immagine si scende a 15.696 campioni e 122 step, e
# in piu' si ottiene piu' varieta' fra un'epoca e l'altra - che per il
# self-supervised e' un vantaggio, non un compromesso.
CROPS_PER_IMAGE = 8

# Quanti crop ricavare da UNA SOLA decodifica dell'immagine.
#
# MISURATO: un item del TileDataset costa ~70 ms, di cui ~52 ms
# (il 75%) sono la sola apertura+decodifica del JPEG - le panoramiche sono
# 2444x1292 e vengono riaperte da zero per ogni singolo crop. Con
# CROPS_PER_IMAGE=8 la stessa immagine veniva decodificata 8 volte per epoca,
# e un'epoca costava 263 s: 8.7 h per 120 epoche, fuori budget con la
# presentazione l'11 settembre.
#
# Decodificando una volta e ricavando CROPS_PER_ITEM crop, il costo per tile
# scende a ~52/K + 18 ms.
#
# RIPORTATO A 1. K=4 dava il caricamento 6.6x piu' veloce, ma con
# un effetto collaterale che non era stato dichiarato: un batch da 128 tile
# conteneva solo 32 immagini DISTINTE. Nel self-supervised la varieta' del
# batch non e' un dettaglio, e quel compromesso poteva essere una delle
# ragioni per cui il pre-training rendeva poco. Con K=1 il batch torna a 128
# immagini diverse, al costo di ~70 ms per tile.
CROPS_PER_ITEM = 1

TILE_STRIDE = 168
TILE_MIN_FOREGROUND = 0.15  # scarta i tile quasi tutti neri (bordi radiografia)

# Braccio di confronto: panoramica intera ridimensionata. Serve per
# DIMOSTRARE empiricamente che non funziona, non per usarlo davvero.
RESIZE_H, RESIZE_W = 224, 224

PATCH_SIZE = 16
# 224 e non di piu', ed e' MISURATO. Il ridimensionamento da crop nativo a
# TILE_SIZE riduce la lesione in token, e il progetto si e' dato
# MIN_TOKENS_PER_LESION = 4.0 come soglia (sez.2 dell'analisi). Lato della
# lesione in token, per grado, al variare della finestra:
#
#     crop   scala   PAI 3   PAI 4   PAI 5   sotto soglia
#      224   1.000     3.6     5.1     7.9        50%
#      288   0.778     2.8     3.9     6.1        76%
#      384   0.583     2.1     3.0     4.6        90%
#
# A 224 la scala e' 1.0: NESSUN ricampionamento. Oltre a massimizzare i
# token, questo mette i crop del downstream ESATTAMENTE nella stessa scala
# dei tile del pre-training, che sono anch'essi 224 px nativi non
# ridimensionati: i due stadi vedono il mondo con lo stesso ingrandimento.
#
# LIMITE DA DICHIARARE: le PAI 3 restano a ~3.6 token, sotto la soglia di 4.
# E' inerente - quelle lesioni sono piccole - e il ritaglio relativo
# dava 4.7 token a tutte al prezzo di cancellare la dimensione, che e' il
# segnale piu' predittivo del dataset.
LESION_CROP_PIXELS = 224
MIN_TOKENS_PER_LESION = 4.0  # soglia di allarme in data.py

# ------------------------------------------------------------------- split
# Split a livello di IMMAGINE, mai di lesione: 6029 lesioni in 3926 immagini
# (~1.54 per immagine) significa che splittare per lesione mette la stessa
# radiografia in train e in test.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ------------------------------------------------------------- backbone ViT
# Piccolo, e non e' un ripiego: LeJEPA ottiene i suoi risultati in-domain su
# Galaxy10 con ConvNeXt-V2 Nano e ResNet-34 (15-22M param) battendo il
# transfer da DINOv3. Su ~4k immagini un ViT-Base overfitta l'anatomia
# globale senza imparare nulla di locale.
VIT_VARIANTS = {
    "vit_tiny":  dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
}
# ViT-Small di default: con una GPU da 12 GB ci sta comodamente, ed e' piu'
# vicino alla taglia dei modelli (15-22M param) con cui LeJEPA ottiene i suoi
# risultati in-domain su Galaxy10. Scendete a vit_tiny se la memoria stringe
# o se volete iterare piu' in fretta.
DEFAULT_VARIANT = "vit_small"

PREDICTOR_DIM = 96      # il predictor e' volutamente stretto (shallow)
PREDICTOR_DEPTH = 4
PREDICTOR_HEADS = 3

# ------------------------------------------------------------ I-JEPA (SSL)
# Masking a blocchi come nel paper: 1 blocco di contesto ampio, piu' blocchi
# target piccoli rimossi dal contesto.
NUM_TARGET_BLOCKS = 4
CONTEXT_SCALE = (0.85, 1.0)
TARGET_SCALE = (0.15, 0.20)
TARGET_ASPECT = (0.75, 1.5)

SSL_EPOCHS = 300
SSL_BATCH_SIZE = 128   # su 12 GB con bf16 ci sta; scendete a 64 se OOM
# LR ridotto Dopo il primo allarme di collasso (intervento 1 del
# CollapseMonitor). Misurato dopo: il collasso NON era instabilita' da LR
# alto - le rappresentazioni miglioravano monotonicamente (rango 1.07 -> 1.99,
# coseno medio 0.97 -> 0.64 rispetto all'init) ma troppo lentamente, e il run
# veniva interrotto all'epoca 15. Dividere il LR per 3 rallenta proprio la
# fuga dal collasso, quindi si resta a meta' strada invece che a 5e-5.
# Learning rate. 1.5e-4 e' il valore originale; i run che hanno prodotto i
# risultati usano 3e-5 (configurazione `completa`, spostamento lento) oppure
# 3e-4 (configurazione `spinto`, spostamento rapido). Si passa da riga di
# comando con --lr, e il valore effettivo finisce nel checkpoint.
SSL_LR = 3e-5
SSL_WEIGHT_DECAY = 0.04
SSL_WARMUP_EPOCHS = 15
# MOMENTUM EMA INIZIALE - la leva anti-collasso piu' importante.
#
# Il valore del paper (0.996) COLLASSA su questo dataset, e non e' una
# sfumatura: cio' che conta e' la costante di tempo 1/(1-tau) rapportata
# alla durata del training. Con 171 passi per epoca:
#     0.996   ->   1.5 epoche   il target assorbe il 49.6% della distanza
#                               dal context in UNA epoca: viene raggiunto,
#                               e i due convergono sulla soluzione costante
#     0.9996  ->  14.6 epoche   ne assorbe il 6.6%: insegue senza raggiungere
#
# Misurato (sonda k-NN a 10 epoche, encoder casuale = 0.7030):
#     0.996  -> 0.4354      0.999   -> 0.7002
#     0.9995 -> 0.7093      0.9996  -> 0.7117
#
# Il valore del paper vale per ImageNet, 1.28M immagini contro le nostre
# 2.746: la stessa tau significa un regime completamente diverso.
#
# ATTENZIONE ALLA RIPRODUCIBILITA': in precedenza qui c'era 0.996 mentre
# TUTTI i run passavano --ema-start 0.9996 da riga di comando. Chi clonava
# il repo e lanciava train_ssl.py senza flag otteneva la configurazione che
# collassa, e non riproduceva nulla dei risultati riportati.
SSL_EMA_START = 0.9996
SSL_EMA_END = 1.0
AMP = True


# ------------------------------------------- monitoraggio del collasso
# E' il modo in cui questo progetto fallisce, e fallisce SILENZIOSAMENTE:
# la loss puo' scendere benissimo mentre tutti gli embedding convergono a
# una costante (predire un target costante e' banale). Loggate sempre
# questi segnali, non solo la loss.
MONITOR_EVERY = 1        # epoche
# Quanti embedding accumulare per la stima del rango: con pochi campioni la
# stima e' limitata dal loro numero, non dalla salute del modello.
MONITOR_SAMPLES = 1024
# Norma massima del gradiente. I-JEPA usa il clipping; senza, un picco
# iniziale puo' mandare la loss a NaN a nottata gia' avviata.
GRAD_CLIP = 3.0
# Ogni 10 e non 20: e' il segnale d'allarme piu' onesto ed e' il criterio del
# gate, ma non e' MAI stato eseguito perche' i run morivano
# all'epoca 15. Costa secondi, tenetelo fitto finche' il gate non e' passato.
# CANCELLO ANTI-SPRECO
#
# Il pre-training va giudicato mentre gira, non dopo. Prima di toccare i
# pesi si misura la sonda k-NN sull'encoder APPENA INIZIALIZZATO: quello e'
# il "modello casuale", ed e' il riferimento da battere. Se dopo
# GATE_EPOCH epoche la sonda non ha mai superato quel valore, il
# pre-training sta PEGGIORANDO le rappresentazioni e si ferma da solo,
# invece di consumare 8 ore per confermarlo.
#
# Il margine evita di fermarsi per rumore: la sonda oscilla di ~0.01 fra
# epoche adiacenti.
# GATE_EPOCH e' l'epoca PRIMA della quale non si giudica: all'inizio la
# sonda e' rumorosa e un run puo' legittimamente peggiorare per poi
# risalire.
GATE_EPOCH = 15
GATE_MARGINE = 0.01

# Quante sonde consecutive sotto il riferimento bastano per fermarsi.
# Due: una sola puo' essere rumore, due di fila sono una tendenza.
GATE_SONDE_SOTTO = 2

# Scorciatoia per il crollo netto: una singola sonda sotto il riferimento
# di piu' di questo numero di margini e' gia' una risposta, non rumore.
GATE_CROLLO = 5

# Ogni quante epoche stampare il RESOCONTO: non una riga in piu', ma la
# tabella di tutte le sonde fatte finora con i delta rispetto all'encoder
# casuale. Serve a rispondere a colpo d'occhio all'unica domanda che conta
# durante un run lungo - "sta migliorando o no?" - senza dover rileggere
# centinaia di righe di log.
# Profondita' concatenate per il downstream. L'ultimo blocco e' il piu'
# COMPRESSO: con una lettura lineare - che e' cio' che fa la testa - un
# blocco intermedio rende molto di piu'.
LAYERS_DOWNSTREAM = [2, 7, 11]

# Quante lesioni di train usare per la sonda DOWNSTREAM durante il
# pre-training. Tutte e 4719 costerebbero ~2.1 GB di token a ogni controllo;
# un sottoinsieme fisso basta per seguire una tendenza, e resta fisso
# proprio perche' i valori vadano confrontati fra loro.
DOWNSTREAM_SONDA_TRAIN = 2500

RESOCONTO_OGNI = 10

KNN_PROBE_EVERY = 5
KNN_K = 20
KNN_SUBSET = 1500

# Da quale epoca la guardia anti-collasso puo' interrompere il run, e per
# quante epoche consecutive i segnali devono restare degradati.
#
# MISURATO, ed e' la ragione per cui questi due valori esistono:
# con min_epoch = SSL_WARMUP_EPOCHS (15) la guardia scattava SEMPRE
# all'epoca 15 esatta, cioe' nell'istante in cui il warmup finisce e il LR
# raggiunge il massimo - il modello moriva prima di un solo step a LR pieno.
# Il criterio dichiarato dal progetto e' invece a ~100 epoche (vedi
# ANALISI_PROGETTO_8.md sez.5 e la docstring di train_ssl.run_probe):
# "se dopo 100 epoche il k-NN resta al livello della maggioritaria, cambiate
# qualcosa". La guardia ora e' allineata a quel criterio.
#
# Nota sulla scala: un ViT APPENA INIZIALIZZATO misura rango 1.07/279 (0.4%)
# e coseno medio 0.97. Il punto di partenza e' gia' sotto la soglia del 15%:
# pretendere di superarla entro l'epoca 15 non era realistico.
COLLAPSE_MIN_EPOCH = 100
COLLAPSE_PATIENCE = 5

# Soglia sul rapporto rango_misurato / rango_di_riferimento sotto la quale la
# rappresentazione e' considerata collassata.
#
# ANCORATA A MISURE, non scelta a priori. Sulle 384 dimensioni del ViT-Small,
# con 1024 campioni il riferimento isotropo e' 279.8, e:
#     ViT appena inizializzato    1.07/280 = 0.4%
#     run migliore ottenuto      13   /280 = 4.6%
# Il valore precedente era 0.15, cioe' 42 direzioni: irraggiungibile in
# questo dominio, e infatti la guardia ha interrotto all'epoca 100 il run che
# aveva il miglior k-NN della serie (macro-F1 0.4534 contro 0.4095 del
# precedente). 0.02 corrisponde a ~5.6 direzioni: sta sopra il rumore
# dell'inizializzazione e sotto un pre-training che sta funzionando.
COLLAPSE_RANK_FLOOR = 0.02
# Il riferimento per il k-NN probe viene MISURATO sulle etichette di
# validazione dentro train_ssl.run_probe(), non fissato qui: dipende dallo
# split e i numeri del brief non coincidono con il dataset reale.

# --------------------------------------------------------------- downstream
# Dopo il pre-training si estraggono i latenti UNA VOLTA e si cachano.
# Da quel momento ogni esperimento sullo sbilanciamento gira in secondi,
# anche su CPU: l'ablation diventa praticamente gratuito.
HEAD_EPOCHS = 100
HEAD_BATCH_SIZE = 128
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-4
ATTN_POOL_HEADS = 4

# Come i token dentro la bbox diventano un vettore solo. Tre ipotesi
# diverse su dove sta il collo di bottiglia, vedi network.py:
#   attn   query appresa ma FISSA, softmax denso su tutti i token (attuale)
#   gated  punteggio non lineare per token, stile MIL (Ilse et al. 2018)
#   topk   solo i TOP_K token piu' forti, il resto scartato
POOL_TYPES = ["attn", "gated", "topk"]
TOP_K = 8      # bbox mediane: 9 / 16 / 64 token per PAI 3 / 4 / 5

# Metodi per lo sbilanciamento da confrontare. 'balanced_tokens' e' la
# novita' proposta (vedi imbalance.py).
IMBALANCE_METHODS = [
    "none",             # cross-entropy semplice
    "class_weighted",   # CE pesata
    "focal",            # focal loss
    "oversample",       # oversampling della minoritaria
    "balanced_tokens",  # NOVITA' proposta
]

# Controlli, non baseline: non entrano nella griglia dell'obiettivo 4
# perche' non sono metodi che qualcuno userebbe: servono ad ATTRIBUIRE il
# risultato della novita'. `random_tokens` toglie il ribilanciamento e
# lascia l'augmentation a budget identico; se le due misure coincidono, la
# novita' non ribilancia niente e vince per un altro motivo.
METODI_CONTROLLO = ["random_tokens"]
FOCAL_GAMMA = 2.0

# Teste: piatta (come da brief) vs ordinale (piu' appropriata al PAI).
HEAD_TYPES = ["flat", "ordinal"]

# Teste che predicono soglie cumulative invece di logit per classe. Serve
# una funzione sola, chiamata da tutti: la loss e la valutazione decidevano
# con `head_type == "ordinal"`, quindi qualunque testa ordinale con un nome
# diverso sarebbe stata addestrata con la cross-entropy e valutata come se
# fosse piatta - in silenzio, senza errori, con numeri plausibili e sbagliati.
HEAD_TYPES_ESTESE = ["flat", "ordinal", "norm", "norm_ord", "mlp", "mlp_ord",
                     "mil"]


def e_ordinale(head_type):
    """True se la testa produce K-1 soglie cumulative invece di K logit."""
    return head_type in ("ordinal", "norm_ord", "mlp_ord")

N_SEEDS = 5   # lo sbilanciamento e' 7:1, non 100:1: i margini sono stretti
              # e servono intervalli di confidenza, non un singolo run.

for _d in (CKPT_DIR, FIG_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)
