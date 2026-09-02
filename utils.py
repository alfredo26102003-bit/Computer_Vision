"""
PIPELINE — seed, checkpoint, monitor del collasso, freno hardware.

Utils - seed, checkpoint, monitoraggio del collasso, k-NN probe, plot.

Sezione "Utils" della struttura richiesta dal corso.
"""

import json
import os
import random

import numpy as np
import time

import torch

from globals import CKPT_DIR, FIG_DIR, KNN_K, SEED


# --------------------------------------------------------- riproducibilita'
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# -------------------------------------------------------------- checkpoint
# Il pre-training SSL supera facilmente le 12 h di una sessione Kaggle
# (300 epoche ~ 6 h, 600 ~ 12.5 h). Il resume non e' opzionale.
def save_checkpoint(state: dict, name: str) -> str:
    """
    Salvataggio ATOMICO: prima su file temporaneo, poi rinomina.

    Perche' non si scrive diretti sul file finale: il checkpoint pesa ~350 MB
    e la scrittura dura secondi. Se la macchina si spegne in quella finestra,
    il file resta troncato e torch.load fallisce con
    "PytorchStreamReader failed reading zip archive" - e si perde ANCHE il
    checkpoint dell'epoca precedente, che era gia' stato sovrascritto.
    E' successo: un arresto improvviso ha distrutto un run a 22
    epoche che avrebbe potuto riprendere dall'epoca 21.

    Con la rinomina (atomica sullo stesso filesystem) il file finale o e'
    quello vecchio integro o quello nuovo completo, mai una via di mezzo.

    MA LA RINOMINA DA SOLA NON BASTA, e L'abbiamo scoperto a spese
    di un checkpoint. torch.save scrive nella cache del sistema operativo:
    os.replace scambia la voce di directory SUBITO, mentre i blocchi di dati
    possono essere ancora in memoria. Se manca la corrente in quella
    finestra, la directory punta a un file il cui contenuto non e' mai
    arrivato sul disco - e torch.load fallisce esattamente come nel caso che
    la rinomina doveva prevenire.
    L'os.fsync forza la scrittura fisica PRIMA dello scambio, e chiude la
    finestra. Costa qualche decimo di secondo su 350 MB, contro il rischio
    di perdere l'intero run: su questa macchina, che si e' spenta da sola
    nove volte in quattro giorni, non e' un compromesso discutibile.
    """
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())   # i byte sono sul disco, non solo nella cache
    os.replace(tmp, path)      # atomica: sostituisce anche se path esiste
    return path


def load_checkpoint(name: str, map_location=None):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def leggi_righe_risultati(path):
    """
    Le righe di un file di risultati, nei due formati che esistono.

    run_grid avvolge le righe in un dizionario con la
    provenienza dei latenti: {"latenti": ..., "quando": ..., "righe": [...]}.
    I file prodotti prima sono una lista nuda. Entrambi vanno letti, perche'
    i vecchi restano su disco come prova di com'erano.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["righe"] if isinstance(d, dict) and "righe" in d else d


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ==========================================================================
# MONITORAGGIO DEL COLLASSO
# ==========================================================================
# Il collasso della rappresentazione e' il modo in cui questo progetto
# fallisce, ed e' silenzioso: la loss I-JEPA scende regolarmente mentre tutti
# gli embedding convergono a una costante, perche' predire un target costante
# e' banale. Se ve ne accorgete il 5 settembre, il progetto e' finito.
#
# Queste tre funzioni vanno chiamate a ogni epoca. Costano nulla.
# ==========================================================================
@torch.no_grad()
def embedding_std(embeddings: torch.Tensor) -> float:
    """
    Deviazione standard media per dimensione, su embedding L2-normalizzati.

    Collasso completo -> 0. Riferimento sano: per embedding normalizzati di
    dimensione d, una distribuzione isotropa da' circa 1/sqrt(d).
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    return z.std(dim=0).mean().item()


@torch.no_grad()
def effective_rank(embeddings: torch.Tensor) -> float:
    """
    Rango effettivo via participation ratio: (sum l)^2 / sum l^2 sugli
    autovalori della matrice dei momenti secondi.

    Dice quante direzioni dello spazio sono realmente usate.
    Collasso -> 1. Isotropia perfetta su d dimensioni -> d.

    DUE DETTAGLI DI IMPLEMENTAZIONE CHE CONTANO, e su cui e' facile
    sbagliare:

    1. Si L2-normalizza prima. Senza, la metrica confonde "embedding tutti
       nella stessa direzione ma di norma diversa" con una distribuzione
       ricca: la varianza radiale gonfia il rango.

    2. NON si centra sulla media. Usando la covarianza centrata, embedding
       tutti identici piu' un epsilon di rumore danno rango ~d invece di 1,
       perche' dopo il centraggio resta solo il rumore, che e' isotropo.
       Il collasso costante - cioe' la modalita' di collasso PIU' CLASSICA -
       passerebbe inosservato. Con la matrice dei momenti secondi, vettori
       identici danno una matrice di rango 1 e il participation ratio va a 1
       come deve.

    Verificato su quattro casi noti nel blocco __main__ di questo file.
    """
    z = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    second_moment = (z.T @ z) / max(z.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(second_moment).clamp(min=0)
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    return (s1 ** 2 / s2).item() if s2 > 0 else 1.0


@torch.no_grad()
def effective_rank_centered(embeddings: torch.Tensor) -> float:
    """
    Rango effettivo DOPO aver tolto la media. Da leggere insieme, non al
    posto, a effective_rank().

    PERCHE' SERVIVANO ENTRAMBI. La versione non centrata e' corretta per
    quello che dichiara - vede il collasso costante - ma non distingue
    "tutti gli embedding identici" da "embedding diversi che condividono
    una grande componente media". Negli embedding di un ViT la seconda e'
    la norma: misurato Sull'encoder CASUALE, quello che ottiene
    k-NN macro-F1 0.7030 ed e' il miglior estrattore che abbiamo, la norma
    del vettore medio e' il 98.2% della norma media dei vettori. Risultato:
    rango non centrato 1.07 su 384, std 0.008 contro 0.051 "sano".

    Cioe' il monitor dichiarava COLLASSO TOTALE sull'encoder che funziona
    meglio di tutti. Ogni allarme di collasso letto finora andava riletto
    con questa correzione: non misurava la salute del modello.

    Il rango centrato misura la diversita' RESIDUA, che e' quella che porta
    informazione. Sullo stesso encoder casuale vale 1.68: basso davvero, ma
    non e' una patologia - il segnale del problema e' quasi
    unidimensionale (estensione della radiotrasparenza), ed e' proprio quel
    poco che basta a fare 0.70 di macro-F1.
    """
    z = embeddings.float()
    z = z - z.mean(dim=0, keepdim=True)
    z = torch.nn.functional.normalize(z, dim=-1)
    second_moment = (z.T @ z) / max(z.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(second_moment.double()).clamp(min=0)
    s1, s2 = eigvals.sum(), (eigvals ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > 0 else 1.0


def rank_reference(n_samples: int, dim: int, seed: int = 0) -> float:
    """
    Rango effettivo di embedding perfettamente sani, con QUELLO stesso numero
    di campioni e dimensioni.

    Serve perche' il rango effettivo e' limitato dal numero di campioni, non
    solo dal collasso: con 128 campioni in 384 dimensioni, embedding isotropi
    danno ~97, non 384. Senza questo riferimento un valore di 97 sembra un
    collasso in corso quando invece e' il massimo ottenibile.

    Il rapporto misurato/riferimento e' la quantita' leggibile: ~1.0 sano,
    vicino a 0 collassato.
    """
    g = torch.Generator().manual_seed(seed)
    return effective_rank(torch.randn(n_samples, dim, generator=g))


@torch.no_grad()
def knn_probe(train_feats, train_labels, test_feats, test_labels, k=KNN_K):
    """
    Probe k-NN sulle feature congelate - il segnale d'allarme piu' onesto.

    Ogni ~20 epoche: se dopo 100 epoche resta al livello della classe
    maggioritaria (0.612 per PAI 3), il pre-training non sta imparando
    niente di utile e va cambiato qualcosa PRIMA di aver bruciato giorni.

    Ritorna: (accuracy, macro-F1). Guardate la macro-F1: con 61% di PAI 3
    l'accuracy da' sola un'impressione ottimistica.
    """
    tr = torch.nn.functional.normalize(train_feats.float(), dim=-1)
    te = torch.nn.functional.normalize(test_feats.float(), dim=-1)
    sims = te @ tr.T
    idx = sims.topk(min(k, tr.shape[0]), dim=-1).indices
    votes = train_labels[idx]

    n_cls = int(max(train_labels.max().item(), test_labels.max().item())) + 1
    onehot = torch.zeros(votes.shape[0], n_cls)
    for c in range(n_cls):
        onehot[:, c] = (votes == c).sum(dim=-1).float()
    pred = onehot.argmax(dim=-1)

    acc = (pred == test_labels).float().mean().item()

    f1s = []
    for c in range(n_cls):
        tp = ((pred == c) & (test_labels == c)).sum().item()
        fp = ((pred == c) & (test_labels != c)).sum().item()
        fn = ((pred != c) & (test_labels == c)).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    return acc, float(np.mean(f1s))


class CollapseMonitor:
    """
    Tiene la storia dei segnali di collasso e avvisa quando degradano.

    Uso nel loop di training:
        mon = CollapseMonitor()
        ...
        mon.update(epoch, loss, embeddings)
        if mon.is_collapsing():
            print("Fermarsi e cambiare qualcosa.")
    """

    def __init__(self, std_floor=1e-3, rank_ratio_floor=None, min_epoch=None):
        self.history = []
        self.iniziale = None

        # ATTENZIONE alla std: rileva SOLO il collasso costante, cioe' tutti
        # gli embedding identici. Misurato su casi noti a 384 dimensioni
        # (riferimento sano 1/sqrt(384) = 0.0510):
        #     isotropo      std 0.0510    rango 280
        #     rango 60      std 0.0508    rango  50
        #     rango  6      std 0.0488    rango   6   <- il 96% del sano!
        #     rango  2      std 0.0456    rango   2
        #     costante      std 0.0000    rango   1
        # Fra rango 2 e rango 384 la std si muove dell'11%: chi monitora la
        # varianza crede che vada tutto bene mentre il modello usa 6
        # direzioni su 384. Si tiene come rete di sicurezza per il collasso
        # totale, non come diagnostico: l'unico segnale che discrimina e' il
        # rango.
        self.std_floor = std_floor

        # Soglia sul RAPPORTO rango misurato / rango di riferimento, non sul
        # valore assoluto: vedi rank_reference().
        #
        # Era 0.15, cioe' 42 direzioni su 280, e non e' un valore
        # raggiungibile in questo dominio: misurato, un ViT appena
        # inizializzato sta a 1.07/280 (0.4%) e il run MIGLIORE ottenuto
        # arriva a 13/280 (4.6%). Con 0.15 la guardia ha interrotto proprio
        # il run che stava dando il miglior k-NN della serie.
        # 0.02 (5.6/280) sta sopra l'inizializzazione casuale e sotto un run
        # sano: separa il collasso vero dall'apprendimento lento.
        from globals import COLLAPSE_RANK_FLOOR
        self.rank_ratio_floor = (COLLAPSE_RANK_FLOOR if rank_ratio_floor is None
                                 else rank_ratio_floor)
        # Guardia di warmup: all'inizio del training gli embedding sono
        # legittimamente quasi identici - la rete non ha ancora imparato a
        # distinguere niente, quindi il rango effettivo parte basso. Senza
        # questa guardia is_collapsing() scatta all'epoca 2 e interrompe
        # ogni run prima che possa convergere.
        #
        # Il valore era SSL_WARMUP_EPOCHS (15), cioe' esattamente la fine del
        # warmup: la guardia si armava nell'istante in cui il LR arrivava al
        # massimo e uccideva ogni run all'epoca 15 senza un solo step a LR
        # pieno. Ora usa COLLAPSE_MIN_EPOCH, allineato al criterio dichiarato
        # dal progetto (~100 epoche). Vedi il commento in globals.py.
        from globals import COLLAPSE_MIN_EPOCH
        self.min_epoch = COLLAPSE_MIN_EPOCH if min_epoch is None else min_epoch

    def update(self, epoch, loss, embeddings, knn=None):
        n, d = embeddings.shape[0], embeddings.shape[-1]
        std = embedding_std(embeddings)
        rank = effective_rank(embeddings)
        rank_c = effective_rank_centered(embeddings)
        rif = rank_reference(n, d)
        ratio = rank / rif if rif > 0 else 0.0

        # Riferimento onesto: il modello all'inizio del run, non un gaussiano
        # isotropo. L'isotropo e' irraggiungibile per un ViT (vedi
        # effective_rank_centered), quindi "% del sano" verso l'isotropo dava
        # sempre ~0% e non distingueva nulla.
        if self.iniziale is None:
            self.iniziale = {"eff_rank": rank, "eff_rank_c": rank_c, "std": std}

        entry = {"epoch": epoch, "loss": float(loss), "std": std,
                 "eff_rank": rank, "eff_rank_centrato": rank_c,
                 "rank_ref": rif, "rank_ratio": ratio,
                 "n": n, "dim": d}
        if knn is not None:
            # `knn` e' il pannello diagnostico completo (dict): si
            # riversano tutte le voci numeriche, cosi' aggiungerne una in
            # train_ssl.py non richiede di toccare anche questo file.
            entry.update({k: v for k, v in knn.items()
                          if isinstance(v, (int, float))})
        self.history.append(entry)

        # COLLASSO COSTANTE: l'unica cosa che la std sa davvero rilevare.
        flag = "   <-- COLLASSO COSTANTE" if std < self.std_floor else ""
        vs = rank_c / self.iniziale["eff_rank_c"] if self.iniziale["eff_rank_c"] else 1.0
        print(f"  [monitor] ep{epoch:03d} loss={loss:.4f} std={std:.5f} "
              f"rango={rank:.2f} centrato={rank_c:.2f} "
              f"({vs:.2f}x l'iniziale){flag}")
        return entry

    def is_collapsing(self, patience=None):
        """
        True solo se i segnali restano degradati per `patience` epoche
        consecutive E siamo oltre il warmup. Le due condizioni servono
        entrambe: la prima evita i falsi allarmi da rumore di una singola
        epoca, la seconda evita di scambiare l'inizializzazione per collasso.

        Terza condizione, aggiunta dopo i run: se i segnali stanno
        MIGLIORANDO non e' collasso, e' apprendimento lento. Un run che sale
        da rango 1 a rango 4 sta uscendo dal collasso, non entrandoci, e
        interromperlo butta via l'unica cosa che stava funzionando.
        """
        if patience is None:
            from globals import COLLAPSE_PATIENCE
            patience = COLLAPSE_PATIENCE
        if len(self.history) < patience:
            return False
        if self.history[-1]["epoch"] < self.min_epoch:
            return False
        recent = self.history[-patience:]

        # SOLO il collasso costante. Il criterio sul rapporto verso un
        # gaussiano isotropo e' stato tolto: misurato, l'encoder
        # casuale - il miglior estrattore che abbiamo, k-NN 0.7030 - sta a
        # 1.07/280 cioe' 0.4%, ben sotto qualunque soglia sensata. Quel
        # criterio era vero SEMPRE, quindi avrebbe interrotto anche i run
        # sani appena passato COLLAPSE_MIN_EPOCH.
        #
        # Il giudizio su "il pre-training sta aiutando o no" spetta al
        # cancello sulla sonda k-NN in train_ssl.py, che confronta con
        # l'encoder casuale misurato con lo stesso protocollo.
        degradati = all(e["std"] < self.std_floor for e in recent)
        if not degradati:
            return False

        # I segnali sono bassi, ma stanno salendo? Allora il modello sta
        # uscendo dal collasso e va lasciato lavorare. Si confronta la prima
        # meta' della finestra con la seconda: serve un miglioramento netto
        # (>5%) di almeno uno dei due segnali, non il rumore di un'epoca.
        meta = max(len(recent) // 2, 1)
        for chiave in ("eff_rank_centrato", "std"):
            prima = sum(e[chiave] for e in recent[:meta]) / meta
            dopo = sum(e[chiave] for e in recent[meta:]) / max(len(recent) - meta, 1)
            if dopo > prima * 1.05:
                return False
        return True

    def save(self, path):
        save_json(self.history, path)

    def plot(self, name="collapse_monitor"):
        import matplotlib.pyplot as plt

        if not self.history:
            return None
        ep = [e["epoch"] for e in self.history]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        axes[0].plot(ep, [e["loss"] for e in self.history], color="#1f77b4")
        axes[0].set_title("Loss I-JEPA")
        axes[1].plot(ep, [e["std"] for e in self.history], color="#d62728")
        axes[1].axhline(self.std_floor, ls="--", c="gray", lw=1)
        axes[1].set_title("Std degli embedding")
        axes[2].plot(ep, [100 * e["rank_ratio"] for e in self.history], color="#2ca02c")
        axes[2].axhline(100 * self.rank_ratio_floor, ls="--", c="gray", lw=1)
        axes[2].set_ylim(0, 110)
        axes[2].set_title("Rango effettivo (% del sano)")
        for a in axes:
            a.set_xlabel("epoca")
            a.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return path


if __name__ == "__main__":
    # Verifica del monitoraggio del collasso su casi con esito noto.
    # Lanciare con: python utils.py
    print("=== effective_rank / embedding_std su casi noti ===")
    D, N = 192, 512
    torch.manual_seed(0)
    casi = {
        "isotropo (sano)":        torch.randn(N, D),
        "costante (collasso)":    torch.randn(1, D).repeat(N, 1) + 1e-6 * torch.randn(N, D),
        "rango 1":                torch.randn(N, 1) @ torch.randn(1, D),
        "rango 8":                torch.randn(N, 8) @ torch.randn(8, D),
    }
    attesi = {"isotropo (sano)": "~D", "costante (collasso)": "~1",
              "rango 1": "~1", "rango 8": "~8"}
    for nome, e in casi.items():
        print(f"  {nome:22s} std={embedding_std(e):.6f}  "
              f"eff_rank={effective_rank(e):7.2f} / {D}   atteso {attesi[nome]}")

    print("\n=== CollapseMonitor: loss che scende, embedding collassati ===")
    # min_epoch=0 per testare la logica di rilevamento: con il default la
    # guardia di warmup sopprime l'allarme nelle prime epoche, ed e' giusto
    # cosi' in training ma qui maschererebbe il test.
    # Servono almeno COLLAPSE_PATIENCE epoche perche' la guardia possa
    # esprimersi: con meno campioni is_collapsing() si astiene per design.
    from globals import COLLAPSE_PATIENCE

    n_ep = COLLAPSE_PATIENCE + 1
    mon = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        mon.update(ep, 0.5 / (ep + 1), casi["costante (collasso)"])
    print(f"  is_collapsing() = {mon.is_collapsing()}   (atteso True)")
    print("  ^ la loss scendeva regolarmente: e' il fallimento silenzioso.")

    mon2 = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        mon2.update(ep, 0.5 / (ep + 1), casi["isotropo (sano)"])
    print(f"  su embedding sani: is_collapsing() = {mon2.is_collapsing()}   (atteso False)")

    # Terzo caso, quello che ha causato le interruzioni premature:
    # segnali bassi ma in MIGLIORAMENTO. Non e' collasso, e' apprendimento
    # lento, e la guardia non deve interrompere.
    mon3 = CollapseMonitor(min_epoch=0)
    for ep in range(n_ep):
        k = 1 + ep                     # rango che cresce 1, 2, 3, ...
        e = torch.randn(N, k) @ torch.randn(k, D)
        mon3.update(ep, 0.5 / (ep + 1), e)
    print(f"  su segnali BASSI ma in salita: is_collapsing() = {mon3.is_collapsing()}   (atteso False)")


# ==========================================================================
# Freno: ridurre il consumo medio quando l'hardware non regge
# ==========================================================================
class Freno:
    """
    Limita il CICLO DI LAVORO della GPU a una percentuale del tempo.

    `carico=80` significa: la GPU lavora l'80% del tempo e riposa il 20%,
    quindi il consumo medio scende all'80% e il run dura 1/0.8 = 1.25 volte.

    PERCHE' NON SI PUO' FARE MEGLIO. La strada pulita sarebbe abbassare il
    limite di potenza della scheda, ma su questo portatile e' bloccato dal
    produttore: `nvidia-smi -pl` risponde "not supported in current scope".
    La GPU gira a 175 W contro un limite PREDEFINITO di 60 W, su un
    alimentatore da 330 W che deve reggere anche un i9-13980HX. Sei
    spegnimenti improvvisi (Kernel-Power 41) in quattro giorni, tutti sotto
    carico sostenuto.

    Non potendo limitare la potenza istantanea si limita il tempo in cui
    viene assorbita. Non protegge dai picchi di accensione, ma toglie il
    carico sostenuto, che e' quello che fa scattare le protezioni.

    Costo, per i valori utili:
        carico 100  ->  1.00x   nessun freno
        carico  90  ->  1.11x
        carico  80  ->  1.25x
        carico  70  ->  1.43x
        carico  50  ->  2.00x

        freno = Freno(carico=80)
        for batch in loader:
            t = time.perf_counter()
            ... passo di training ...
            freno.pausa(t)
    """

    def __init__(self, carico=100):
        if not 1 <= carico <= 100:
            raise ValueError(f"carico deve stare fra 1 e 100, ricevuto {carico}")
        self.carico = float(carico)
        # quota di pausa rispetto al tempo di calcolo: con carico c il
        # rapporto pausa/calcolo e' (100 - c) / c
        self.frazione = (100.0 - self.carico) / self.carico
        self._t0 = None
        self.pausa_totale = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._t0 is not None:
            self.pausa(self._t0)
        return False

    def pausa(self, t0):
        """Pausa proporzionale al tempo trascorso da `t0`.

        Forma esplicita, da chiamare a fine passo. Si preferisce al gestore
        di contesto dentro cicli gia' profondi: avvolgere il corpo di un
        `for` in un `with` aggiunge un livello di indentazione a decine di
        righe e rende il diff illeggibile."""
        if self.frazione <= 0:
            return 0.0
        p = (time.perf_counter() - t0) * self.frazione
        time.sleep(p)
        self.pausa_totale += p
        return p

    def __repr__(self):
        return (f"Freno(carico={self.carico:.0f}%, "
                f"run {1/(self.carico/100):.2f}x piu' lungo)")


# ==========================================================================
# Termostato: pieno regime, ma con pause quando la GPU scotta
# ==========================================================================
def temperatura_gpu():
    """Temperatura della GPU in gradi, o None se nvidia-smi non risponde.

    Il timeout non e' cosmetico: se nvidia-smi si impianta, un controllo
    ogni pochi secondi bloccherebbe il training invece di proteggerlo."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Termostato:
    """
    Lascia lavorare la GPU a PIENO REGIME e la ferma solo quando scotta.

    DIFFERENZA DAL FRENO A CICLO FISSO. Freno(carico=80) toglie il 20% del
    tempo sempre, anche quando la scheda e' fredda: paga il 25% di durata in
    piu' a prescindere. Il termostato non paga niente finche' la temperatura
    resta sotto soglia, e interviene solo quando serve. Su un carico che
    scalda a intermittenza - come una griglia di teste, dove fra un
    addestramento e l'altro la GPU respira - la differenza e' grossa.

    ISTERESI, ed e' il punto che rende il meccanismo utile invece che
    fastidioso: si ferma a `soglia` ma riparte solo a `riparti`, piu' bassa.
    Con una soglia sola il sistema oscillerebbe attorno a quel valore
    facendo micro-pause continue, senza mai far scendere davvero la
    temperatura.

    VALORI, misurati su questa macchina:
        88 C   griglia a pieno regime -> spegnimento improvviso
        71-72  con freno all'80%, stabile per ore
        60-71  a riposo
    Da cui soglia 80 e ripartenza 72: si interviene prima della fascia in
    cui la macchina e' morta, e si torna al regime che ha retto quattro ore.

    ATTENZIONE: la temperatura NON e' la causa provata degli spegnimenti.
    La diagnosi piu' probabile e' la batteria al 74% che non tampona i
    picchi di corrente, e quelli il termostato non li tocca. Riduce il
    carico termico sostenuto, che e' un fattore correlato, non il
    meccanismo.

        term = Termostato(soglia=80, riparti=72)
        for batch in loader:
            ... passo ...
            term.controlla()
    """

    def __init__(self, soglia=80, riparti=72, ogni=20.0, attesa_max=300):
        if riparti >= soglia:
            raise ValueError("`riparti` deve essere sotto `soglia`: senza "
                             "isteresi il termostato oscilla")
        self.soglia, self.riparti = soglia, riparti
        self.ogni = ogni                  # secondi fra un controllo e l'altro
        self.attesa_max = attesa_max      # non bloccarsi all'infinito
        self._ultimo = 0.0
        self.pause = 0
        self.pausa_totale = 0.0
        self.t_max = 0

    def controlla(self, verbose=True):
        """Da chiamare a fine passo. Legge la temperatura al massimo una
        volta ogni `ogni` secondi: leggerla a ogni passo costerebbe piu' del
        passo stesso."""
        ora = time.time()
        if ora - self._ultimo < self.ogni:
            return 0.0
        self._ultimo = ora

        t = temperatura_gpu()
        if t is None:
            return 0.0
        self.t_max = max(self.t_max, t)
        if t < self.soglia:
            return 0.0

        t0 = time.time()
        if verbose:
            print(f"  [termostato] {t} C sopra la soglia di {self.soglia}: "
                  f"pausa fino a {self.riparti} C", flush=True)
        while time.time() - t0 < self.attesa_max:
            time.sleep(5)
            t = temperatura_gpu()
            if t is None or t <= self.riparti:
                break
        att = time.time() - t0
        self.pause += 1
        self.pausa_totale += att
        self._ultimo = time.time()
        if verbose:
            print(f"  [termostato] ripreso a {t} C dopo {att:.0f}s "
                  f"(pausa {self.pause}, totale {self.pausa_totale/60:.1f} min)",
                  flush=True)
        return att

    def __repr__(self):
        return (f"Termostato(pausa sopra {self.soglia} C, riprende a "
                f"{self.riparti} C, controllo ogni {self.ogni:.0f}s)")
