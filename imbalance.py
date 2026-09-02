"""
PIPELINE — baseline per lo sbilanciamento e la novita' balanced_token_sampling.

Imbalance - metodi per lo sbilanciamento di classe, baseline e la novita'.

Copre l'obiettivo 3 del brief: "formulate and integrate an original
algorithmic novelty specifically designed to combat class imbalance in the
latent space".

--------------------------------------------------------------------------
LA NOVITA' E' VOSTRA. Qui trovate:
  - le baseline complete e funzionanti (CE, pesata, focal, oversampling)
  - la novita' proposta, `balanced_token_sampling`

Lo sbilanciamento e' 7:1, non 100:1. E' serio ma non estremo, quindi le
baseline standard funzioneranno decentemente e la vostra novita' deve
batterle su un margine ristretto. Progettate la valutazione per rilevare
differenze piccole: N_SEEDS seed e intervalli di confidenza, non un run.
--------------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F

from globals import FOCAL_GAMMA, NUM_CLASSES, e_ordinale


# ==========================================================================
# Pesi di classe
# ==========================================================================
def class_counts(labels, num_classes=NUM_CLASSES):
    return torch.bincount(labels.long(), minlength=num_classes).float()


def inverse_frequency_weights(labels, num_classes=NUM_CLASSES):
    c = class_counts(labels, num_classes).clamp(min=1)
    w = c.sum() / (num_classes * c)
    return w / w.mean()


# ==========================================================================
# Loss
# ==========================================================================
def focal_loss(logits, targets, gamma=FOCAL_GAMMA, weight=None):
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets[:, None]).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def ordinal_loss(cum_logits, targets, num_classes=NUM_CLASSES, weight=None,
                 focal=False, gamma=FOCAL_GAMMA):
    """
    BCE sui logit cumulativi, per la testa ordinale (stile CORAL).

    Penalizza correttamente gli errori a due gradi di distanza: e' il motivo
    per cui la testa ordinale ha senso su una scala come il PAI.

    `focal=True` applica la modulazione focale a ciascuna delle K-1 decisioni
    binarie cumulative, pesando (1 - p_t)^gamma come nella focal loss
    ordinaria. Serve perche' senza, `focal` e `class_weighted` con la testa
    ordinale finivano ENTRAMBI qui con gli stessi pesi e producevano numeri
    identici cifra per cifra: due righe dell'ablation erano lo stesso
    esperimento presentato come due confronti distinti.
    """
    lv = torch.arange(num_classes - 1, device=targets.device)[None, :]
    t = (targets[:, None] > lv).float()
    loss = F.binary_cross_entropy_with_logits(cum_logits, t, reduction="none")

    if focal:
        # p_t e' la probabilita' assegnata alla classe CORRETTA di ciascuna
        # decisione binaria: alta sugli esempi facili, che vengono cosi'
        # attenuati, bassa su quelli difficili, che restano pesanti.
        p = torch.sigmoid(cum_logits)
        p_t = p * t + (1 - p) * (1 - t)
        loss = loss * (1 - p_t).pow(gamma)

    if weight is not None:
        loss = loss * weight[targets][:, None]
    return loss.mean()


# ==========================================================================
# Sampler
# ==========================================================================
def balanced_sampler_weights(labels, num_classes=NUM_CLASSES):
    """Pesi per WeightedRandomSampler: oversampling della minoritaria."""
    c = class_counts(labels, num_classes).clamp(min=1)
    return (1.0 / c)[labels.long()]


# ==========================================================================
# NOVITA' PROPOSTA - Balanced token sampling
# ==========================================================================
def n_views_per_class(counts, alpha=0.5, num_classes=NUM_CLASSES):
    """
    Quante viste per classe: n_c = ceil((max_count / count_c) ** alpha).

    alpha e' il grado di aggressivita' del ribilanciamento ed e' il parametro
    da mettere a sweep nell'ablation: 0 = nessun ribilanciamento (una vista
    per tutti), 1 = pareggio completo delle frequenze.
    """
    c = counts.float().clamp(min=1)
    v = (c.max() / c) ** alpha
    return v.ceil().long()


def balanced_token_sampling(tokens, token_mask, labels, counts,
                            num_classes=NUM_CLASSES, generator=None,
                            alpha=0.5, p_min=0.6, p_max=1.0, min_tokens=4,
                            attn_weights=None):
    """
    Balanced token sampling - la novita' metodologica del progetto.

    Il brief nomina esplicitamente le "balanced token-sampling strategies"
    tra gli esempi, ed e' la piu' elegante delle tre opzioni discusse
    nell'analisi.

    L'IDEA
    L'attention pooling aggrega i token che cadono dentro la bbox della
    lesione. Una bbox ne contiene diversi (verificatelo: dovrebbero essere
    ~10-36 alla risoluzione corretta). Invece di aggregarli sempre tutti,
    per le classi minoritarie si campionano SOTTOINSIEMI DIVERSI di token
    dalla stessa lesione, e ogni sottoinsieme diventa un'istanza di training
    distinta.

    E' oversampling nello spazio dei token:
      - non duplica immagini (a differenza dell'oversampling classico, che
        ripresenta lo stesso vettore identico e invita all'overfitting)
      - non genera pixel sintetici (a differenza di SMOTE, che interpola
        in un latente dove l'interpolazione puo' non avere senso anatomico)
      - ogni istanza e' una vista GENUINA della stessa lesione reale

    IMPLEMENTAZIONE
      1. numero di viste per classe inversamente proporzionale alla
         frequenza: n_views_c = ceil((max_count / count_c) ** alpha)
         Con 7:1 e alpha=0.5, PAI 5 riceve ~3 viste e PAI 3 una sola.
      2. per ogni vista, tenere una frazione casuale p ~ U(p_min, p_max) dei
         token dentro la maschera (es. 0.6-1.0)
      3. garantire un minimo di token per vista, altrimenti su bbox piccole
         si aggrega rumore
      4. restituire le viste espanse con le etichette replicate

    SCELTE DA GIUSTIFICARE IN PRESENTAZIONE (ve le chiederanno)
      - alpha: quanto aggressivo e' il ribilanciamento? Sweep su
        {0.25, 0.5, 0.75, 1.0}.
      - p_min/p_max: se togliete troppi token perdete il segnale, se ne
        togliete troppo pochi le viste sono quasi identiche e non aggiungono
        niente. C'e' un optimum e va mostrato.
      - campionamento casuale uniforme o pesato per il peso di attenzione?
        Quest'ultimo e' piu' interessante: tenere i token a cui il pooling
        da' meno peso forza il modello a non dipendere da un solo token.
      - applicarlo solo in training, MAI in valutazione. In test si usano
        tutti i token: altrimenti confrontate cose diverse.

    ABLATION RICHIESTO DALL'OBIETTIVO 4
      - senza la novita' (CE semplice)
      - la novita' con alpha variabile
      - la novita' contro ogni baseline di questo file
      - metriche sulla minoritaria: Macro-F1, PR-AUC, confusion matrix
        (non l'accuracy globale: PAI 3 e' il 61%, un modello costante fa 61%)

    NOTA SULL'USO: va chiamata PER BATCH dentro il ciclo di training, non una
    volta sola sul dataset. Due motivi: le viste cambiano a ogni epoca (che e'
    il punto - viste identiche non aggiungono niente) e non si duplica in
    memoria l'intero tensore dei token.

    Ritorna: (tokens_espansi, mask_espansa, labels_espanse, indice_origine)

    La generazione delle viste vive in `_espandi`, condivisa con
    `random_token_sampling`: l'unica differenza fra i due e' la riga qui
    sotto, cioe' quante viste riceve ogni classe. Tenerla condivisa e' il
    motivo per cui il controllo a budget uguale misura la politica di
    allocazione e non due implementazioni diverse.
    """
    views = n_views_per_class(counts, alpha, num_classes).to(labels.device)
    per_sample = views[labels.long()]                      # viste per campione
    return _espandi(tokens, token_mask, labels, per_sample, generator,
                    p_min, p_max, min_tokens, attn_weights)


def n_views_uniform(counts, alpha=0.5, num_classes=NUM_CLASSES):
    """
    Lo STESSO budget di viste di `n_views_per_class`, ma diviso in parti
    uguali fra le classi invece che a favore della minoritaria.

    Serve al controllo dell'obiettivo 4: `balanced_token_sampling` fa DUE
    cose insieme, e finche' restano insieme non si sa quale delle due
    produce il risultato.
      (a) genera viste - sottoinsiemi di token della stessa lesione, che e'
          una forma di augmentation
      (b) ne assegna di piu' alle classi rare, che e' ribilanciamento
    Il campionamento uniforme tiene (a) e toglie (b). Se le due misure
    coincidono, il merito e' dell'augmentation e la novita' non ribilancia
    niente; se la versione bilanciata vince, il merito e' del
    ribilanciamento nello spazio dei token.

    Il budget e' il numero TOTALE di istanze per epoca, non il numero di
    viste per campione: e' quello che determina quanti passi di gradiente
    vede la testa, ed e' quello che va tenuto uguale.

    Restituisce viste frazionarie (float): il numero intero di viste per
    campione si estrae poi a caso in modo che la MEDIA sia questa. Con
    conteggi 3017/1229/473 e alpha 0.5 il budget e' 6894 istanze su 4719
    campioni, cioe' 1.461 viste ciascuno per tutte e tre le classi.
    """
    c = counts.float().clamp(min=1)
    budget = float((n_views_per_class(counts, alpha, num_classes).float() * c).sum())
    return torch.full((num_classes,), budget / float(c.sum()), device=counts.device)


def _espandi(tokens, token_mask, labels, per_sample, generator=None,
             p_min=0.6, p_max=1.0, min_tokens=4, attn_weights=None):
    """
    Nucleo condiviso: dato un numero di viste per campione, produce le viste.

    Estratto da `balanced_token_sampling` perche' la variante uniforme deve
    usare ESATTAMENTE la stessa procedura di generazione delle viste. Se le
    due la reimplementassero separatamente, una differenza nel confronto
    potrebbe venire da un dettaglio dell'implementazione invece che dalla
    politica di allocazione, che e' l'unica cosa che il controllo vuole
    isolare.
    """
    idx = torch.repeat_interleave(
        torch.arange(labels.shape[0], device=labels.device), per_sample
    )
    tok = tokens[idx]
    base = token_mask[idx]
    y = labels[idx]

    # La PRIMA vista di ogni campione resta integra: e' l'istanza originale.
    # Senza questa garanzia anche la maggioritaria verrebbe sottocampionata e
    # il confronto con le baseline non sarebbe piu' alla pari.
    first = torch.zeros(idx.shape[0], dtype=torch.bool, device=labels.device)
    conf = torch.cumsum(per_sample, 0) - per_sample
    first[conf[per_sample > 0]] = True

    n, t = base.shape
    p = torch.empty(n, device=base.device).uniform_(p_min, p_max, generator=generator)

    if attn_weights is None:
        score = torch.rand(n, t, device=base.device, generator=generator)
    else:
        # Campionamento pesato al CONTRARIO dell'attenzione: si tengono di
        # preferenza i token a cui il pooling da' meno peso, cosi' il modello
        # non puo' appoggiarsi a un singolo token dominante. E' la variante
        # piu' interessante da difendere in presentazione.
        w = attn_weights[idx].clamp(min=1e-6)
        score = torch.rand(n, t, device=base.device, generator=generator) / w

    # Si ordina solo dentro la maschera: i token fuori bbox non sono candidati.
    score = score.masked_fill(~base, float("inf"))
    n_in = base.sum(1)
    keep_n = torch.maximum((n_in.float() * p).long(),
                           torch.minimum(n_in, torch.full_like(n_in, min_tokens)))

    order = score.argsort(dim=1)
    ranks = order.argsort(dim=1)
    new_mask = ranks < keep_n[:, None]
    new_mask &= base
    new_mask[first] = base[first]

    # Salvagente: una vista senza token non e' aggregabile.
    vuote = ~new_mask.any(dim=1)
    if vuote.any():
        new_mask[vuote] = base[vuote]

    # `idx` dice da quale campione originale viene ogni vista. Serve a far
    # seguire alle viste qualunque altro attributo del campione - la
    # geometria della bbox, per esempio. Ricavarlo a valle dividendo per un
    # numero fisso di viste funziona solo quando le viste per campione sono
    # tutte uguali, che con il campionamento uniforme non e' vero: li'
    # varia da 1 a 2 e un fallback per posizione allineerebbe la bbox
    # sbagliata alla vista sbagliata, in silenzio.
    return tok, new_mask, y, idx


def random_token_sampling(tokens, token_mask, labels, counts,
                          num_classes=NUM_CLASSES, generator=None,
                          alpha=0.5, p_min=0.6, p_max=1.0, min_tokens=4,
                          attn_weights=None):
    """
    Controllo a budget uguale: stesse viste, allocate senza guardare la classe.

    E' `balanced_token_sampling` con l'unica differenza che conta: il numero
    di viste NON dipende dalla frequenza della classe. Il totale di istanze
    per epoca resta lo stesso a meno dell'arrotondamento casuale, quindi la
    testa vede lo stesso numero di passi di gradiente e la stessa quantita'
    di augmentation. Cambia solo CHI la riceve.

    Le viste per campione sono frazionarie (1.461 con alpha 0.5), e un
    campione non puo' ricevere 1.461 viste: se ne assegnano 1 con
    probabilita' 0.539 e 2 con probabilita' 0.461, cosi' la MEDIA e' esatta
    e il budget e' rispettato in valore atteso invece che in ogni batch.
    Arrotondare per difetto avrebbe dato un budget sistematicamente piu'
    basso e il controllo avrebbe confrontato anche il numero di passi.

    Ritorna: (tokens_espansi, mask_espansa, labels_espanse, indice_origine)
    """
    v = n_views_uniform(counts, alpha, num_classes).to(labels.device)
    atteso = v[labels.long()]
    base_n = atteso.floor()
    extra = torch.rand(atteso.shape[0], device=atteso.device,
                       generator=generator) < (atteso - base_n)
    per_sample = (base_n + extra.float()).long().clamp(min=1)
    return _espandi(tokens, token_mask, labels, per_sample, generator,
                    p_min, p_max, min_tokens, attn_weights)


# ==========================================================================
# Dispatcher
# ==========================================================================
def compute_loss(logits, targets, method="none", head_type="flat",
                 train_labels=None):
    """
    Contratto unico verso train_downstream.py.

    Metodo e tipo di testa entrano da qui, cosi' aggiungere una baseline non
    richiede di toccare il ciclo di addestramento.
    """
    weight = None
    if method in ("class_weighted", "focal") and train_labels is not None:
        weight = inverse_frequency_weights(train_labels).to(logits.device)

    if e_ordinale(head_type):
        # `focal` deve restare distinto da `class_weighted` anche qui:
        # entrambi passano da ordinal_loss, ma solo il primo attiva la
        # modulazione focale. Vedi la docstring di ordinal_loss.
        return ordinal_loss(logits, targets, weight=weight,
                            focal=(method == "focal"))

    if method == "focal":
        return focal_loss(logits, targets, weight=weight)
    return F.cross_entropy(logits, targets, weight=weight)


if __name__ == "__main__":
    # distribuzione reale attesa del dataset
    labels = torch.cat([
        torch.zeros(3691), torch.ones(1817), torch.full((521,), 2)
    ]).long()

    print("Distribuzione:", class_counts(labels).tolist())
    print(f"Sbilanciamento max:min = {3691/521:.2f} : 1")
    print("Pesi inverse-frequency:", [round(v, 3) for v in inverse_frequency_weights(labels).tolist()])

    logits = torch.randn(8, 3)
    t = torch.randint(0, 3, (8,))
    print(f"\nCE          : {compute_loss(logits, t, 'none').item():.4f}")
    print(f"CE pesata   : {compute_loss(logits, t, 'class_weighted', train_labels=labels).item():.4f}")
    print(f"Focal       : {compute_loss(logits, t, 'focal', train_labels=labels).item():.4f}")
    print(f"Ordinale    : {compute_loss(torch.randn(8,2), t, 'none', 'ordinal').item():.4f}")

    w = balanced_sampler_weights(labels)
    print(f"\nPesi sampler: PAI3={w[0]:.2e}  PAI4={w[4000]:.2e}  PAI5={w[-1]:.2e}")
    print(f"  rapporto PAI5/PAI3 = {(w[-1]/w[0]).item():.2f} (atteso ~7.08)")
