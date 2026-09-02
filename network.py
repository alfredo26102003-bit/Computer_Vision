"""
PIPELINE — ViT, I-JEPA, attention pooling, teste di classificazione.

Network - ViT compatto, I-JEPA (context + target EMA + predictor), teste.

Sezione "Network" della struttura richiesta dal corso.

Obiettivo 1 del brief, alla lettera: "a Context Encoder, a Target Encoder
updated via Exponential Moving Average (EMA), and a shallow Predictor
network". E' quello che c'e' qui.

"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import (
    ATTN_POOL_HEADS, CONTEXT_SCALE, NUM_CLASSES, NUM_TARGET_BLOCKS,
    PATCH_SIZE, PREDICTOR_DEPTH, PREDICTOR_DIM, PREDICTOR_HEADS,
    TARGET_ASPECT, TARGET_SCALE, TILE_SIZE, TOP_K, VIT_VARIANTS,
)


# ==========================================================================
# 1. Blocchi ViT
# ==========================================================================
def sincos_pos_embed(dim, grid_h, grid_w):
    """Positional embedding sinusoidale 2D (fisso, non appreso)."""
    def _1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2.0))
        out = pos.flatten()[:, None] * omega[None, :]
        return torch.cat([out.sin(), out.cos()], dim=1)

    gh = torch.arange(grid_h, dtype=torch.float32)
    gw = torch.arange(grid_w, dtype=torch.float32)
    grid = torch.meshgrid(gw, gh, indexing="xy")
    emb = torch.cat([_1d(dim // 2, grid[0]), _1d(dim // 2, grid[1])], dim=1)
    return emb  # (grid_h*grid_w, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """
    ViT che accetta un SOTTOINSIEME di token.

    Il supporto ai sottoinsiemi e' il requisito centrale di I-JEPA: il
    context encoder deve vedere solo le patch di contesto, non l'immagine
    intera con dei token mascherati. Da qui il parametro `keep_indices`.
    """

    def __init__(self, img_size=TILE_SIZE, patch_size=PATCH_SIZE, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(embed_dim, self.grid, self.grid)[None],
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, keep_indices=None, return_layers=None):
        """
        x: (B, C, H, W)
        keep_indices: (B, K) indici dei token da tenere, oppure None per tutti.
        return_layers: indici di blocco da concatenare invece dell'ultimo.

        Ritorna: (B, K o N, D) oppure (B, K o N, D*len(return_layers)).

        PERCHE' return_layers ESISTE. Misurato: le feature
        dell'ultimo blocco sono le piu' COMPRESSE. Con una sonda lineare -
        che e' cio' che fa la testa del downstream - un blocco intermedio
        rende molto di piu' dell'ultimo (+0.054 su pred48, +0.040 su un
        encoder casuale). Col k-NN vale il contrario, perche' quello premia
        l'organizzazione dello spazio e non l'informazione grezza.
        Concatenare piu' profondita' prende entrambe le cose.

        Il LayerNorm finale si applica SOLO all'ultimo blocco: e' calibrato
        su quello, e usarlo su un blocco intermedio cambierebbe cio' che si
        sta misurando.
        """
        t = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, D)
        t = t + self.pos_embed

        if keep_indices is not None:
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, t.shape[-1])
            t = torch.gather(t, 1, idx)

        if return_layers is None:
            for blk in self.blocks:
                t = blk(t)
            return self.norm(t)

        ultimo = len(self.blocks) - 1
        voluti = set(return_layers)
        out = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in voluti:
                out.append(self.norm(t) if i == ultimo else t)
        return torch.cat(out, dim=-1)


# ==========================================================================
# 2. Block masking (strategia I-JEPA)
# ==========================================================================
def sample_block(grid, scale_range, aspect_range, generator=None):
    """Campiona un blocco rettangolare di token; ritorna gli indici piatti."""
    n = grid * grid
    scale = torch.empty(1).uniform_(*scale_range, generator=generator).item()
    aspect = torch.empty(1).uniform_(*aspect_range, generator=generator).item()

    target_area = scale * n
    h = max(1, min(grid, int(round(math.sqrt(target_area / aspect)))))
    w = max(1, min(grid, int(round(math.sqrt(target_area * aspect)))))

    top = torch.randint(0, grid - h + 1, (1,), generator=generator).item()
    left = torch.randint(0, grid - w + 1, (1,), generator=generator).item()

    rows = torch.arange(top, top + h)
    cols = torch.arange(left, left + w)
    return (rows[:, None] * grid + cols[None, :]).flatten()


# Rapporto di mascheramento CONTROLLATO. None = comportamento del paper,
# in cui il rapporto non si imposta ma emerge da CONTEXT_SCALE, TARGET_SCALE
# e dalle sovrapposizioni. Misurato sulla configurazione del paper con
# griglia 14x14: il contesto residuo e' il 46.2% e il mascherato il 53.8%.
# Con una tupla (min, max) il rapporto diventa un parametro e viene
# rispettato su ogni estrazione.
MASK_RATIO = None
BLOCK_SCALE = (0.45, 0.60)


def sample_masks(grid, num_targets=NUM_TARGET_BLOCKS, generator=None):
    """
    Blocchi target rimossi dal contesto (altrimenti il compito e' banale).

    DUE REGIMI
    MASK_RATIO=None - quello del paper: un blocco di contesto ampio meno
      `num_targets` blocchi target. Il rapporto mascherato non e' un
      parametro: viene fuori dalle sovrapposizioni, e vale 53.8%.
    MASK_RATIO=(min,max) - si contano le PATCH: si fissa quante mascherarne
      e si aggiungono blocchi finche' quel numero e' raggiunto, troncando
      l'ultimo se sforerebbe. Cosi' il rapporto dichiarato e' quello che
      accade, su ogni estrazione e non in media.

    PERCHE' ALZARLO. Su radiografie dentali gran parte del campo e' osso
    uniforme: con poco mascheramento il predictor risolve interpolando dal
    vicinato immediato, una scorciatoia locale che non richiede alcuna
    comprensione della struttura. La loss infatti scende a 0.02-0.04 in
    poche epoche. Togliendo il vicinato la predizione deve appoggiarsi a
    regolarita' anatomiche a lungo raggio.

    IL COSTO VA TENUTO SOTTO CONTROLLO: ogni blocco e' una passata del
    predictor. Con blocchi piccoli servirebbero ~12 blocchi per l'80% e il
    passo costerebbe 2.9 volte. Con BLOCK_SCALE=(0.45,0.60) ne bastano 3.9,
    cioe' lo stesso costo dei 4 blocchi del paper. Misurato su 800
    estrazioni; rimisurate se cambiate la griglia.
    """
    n = grid * grid

    if MASK_RATIO is None:
        targets = [sample_block(grid, TARGET_SCALE, TARGET_ASPECT, generator)
                   for _ in range(num_targets)]
        context = sample_block(grid, CONTEXT_SCALE, (1.0, 1.0), generator)
        forbidden = torch.zeros(n, dtype=torch.bool)
        for t in targets:
            forbidden[t] = True
        context = context[~forbidden[context]]
        if context.numel() == 0:
            context = torch.arange(n)[~forbidden]
        if context.numel() == 0:
            context = torch.arange(n)[:1]
        return context, targets

    # --- regime a rapporto controllato
    min_contesto = 4
    voluto = torch.empty(1).uniform_(*MASK_RATIO, generator=generator).item()
    da_mascherare = min(int(round(voluto * n)), n - min_contesto)

    mask = torch.zeros(n, dtype=torch.bool)
    blocchi = []
    for _ in range(200):
        mascherate = int(mask.sum())
        if mascherate >= da_mascherare or len(blocchi) >= 24:
            break
        b = sample_block(grid, BLOCK_SCALE, TARGET_ASPECT, generator)
        nuove = b[~mask[b]]
        if nuove.numel() == 0:
            continue                      # gia' coperto: non consuma blocchi
        mancanti = da_mascherare - mascherate
        if nuove.numel() > mancanti:
            nuove = nuove[:mancanti]      # troncamento: mai sforare
        mask[nuove] = True
        blocchi.append(b[mask[b]])

    # Completamento: senza, il rapporto promesso resta una speranza.
    mancanti = da_mascherare - int(mask.sum())
    if mancanti > 0:
        libere = torch.arange(n)[~mask][:mancanti]
        mask[libere] = True
        blocchi.append(libere)

    return torch.arange(n)[~mask], blocchi


# ==========================================================================
# 3. Predictor
# ==========================================================================
class Predictor(nn.Module):
    """
    Predictor shallow e stretto (obiettivo 1: "a shallow Predictor network").

    Prende i token di contesto codificati piu' dei mask token posizionati
    dove stanno i target, e predice le rappresentazioni target nello spazio
    latente. Deve restare *piccolo*: se ha troppa capacita' risolve il
    compito senza costringere l'encoder a imparare nulla.
    """

    def __init__(self, embed_dim, pred_dim=None, depth=None,
                 heads=PREDICTOR_HEADS, num_patches=196):
        super().__init__()
        # Risolti a runtime e non come default dell'argomento: i default si
        # fissano alla definizione della classe, e gli override degli
        # esperimenti (train_ssl --predictor-dim) non avrebbero effetto.
        pred_dim = PREDICTOR_DIM if pred_dim is None else pred_dim
        depth = PREDICTOR_DEPTH if depth is None else depth
        self.proj_in = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        grid = int(math.sqrt(num_patches))
        self.register_buffer(
            "pos_embed", sincos_pos_embed(pred_dim, grid, grid)[None], persistent=False
        )
        self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(pred_dim)
        self.proj_out = nn.Linear(pred_dim, embed_dim)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        """
        ctx_tokens: (B, Kc, D) uscita del context encoder
        ctx_idx:    (B, Kc) posizioni dei token di contesto
        tgt_idx:    (B, Kt) posizioni da predire
        Ritorna:    (B, Kt, D)
        """
        b, d = ctx_tokens.shape[0], self.mask_token.shape[-1]
        x = self.proj_in(ctx_tokens)
        x = x + torch.gather(
            self.pos_embed.expand(b, -1, -1), 1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, d)
        )

        m = self.mask_token.expand(b, tgt_idx.shape[1], -1)
        m = m + torch.gather(
            self.pos_embed.expand(b, -1, -1), 1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, d)
        )

        z = torch.cat([x, m], dim=1)
        for blk in self.blocks:
            z = blk(z)
        return self.proj_out(self.norm(z[:, x.shape[1]:]))


# ==========================================================================
# 4. I-JEPA
# ==========================================================================
class IJEPA(nn.Module):
    """
    Pipeline I-JEPA completa: context encoder + target encoder EMA + predictor.

    Il target encoder e' una copia dei pesi del context encoder aggiornata per
    media mobile esponenziale e MAI dai gradienti - e' il meccanismo che
    impedisce la soluzione banale, ed e' anche il pezzo piu' fragile: fuori
    dal suo regime di iperparametri collassa. Monitoratelo (utils.py).
    """

    def __init__(self, variant="vit_tiny", img_size=TILE_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        cfg = VIT_VARIANTS[variant]
        self.context_encoder = VisionTransformer(img_size, patch_size, **cfg)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            cfg["embed_dim"], num_patches=self.context_encoder.num_patches
        )
        self.grid = self.context_encoder.grid
        self.embed_dim = cfg["embed_dim"]

    @torch.no_grad()
    def update_target(self, momentum: float):
        """EMA: theta_target <- m * theta_target + (1-m) * theta_context."""
        for pt, pc in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)
        for bt, bc in zip(self.target_encoder.buffers(),
                          self.context_encoder.buffers()):
            bt.copy_(bc)

    def forward(self, images, generator=None):
        """
        Ritorna (loss, embeddings_per_monitoraggio).

        Gli embedding restituiti servono a utils.CollapseMonitor: sono la
        media dei token del target encoder, cioe' la rappresentazione che
        userete a valle. Se collassano, collassa il progetto.
        """
        b = images.shape[0]
        device = images.device

        ctx_idx, tgt_blocks = sample_masks(self.grid, generator=generator)
        ctx_idx = ctx_idx.to(device)[None].expand(b, -1)

        ctx_tokens = self.context_encoder(images, ctx_idx)

        with torch.no_grad():
            full = self.target_encoder(images)          # (B, N, D)
            full = F.layer_norm(full, (full.shape[-1],))

        loss = images.new_zeros(())
        for tgt in tgt_blocks:
            tgt_idx = tgt.to(device)[None].expand(b, -1)
            target = torch.gather(
                full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1])
            )
            pred = self.predictor(ctx_tokens, ctx_idx, tgt_idx)
            loss = loss + F.smooth_l1_loss(pred, target)

        loss = loss / len(tgt_blocks)


        return loss, full.mean(dim=1).detach()

    @torch.no_grad()
    def encode(self, images, return_layers=None):
        """Encoder congelato per il downstream: token del target encoder."""
        return self.target_encoder(images, return_layers=return_layers)


# ==========================================================================
# 4b. Braccio di confronto: encoder ImageNet congelato
# ==========================================================================
class FrozenImageNetEncoder(nn.Module):
    """
    ViT-B/16 pre-addestrato su ImageNet, congelato ed esposto con la STESSA
    interfaccia di IJEPA (encode / grid / embed_dim).

    E' il braccio 2 della sez.9 dell'analisi, quello definito "critico e non
    negoziabile": se il JEPA in-domain su ~4k immagini non batte il transfer
    da ImageNet, quello E' il risultato del progetto e va detto. Senza questo
    confronto, un numero come "Macro-F1 0.62" non dimostra niente, ed e' la
    prima domanda che arriva in sede d'esame.

    Il patch da 16 su tile da 224 da' la stessa griglia 14x14 del nostro ViT,
    quindi bbox_to_token_mask e l'attention pooling funzionano identici e il
    confronto e' davvero alla pari: cambia l'encoder, nient'altro.
    """

    def __init__(self, img_size=TILE_SIZE, patch_size=PATCH_SIZE):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.net = vit_b_16(weights=weights)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

        self.grid = img_size // patch_size
        self.embed_dim = self.net.hidden_dim

        # I tile arrivano da data.py in [-1, 1] (grayscale replicato su 3
        # canali). Il ViT di torchvision vuole invece le statistiche di
        # ImageNet: si torna in [0, 1] e si ri-normalizza. Saltare questo
        # passaggio non da' errore, da' solo feature peggiori - cioe'
        # sabotarebbe silenziosamente proprio il braccio di confronto.
        t = weights.transforms()
        self.register_buffer("mean", torch.tensor(t.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(t.std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, images, return_layers=None):
        """
        `return_layers` concatena piu' profondita', come per il nostro ViT.
        Serve perche' il confronto fra bracci resti alla pari: se il JEPA
        usa piu' layer e ImageNet solo l'ultimo, non si confrontano piu' gli
        encoder ma i protocolli di estrazione.
        """
        x = (images * 0.5 + 0.5 - self.mean) / self.std
        x = self.net._process_input(x)
        cls = self.net.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        if return_layers is None:
            return self.net.encoder(x)[:, 1:]

        # torchvision tiene i blocchi in encoder.layers; qui si replica il
        # percorso a mano per poter intercettare le profondita' intermedie.
        x = self.net.encoder.dropout(x + self.net.encoder.pos_embedding)
        blocchi = self.net.encoder.layers
        ultimo = len(blocchi) - 1
        voluti = set(return_layers)
        out = []
        for i, blk in enumerate(blocchi):
            x = blk(x)
            if i in voluti:
                y = self.net.encoder.ln(x) if i == ultimo else x
                out.append(y[:, 1:])
        return torch.cat(out, dim=-1)

# RIPRISTINATA IL 27 AGOSTO, dopo essere stata tolta da `pulizia_uno` come
# codice non richiesto dal brief. Serve a rispondere alla sola domanda che
# i nostri tre encoder non possono rispondere da soli: se il pre-training
# in-domain non batte l'inizializzazione casuale, e' un limite del dominio
# o della nostra implementazione? Un encoder pre-addestrato DA ALTRI, su
# milioni di immagini, con un'implementazione notoriamente corretta, e'
# l'unico controllo esterno possibile.
#
# ATTENZIONE AL CONFRONTO. E' un ViT-B/16 (768 dim) contro il nostro
# ViT-S/16 (384): con `--layers 2 7 11` sono 2304 dimensioni contro 1152.
# Il vantaggio e' suo. Questo rende un risultato NULLO piu' forte, non piu'
# debole: se non batte il casuale neanche col doppio delle dimensioni, il
# tetto non e' dell'architettura.

# ==========================================================================
# 5. Teste downstream
# ==========================================================================
class AttentionPooling(nn.Module):
    """
    Attention pooling sui token, con maschera opzionale.

    La maschera limita l'aggregazione ai token che cadono dentro la bbox
    della lesione - che e' quello che chiede il brief ("extract the latent
    vectors corresponding to the lesion areas").
    """

    def __init__(self, dim, heads=ATTN_POOL_HEADS):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, token_mask=None):
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        kpm = ~token_mask if token_mask is not None else None
        out, w = self.attn(q, tokens, tokens, key_padding_mask=kpm)
        return self.norm(out.squeeze(1)), w


class GatedAttentionPooling(nn.Module):
    """
    Attention pooling gated in stile MIL (Ilse et al. 2018, "Attention-based
    Deep Multiple Instance Learning").

    PERCHE' QUI. Il problema e' letteralmente un problema di multiple
    instance learning: la bbox contiene un sacchetto di token e l'etichetta
    e' del sacchetto, non dei singoli token. L'attention pooling attuale usa
    una query APPRESA MA FISSA - lo stesso vettore per ogni lesione - quindi
    i pesi dipendono dal token solo attraverso un prodotto scalare con una
    direzione unica. Il gating rende il punteggio del token una funzione non
    lineare del token stesso:

        a_i proporzionale a  w^T ( tanh(V h_i)  *  sigmoid(U h_i) )

    Il ramo tanh propone quanto il token e' rilevante, il ramo sigmoid
    decide quanto lasciarlo passare. Il prodotto fra i due e' cio' che la
    query fissa non puo' esprimere: "questo token conta SE anche
    quest'altra caratteristica e' presente". Sul PAI e' esattamente la
    struttura del problema - una regione conta se e' insieme grande E scura.

    COSTO. Con dim=1152 e nascosto=128 sono ~0.3M parametri, contro i 5.3M
    dell'attention multi-testa che sostituisce: e' piu' LEGGERO, non piu'
    pesante. Il brief chiede una testa leggera e questo la rispetta.
    """

    def __init__(self, dim, nascosto=128):
        super().__init__()
        self.V = nn.Linear(dim, nascosto)
        self.U = nn.Linear(dim, nascosto)
        self.w = nn.Linear(nascosto, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, token_mask=None):
        a = self.w(torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens)))
        a = a.squeeze(-1)                                   # (B, T)
        if token_mask is not None:
            # I token fuori bbox non sono candidati: -inf li azzera dopo il
            # softmax. Mascherare DOPO il softmax non basterebbe, i pesi
            # rimasti non sommerebbero piu' a uno.
            a = a.masked_fill(~token_mask, float("-inf"))
        w = torch.softmax(a, dim=-1)
        # Una riga interamente mascherata darebbe softmax di soli -inf, cioe'
        # NaN. Non dovrebbe succedere - bbox_to_token_mask garantisce almeno
        # un token - ma un NaN qui si propaga a tutti i pesi e il run muore
        # senza dire perche'.
        w = torch.nan_to_num(w)
        out = torch.einsum("bt,btd->bd", w, tokens)
        return self.norm(out), w[:, None, :]


class TopKAttentionPooling(nn.Module):
    """
    Attenzione ristretta ai k token con punteggio piu' alto.

    PERCHE'. Il softmax e' denso: distribuisce peso su TUTTI i token della
    bbox, anche quelli che contengono solo osso sano al bordo della
    radiotrasparenza. Le bbox mediane per classe hanno lati 57/80/127 px,
    cioe' da ~9 a ~64 token, e su quelle grandi la lesione vera occupa una
    frazione del rettangolo: il resto e' contorno. Un softmax denso lo media
    dentro comunque.

    Il top-k tiene i k token piu' forti e rinormalizza solo fra loro. E'
    l'ipotesi opposta a quella della novita': la novita' dice "non
    appoggiarti a pochi token", il top-k dice "appoggiati solo ai migliori".
    Vale la pena misurarle contro, perche' se il top-k vince l'argomento
    della novita' si indebolisce, e questo va saputo prima della
    presentazione invece che durante.

    k e' un CAP, non un numero fisso: se la bbox ha meno di k token si
    tengono quelli che ci sono. Con k >= max token la funzione degenera
    esattamente nel softmax denso, che e' il controllo giusto.
    """

    def __init__(self, dim, k=8, heads=ATTN_POOL_HEADS):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.punteggio = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.k = k

    def forward(self, tokens, token_mask=None):
        a = self.punteggio(tokens).squeeze(-1)              # (B, T)
        if token_mask is not None:
            a = a.masked_fill(~token_mask, float("-inf"))

        k = min(self.k, a.shape[1])
        soglia = a.topk(k, dim=-1).values[:, -1:]           # k-esimo valore
        # `>=` e non `>`: con pareggi al k-esimo posto un `>` scarterebbe
        # tutti gli ex aequo e potrebbe svuotare la riga.
        tenuti = (a >= soglia) & torch.isfinite(a)
        a = a.masked_fill(~tenuti, float("-inf"))

        w = torch.nan_to_num(torch.softmax(a, dim=-1))
        out = torch.einsum("bt,btd->bd", w, tokens)
        return self.norm(out), w[:, None, :]


def bbox_to_token_mask(bbox, grid, patch_size=PATCH_SIZE):
    """
    Converte bbox in coordinate pixel in una maschera booleana sui token.

    Se la maschera risulta vuota per qualche campione, la bbox e' piu'
    piccola di un token: e' il sintomo del problema di scala descritto in
    ANALISI_PROGETTO_8.md sez.2. Qui teniamo almeno il token del centro, ma se
    succede spesso la risoluzione e' sbagliata.
    """
    b = bbox.shape[0]
    device = bbox.device
    x0 = (bbox[:, 0] / patch_size).floor().long().clamp(0, grid - 1)
    y0 = (bbox[:, 1] / patch_size).floor().long().clamp(0, grid - 1)
    x1 = (bbox[:, 2] / patch_size).ceil().long().clamp(1, grid)
    y1 = (bbox[:, 3] / patch_size).ceil().long().clamp(1, grid)

    cols = torch.arange(grid, device=device)[None, :]
    mask_x = (cols >= x0[:, None]) & (cols < x1[:, None])
    mask_y = (cols >= y0[:, None]) & (cols < y1[:, None])
    mask = (mask_y[:, :, None] & mask_x[:, None, :]).reshape(b, -1)

    empty = ~mask.any(dim=1)
    if empty.any():
        cx = ((bbox[:, 0] + bbox[:, 2]) / 2 / patch_size).long().clamp(0, grid - 1)
        cy = ((bbox[:, 1] + bbox[:, 3]) / 2 / patch_size).long().clamp(0, grid - 1)
        mask[empty, (cy * grid + cx)[empty]] = True
    return mask


class FlatHead(nn.Module):
    """Softmax a 3 vie - quello che chiede il brief."""

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """
    Testa con uno strato nascosto, normalizzazione e dropout.

    PERCHE'. La testa precedente e' un solo nn.Linear(1152, 3): puo' solo
    separare le classi con degli iperpiani. Ma il grado PAI dipende dalla
    CONGIUNZIONE di due grandezze - quanto e' grande la radiotrasparenza e
    quanto e' scura - e "grande E scura" non e' una funzione lineare di
    "grande" e "scura" prese separatamente. Uno strato nascosto con una non
    linearita' puo' rappresentare quell'interazione.

    IL LayerNorm IN INGRESSO NON E' ORNAMENTALE. Le feature sono la
    concatenazione dei blocchi 2, 7 e 11 del ViT, che hanno scale diverse
    fra loro. Senza normalizzare, il blocco con la scala piu' grande domina
    il gradiente e gli altri due contano poco: si userebbe un terzo
    dell'informazione credendo di usarla tutta.

    RESTA LEGGERA, come chiede il brief ("a lightweight multi class
    classification head"): con nascosto=256 sono ~0.3M parametri contro i
    5.3M dell'attention pooling che c'e' gia'. Il dropout serve perche' le
    lesioni di training sono 4719: con una testa piu' capace il rischio di
    sovradattamento cresce, e va contenuto invece che scoperto dopo.
    """

    def __init__(self, dim, num_classes=NUM_CLASSES, nascosto=256, dropout=0.2,
                 ordinale=False):
        super().__init__()
        uscite = (num_classes - 1) if ordinale else num_classes
        self.norm = nn.LayerNorm(dim)
        self.rete = nn.Sequential(
            nn.Linear(dim, nascosto),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nascosto, uscite),
        )

    def forward(self, x):
        return self.rete(self.norm(x))


class NormFlatHead(nn.Module):
    """
    Solo LayerNorm piu' lineare: serve a separare l'effetto della
    normalizzazione da quello dello strato nascosto.

    Senza questo braccio, un eventuale miglioramento di MLPHead non si
    saprebbe attribuire: puo' venire dalla non linearita' o soltanto
    dall'aver messo le tre profondita' sulla stessa scala.
    """

    def __init__(self, dim, num_classes=NUM_CLASSES, ordinale=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, (num_classes - 1) if ordinale else num_classes)

    def forward(self, x):
        return self.fc(self.norm(x))


class OrdinalHead(nn.Module):
    """
    Testa ordinale in stile CORAL - piu' appropriata al PAI.

    Il PAI e' una scala ordinale: 3 < 4 < 5. Confondere PAI 3 con PAI 5 e'
    clinicamente peggio che confondere 4 con 5, ma una softmax piatta li
    tratta allo stesso modo. Qui si predicono K-1 soglie cumulative
    P(y > 3), P(y > 4) con un peso condiviso e bias separati, il che impone
    la monotonicita'.

    Tenete comunque FlatHead come braccio di confronto: la scelta va
    argomentata con i numeri, non per principio.
    """

    def __init__(self, dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.shared = nn.Linear(dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        self.num_classes = num_classes

    def forward(self, x):
        return self.shared(x) + self.biases     # (B, K-1) logit cumulativi

    @staticmethod
    def logits_to_class(cum_logits):
        return (cum_logits > 0).sum(dim=1)

    @staticmethod
    def targets(labels, num_classes=NUM_CLASSES):
        """label k -> [1]*k + [0]*(K-1-k)"""
        lv = torch.arange(num_classes - 1, device=labels.device)[None, :]
        return (labels[:, None] > lv).float()


class InstanceMILHead(nn.Module):
    """
    Multiple Instance Learning a livello di ISTANZA: si giudica ogni token,
    poi si mediano le probabilita'. Non si aggregano le feature.

    PERCHE'. Il brief dice di "estrarre i vettori latenti corrispondenti
    alle aree lesionate" - plurale, cardinalita' variabile: 16 / 36 / 64
    token secondo la lesione. E' un problema di Multiple Instance Learning
    alla lettera, e il difetto noto di quella formulazione e' il BAG-SIZE
    BIAS: quando la cardinalita' del sacchetto correla con l'etichetta, il
    modello impara a contare invece che a guardare. Qui la correlazione e'
    quasi perfetta, perche' il grado PAI e' quasi tutto dimensione: la sola
    maschera one-hot, senza un pixel, da' macro-F1 0.7708.

    L'INVARIANZA, in una riga. La media di N probabilita' non dipende da N.
    Il conteggio sparisce PER COSTRUZIONE, non per compensazione - ed e'
    una proprieta' algebrica dell'aggregatore, non un effetto misurato.

    PERCHE' SERVE LA NON LINEARITA'. Con una testa lineare istanza e
    embedding coincidono: media(W x_i) = W media(x_i), cioe' classificare
    ogni token e mediare le decisioni E' applicare il classificatore alla
    media delle feature. La differenza esiste solo se il giudizio per token
    e' non lineare, ed e' per questo che qui c'e' un MLP e non un Linear.

    Chiede a ogni token "quanto sei tessuto da PAI 5?" e media le risposte,
    invece di chiedere "com'e' questa regione nel complesso?".

    Restituisce il LOGARITMO della probabilita' media. Serve perche' a valle
    si applica cross_entropy, che fa log_softmax: e siccome
    log_softmax(log p) = log p quando p somma a uno, la loss risulta la NLL
    corretta della probabilita' media. Restituire i logit grezzi la
    trasformerebbe in qualcos'altro, in silenzio.
    """

    def __init__(self, dim, num_classes=NUM_CLASSES, nascosto=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.rete = nn.Sequential(
            nn.Linear(dim, nascosto),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nascosto, num_classes),
        )

    def forward(self, tokens, token_mask=None):
        p = torch.softmax(self.rete(self.norm(tokens)), dim=-1)   # (B, N, C)
        if token_mask is None:
            media = p.mean(1)
        else:
            w = token_mask.float()[..., None]
            media = (p * w).sum(1) / w.sum(1).clamp(min=1)
        # I token fuori maschera non pesano: la media e' sui soli token
        # della lesione, come chiede la traccia.
        return torch.log(media.clamp(min=1e-8))


class LesionClassifier(nn.Module):
    """
    Encoder CONGELATO + attention pooling + testa. Solo pooling e testa si
    addestrano.

    `geom_dim` aggiunge alla rappresentazione aggregata la GEOMETRIA della
    bbox in pixel nativi (larghezza, altezza, lato massimo, radice
    dell'area). Non e' un abbellimento: misurato, due sole soglie
    sul lato della bbox danno macro-F1 0.7567 e kappa 0.7779 sul test, senza
    usare nessuna rete. Il grado PAI
    e' in larga parte l'estensione della radiotrasparenza, e i lati mediani
    per classe sono 57 / 81 / 126 px: quasi separabili da soli.

    Il Task dice "using the provided bounding box coordinates": le
    coordinate sono un input fornito dal dataset, quindi usarne la
    geometria e' dentro la traccia. Va pero' DICHIARATO in presentazione,
    e va riportata anche la versione senza, altrimenti non si capisce
    quanto pesa l'encoder e quanto la geometria.
    """

    def __init__(self, embed_dim, grid, head_type="flat", geom_dim=0,
                 pool_type="attn", top_k=TOP_K):
        super().__init__()
        # Il pooling e' l'altra meta' della parte addestrabile, ed e' quella
        # che decide COME i token della bbox diventano un vettore solo.
        # Tenerlo scambiabile permette di misurare se il collo di bottiglia
        # sta li' invece che nella testa - due ipotesi diverse che senza
        # questo interruttore non si separano.
        pooling = {
            "attn":  lambda: AttentionPooling(embed_dim),
            "gated": lambda: GatedAttentionPooling(embed_dim),
            "topk":  lambda: TopKAttentionPooling(embed_dim, k=top_k),
        }
        if pool_type not in pooling:
            raise ValueError(f"pooling sconosciuto: {pool_type}. "
                             f"Disponibili: {sorted(pooling)}")
        self.pool = pooling[pool_type]()
        self.pool_type = pool_type
        self.geom_dim = geom_dim
        dim = embed_dim + geom_dim
        if geom_dim:
            # Normalizzata a parte: le scale di latenti e geometria sono
            # diverse di ordini di grandezza e senza questo la testa vedrebbe
            # solo i latenti.
            self.geom_norm = nn.LayerNorm(geom_dim)
        # Sei teste: due baseline piu' quattro varianti. Le varianti sono
        # combinazioni di due leve indipendenti - normalizzazione in
        # ingresso e strato nascosto - cosi' l'effetto di ciascuna si puo'
        # attribuire invece di misurarne solo la somma.
        teste = {
            "flat":        lambda: FlatHead(dim),
            "ordinal":     lambda: OrdinalHead(dim),
            "norm":        lambda: NormFlatHead(dim),
            "norm_ord":    lambda: NormFlatHead(dim, ordinale=True),
            "mlp":         lambda: MLPHead(dim),
            "mlp_ord":     lambda: MLPHead(dim, ordinale=True),
            # `mil` non e' una testa come le altre: sostituisce ANCHE il
            # pooling, perche' aggrega decisioni invece che feature.
            "mil":         lambda: InstanceMILHead(dim),
        }
        if head_type not in teste:
            raise ValueError(f"testa sconosciuta: {head_type}. "
                             f"Disponibili: {sorted(teste)}")
        self.head = teste[head_type]()
        self.head_type = head_type
        self.grid = grid

    def forward(self, tokens, bbox=None, token_mask=None, geom=None):
        if token_mask is None and bbox is not None:
            token_mask = bbox_to_token_mask(bbox, self.grid)
        if self.head_type == "mil":
            # Niente pooling: il giudizio e' per token e si media dopo.
            # L'attention pooling resta costruito ma inutilizzato, cosi' il
            # numero di parametri addestrabili non cambia fra i due rami e
            # il confronto non misura anche la capacita'.
            return self.head(tokens, token_mask), None, None
        pooled, attn = self.pool(tokens, token_mask)
        if self.geom_dim:
            if geom is None:
                raise ValueError("geom_dim > 0 ma nessuna geometria passata")
            pooled = torch.cat([pooled, self.geom_norm(geom)], dim=-1)
        return self.head(pooled), pooled, attn


def build_ijepa(variant="vit_tiny"):
    return IJEPA(variant)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    for variant in VIT_VARIANTS:
        model = build_ijepa(variant)
        x = torch.randn(2, 3, TILE_SIZE, TILE_SIZE)
        loss, emb = model(x)
        print(f"{variant:10s} params={count_params(model)/1e6:5.2f}M  "
              f"grid={model.grid}x{model.grid}  loss={loss.item():.4f}  "
              f"emb={tuple(emb.shape)}")

    m = build_ijepa("vit_tiny")
    clf = LesionClassifier(m.embed_dim, m.grid, "ordinal")
    tokens = m.encode(torch.randn(4, 3, TILE_SIZE, TILE_SIZE))
    bbox = torch.tensor([[40., 40., 120., 120.]] * 4)
    logits, pooled, _ = clf(tokens, bbox)
    print(f"\ntoken={tuple(tokens.shape)} -> pooled={tuple(pooled.shape)} "
          f"-> logit ordinali={tuple(logits.shape)}")
    print(f"maschera token dentro bbox: {bbox_to_token_mask(bbox, m.grid).sum(1).tolist()}")
