"""
CONFRONTO — PR-AUC PAI 5 fra teste e tipi di pooling.

La testa e' il solo pezzo addestrabile: e' lei il collo di bottiglia?

L'encoder e' congelato per vincolo del brief, quindi tutta la capacita'
disponibile sta nel pooling piu' la testa. La testa attuale e' un solo
nn.Linear(1152, 3): vale la pena chiedersi quanto costa quella semplicita'.

DUE LEVE, misurate separatamente per poterle attribuire:
  norm  LayerNorm in ingresso. Le feature sono i blocchi 2, 7 e 11
        concatenati, con scale diverse: senza normalizzare, quello con la
        scala maggiore domina il gradiente.
  mlp   uno strato nascosto con GELU. Il grado PAI dipende dalla
        congiunzione di dimensione e scurezza, che non e' lineare nelle due
        separate.

Valutazione su VALIDATION: qui si sceglie il protocollo, e sceglierlo
guardando il test sarebbe barare. Il test si tocca solo alla fine.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import N_SEEDS, OUT_DIR
from train_downstream import load_latents, train_head
from utils import Freno, save_json

CARICO = 80

TESTE = [
    ("flat",     "lineare (attuale)"),
    ("norm",     "LayerNorm + lineare"),
    ("mlp",      "LayerNorm + nascosto 256 + GELU"),
    ("ordinal",  "ordinale (attuale)"),
    ("norm_ord", "LayerNorm + ordinale"),
    ("mlp_ord",  "LayerNorm + nascosto + ordinale"),
]


def misura(cached, head, freno, seeds, pool="attn", split="val", metodo="none"):
    f1, pr, rec = [], [], []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, metodo, head, seed=s, pool_type=pool)
        r = evaluate_split(clf, cached["data"][split], head)
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
        f1.append(r["macro_f1"])
        pr.append(r.get("pr_auc_pai5", np.nan))
        rec.append(r["recall_per_class"][2] if "recall_per_class" in r
                   else r.get("recall_pai5", np.nan))
    return np.array(f1), np.array(pr), np.array(rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_geo_completa",
                    help="quale encoder. Il default e' I-JEPA completa_best "
                         "(epoca 179), estratto con provenienza nota")
    ap.add_argument("--controllo", default="_casuale",
                    help="encoder di controllo per le due teste migliori")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--pool", nargs="+", default=["attn", "gated", "topk"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    ap.add_argument("--protocollo", default=None,
                    help="maschera alternativa, es. P3_K16: i K token piu' "
                         "vicini al centro della bbox invece di tutti quelli "
                         "dentro. Toglie il canale del conteggio")
    ap.add_argument("--carico", type=int, default=100)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}\n")
    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)
    if a.protocollo:
        from exp_fixedk import con_maschera
        cached = con_maschera(cached, a.protocollo)
        n = cached["data"]["test"]["mask"].sum(1).float().mean()
        print(f"protocollo {a.protocollo}: {n:.0f} token per lesione, "
              f"uguali per tutte le classi")
    print(f"encoder{a.tag}, embed {cached['embed_dim']}, "
          f"{len(a.seeds)} seed, metodo `none`\n")

    fuori = {"tag": a.tag, "seeds": a.seeds, "teste": {}, "pooling": {},
             "test": {}}
    percorso = os.path.join(OUT_DIR, f"testa_pooling{a.tag}{'_'+a.protocollo if a.protocollo else ''}.json")

    # ---------------- 1. le sei teste, pooling attuale ----------------
    print("SEI TESTE, pooling `attn`, misura su VALIDATION")
    print(f"{'testa':36s} {'macro-F1':>17s} {'PR-AUC PAI5':>17s}")
    print("-" * 74)
    esiti = {}
    for head, nome in TESTE:
        f1, pr, rec = misura(cached, head, freno, a.seeds)
        esiti[head] = (f1, pr)
        fuori["teste"][head] = {"nome": nome,
                                "macro_f1": [float(f1.mean()), float(f1.std())],
                                "pr_auc_pai5": [float(pr.mean()), float(pr.std())]}
        print(f"  {nome:34s} {f1.mean():.4f}+-{f1.std():.4f}  "
              f"{pr.mean():.4f}+-{pr.std():.4f}", flush=True)
        save_json(fuori, percorso)

    print(f"\n{'confronto':36s} {'d macro-F1':>12s} {'z':>7s}")
    print("-" * 60)
    for base, var, eti in [("flat","norm","norm vs lineare"),
                           ("flat","mlp","mlp vs lineare"),
                           ("norm","mlp","mlp vs norm (solo il nascosto)"),
                           ("ordinal","norm_ord","norm_ord vs ordinale"),
                           ("ordinal","mlp_ord","mlp_ord vs ordinale")]:
        (fb,_),(fv,_) = esiti[base], esiti[var]
        d = fv.mean()-fb.mean()
        se = float(np.sqrt(fv.var(ddof=1)/len(fv) + fb.var(ddof=1)/len(fb)))
        print(f"  {eti:34s} {d:+12.4f} {d/se if se>0 else float('inf'):+7.2f}")

    # SI SELEZIONA SULLA PR-AUC DI PAI 5, non sulla macro-F1.
    # La macro-F1 media le tre classi con lo stesso peso e dipende
    # dall'argmax; la PR-AUC su PAI 5 e' la metrica che il brief dichiara
    # primaria - specifica per la minoritaria e indipendente dalla soglia.
    # Selezionare su una metrica e riportare l'altra e' lo stesso errore
    # commesso col checkpoint `_best`, scelto massimizzando una misura che
    # si e' poi rivelata cieca a cio' che contava. Qui, con la macro-F1,
    # vinceva `mlp_ord` che sulla PR-AUC5 perdeva 0.047 contro `flat`.
    migliore = max(esiti, key=lambda h: esiti[h][1].mean())
    print(f"\nTesta migliore su validation: {migliore} "
          f"({esiti[migliore][0].mean():.4f})")

    # ---------------- 2. i tre pooling, con la testa migliore ----------------
    print(f"\nTRE POOLING con la testa `{migliore}`, VALIDATION")
    print(f"{'pooling':36s} {'macro-F1':>17s} {'PR-AUC PAI5':>17s}")
    print("-" * 74)
    ep = {}
    for pool in a.pool:
        f1, pr, _ = misura(cached, migliore, freno, a.seeds, pool=pool)
        ep[pool] = pr
        fuori["pooling"][pool] = {"macro_f1": [float(f1.mean()), float(f1.std())],
                                  "pr_auc_pai5": [float(pr.mean()), float(pr.std())]}
        print(f"  {pool:34s} {f1.mean():.4f}+-{f1.std():.4f}  "
              f"{pr.mean():.4f}+-{pr.std():.4f}", flush=True)
        save_json(fuori, percorso)
    pool_migliore = max(ep, key=lambda p: ep[p].mean())   # PR-AUC5, vedi sopra

    # ---------------- 3. sul TEST, solo la scelta e i riferimenti ----------------
    # Selezionato su validation, il test si tocca una volta sola e solo per
    # la configurazione scelta piu' la baseline. Riportare tutta la griglia
    # di test dopo aver selezionato reintrodurrebbe il massimo su molte
    # estrazioni che la selezione su validation serviva a evitare.
    print(f"\nTEST - configurazione scelta ({migliore} + {pool_migliore}) "
          f"e riferimento (flat + attn), su entrambi gli encoder")
    print(f"{'encoder':16s} {'configurazione':22s} {'macro-F1':>17s} {'PR-AUC PAI5':>17s}")
    print("-" * 78)
    for tag in (a.tag, a.controllo):
        c = cached if tag == a.tag else load_latents(a.variant, layers=a.layers, tag=tag)
        if a.protocollo and tag != a.tag:
            from exp_fixedk import con_maschera
            c = con_maschera(c, a.protocollo)
        for h, pl in dict.fromkeys([(migliore, pool_migliore), ("flat", "attn")]):
            f1, pr, _ = misura(c, h, freno, a.seeds, pool=pl, split="test")
            fuori["test"][f"{tag}|{h}|{pl}"] = {
                "macro_f1": [float(f1.mean()), float(f1.std())],
                "pr_auc_pai5": [float(pr.mean()), float(pr.std())]}
            print(f"{tag:16s} {h + ' + ' + pl:22s} {f1.mean():.4f}+-{f1.std():.4f}  "
                  f"{pr.mean():.4f}+-{pr.std():.4f}", flush=True)
            save_json(fuori, percorso)
        if tag != a.tag:
            del c

    print(f"\nRisultati in {percorso}")
