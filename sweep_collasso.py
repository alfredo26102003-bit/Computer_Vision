"""
CONFRONTO — sonda k-NN fra configurazioni anti-collasso del pre-training.

Ricerca degli iperparametri che evitano il collasso del pre-training.

Non e' un confronto con altre tecnologie: e' la taratura di I-JEPA, il
modello che il brief prescrive. Serve perche' con la configurazione di
riferimento l'encoder collassa - rango 2 su 280, std 0.030 contro 0.051 di
un modello sano - e la sonda k-NN scende da 0.7030 (encoder casuale) a
0.4354 in 10 epoche: il pre-training TOGLIE informazione invece di
aggiungerla.

LE LEVE, in ordine di effetto atteso
  ema_start  momentum del target encoder. E' l'unico meccanismo
             anti-collasso di I-JEPA: il target e' una media mobile del
             context encoder, e se e' troppo veloce il context lo raggiunge
             sulla soluzione costante. Piu' alto = target piu' lento.
  lr         un passo grande rende la scorciatoia costante piu' facile da
             raggiungere prima che l'encoder impari qualcosa.
  pred_dim   un predictor troppo debole non riesce a predire target ricchi,
             e il modo piu' semplice di abbassare la loss diventa rendere i
             target tutti uguali. 96 su 384 e' molto stretto.

PROTOCOLLO
Ogni configurazione gira 11 epoche (~15 min) e viene giudicata sulla sonda
k-NN all'epoca 10 contro lo STESSO riferimento: l'encoder appena
inizializzato, 0.7030. Una configurazione che non supera il riferimento non
merita le altre 290 epoche.
"""

import json
import os
import re
import subprocess
import sys
import time

PY = os.path.join(".venv", "Scripts", "python")
EPOCHE = 11
LOGS = "logs"

CONFIG = [
    # nome        ema      lr       pred_dim
    ("ema999",    0.999,   5e-5,    192),
    ("ema9995",   0.9995,  5e-5,    192),
    ("pred384",   0.999,   1.5e-4,  384),
    ("lento",     0.9996,  3e-5,    96),
]


def lancia(nome, ema, lr, pred):
    log = os.path.join(LOGS, f"sweep_{nome}.log")
    cmd = [PY, "-u", "train_ssl.py", "--variant", "vit_small",
           "--epochs", str(EPOCHE), "--batch-size", "128", "--tag", nome,
           "--ema-start", str(ema), "--lr", str(lr),
           "--predictor-dim", str(pred), "--gate-epoch", "999"]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    testo = open(log, encoding="utf-8", errors="replace").read()

    # Il riferimento e' stampato dal run stesso: si rilegge invece di
    # riscriverlo a mano, cosi' non puo' divergere.
    rif = re.search(r"Da battere: macro-F1 k-NN = ([\d.]+)", testo)
    sonde = re.findall(r"\[k-NN probe\] acc=[\d.]+ macroF1=([\d.]+)", testo)
    return {
        "nome": nome, "ema": ema, "lr": lr, "pred_dim": pred,
        "riferimento": float(rif.group(1)) if rif else None,
        # la prima sonda e' il riferimento stesso: le successive sono il run
        "knn": [float(x) for x in sonde[1:]],
        "minuti": round((time.time() - t0) / 60, 1),
    }


if __name__ == "__main__":
    os.makedirs(LOGS, exist_ok=True)
    scelte = sys.argv[1:] or [c[0] for c in CONFIG]
    esiti = []
    for nome, ema, lr, pred in CONFIG:
        if nome not in scelte:
            continue
        print(f"\n=== {nome}: ema={ema} lr={lr:.0e} pred={pred} ===", flush=True)
        r = lancia(nome, ema, lr, pred)
        esiti.append(r)
        best = max(r["knn"]) if r["knn"] else float("nan")
        rif = r["riferimento"]
        print(f"    k-NN {r['knn']}  miglior {best:.4f}  "
              f"riferimento {rif}  -> {best - rif:+.4f}  ({r['minuti']} min)",
              flush=True)
        with open(os.path.join("runs", "sweep_collasso.json"), "w",
                  encoding="utf-8") as f:
            json.dump(esiti, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'config':10s} {'ema':8s} {'lr':9s} {'pred':5s} {'miglior k-NN':>13s} {'delta':>8s}")
    print("=" * 70)
    for r in sorted(esiti, key=lambda x: -(max(x["knn"]) if x["knn"] else 0)):
        best = max(r["knn"]) if r["knn"] else float("nan")
        print(f"{r['nome']:10s} {r['ema']:<8} {r['lr']:<9.0e} {r['pred_dim']:<5} "
              f"{best:13.4f} {best - r['riferimento']:+8.4f}")
    print("\nUna configurazione con delta positivo batte l'encoder casuale:")
    print("quella va portata a 300 epoche. Se nessuna lo fa, il problema non")
    print("e' negli iperparametri e va cambiato il compito, non la taratura.")
