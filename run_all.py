"""
PIPELINE — esegue l'intera pipeline in un comando.

Pipeline completa, da zero, in un comando solo.

    python run_all.py                 # tutto
    python run_all.py --ssl-epochs 60 # piu' corta
    python run_all.py --skip-ssl      # riusa il checkpoint SSL esistente

Esegue gli stadi IN SEQUENZA. Non lanciateli in parallelo: su una sola GPU
si affamano a vicenda e il totale peggiora invece di migliorare - misurato, due job insieme portavano l'epoca SSL da 50 s a oltre 100 s e la
griglia a una combinazione all'ora.

Stadi:
  0. verifica dati, split e vincolo paziente (il brief lo richiede)
  1. pre-training SSL (obiettivo 1)
  2. caching dei latenti per i tre bracci di confronto (sez.9 dell'analisi)
  3. griglia sbilanciamento x testa x seed sui tre bracci (obiettivi 2-4)
  4. ablation su alpha della novita' (obiettivo 4)
  5. verifica finale: ogni numero ricalcolato dai file salvati
"""

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")


def stadio(nome, args, log):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, log)
    print(f"\n{'=' * 70}\n[{time.strftime('%H:%M:%S')}] {nome}\n"
          f"  {' '.join(args)}\n  log -> logs/{log}\n{'=' * 70}", flush=True)
    t = time.time()
    env = dict(os.environ)
    # Un thread BLAS per processo: con 16 worker su 32 core, i pool di thread
    # annidati esaurivano la memoria (OpenBLAS "Memory allocation failed").
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(path, "w", encoding="utf-8") as f:
        p = subprocess.run([PY, "-u"] + args, stdout=f, stderr=subprocess.STDOUT,
                           cwd=ROOT, env=env)
    dt = time.time() - t
    stato = "ok" if p.returncode == 0 else f"FALLITO (codice {p.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {nome}: {stato} in {dt/60:.1f} min", flush=True)
    if p.returncode != 0:
        print(f"  ultime righe di logs/{log}:")
        with open(path, encoding="utf-8", errors="replace") as f:
            for r in f.readlines()[-15:]:
                print("   ", r.rstrip())
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--ssl-epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--context-scale", type=float, nargs=2, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ema-start", type=float, default=None)
    ap.add_argument("--predictor-dim", type=int, default=None)
    ap.add_argument("--skip-ssl", action="store_true")
    # Senza questo, la pipeline "in un comando" rigenerava tutto col
    # protocollo a singolo layer, producendo in silenzio numeri diversi da
    # quelli degli esperimenti. Il default replica il protocollo in uso.
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    a = ap.parse_args()

    t0 = time.time()
    print(f"Pipeline avviata {time.strftime('%Y-%m-%d %H:%M:%S')}  variante={a.variant}")

    # --- 0. dati, split, vincolo paziente
    stadio("0/5 verifica dati e split", ["data.py", "--splits"], "00_dati.log")

    # --- 1. SSL
    if not a.skip_ssl:
        args = ["train_ssl.py", "--variant", a.variant,
                "--epochs", str(a.ssl_epochs), "--batch-size", str(a.batch_size)]
        if a.context_scale:
            args += ["--context-scale", str(a.context_scale[0]), str(a.context_scale[1])]
        if a.tag:
            args += ["--tag", a.tag]
        # Gli iperparametri scelti dallo sweep anti-collasso viaggiano con
        # la pipeline: senza, run_all lancerebbe il pre-training con i
        # default di globals.py, cioe' proprio la configurazione che
        # collassa (ema 0.996 -> k-NN 0.4354 contro 0.7030 del casuale).
        for flag, val in (("--lr", a.lr), ("--ema-start", a.ema_start),
                          ("--predictor-dim", a.predictor_dim)):
            if val is not None:
                args += [flag, str(val)]
        if not stadio("1/5 pre-training SSL", args, "01_ssl.log"):
            print("\nSSL fallito: gli stadi successivi userebbero un encoder non valido.")
            return 1

    # --- 2. caching
    # Il tag deve arrivare fin qui: lo stadio 1 salva in
    # ijepa_{variant}_{tag}.pt, e senza --ckpt-tag il caching cercherebbe
    # ijepa_{variant}.pt - un checkpoint diverso, o inesistente. La catena
    # si romperebbe a meta' notte, o peggio userebbe l'encoder sbagliato
    # senza dirlo.
    stadio("2/5 caching latenti",
           ["train_downstream.py", "--cache", "--variant", a.variant]
           + (["--ckpt-tag", a.tag] if a.tag else [])
           + (["--layers"] + [str(x) for x in a.layers] if a.layers else []),
           "02_cache.log")

    # --- 3. griglia
    stadio("3/5 griglia sbilanciamento",
           ["train_downstream.py", "--grid", "--variant", a.variant]
           + (["--layers"] + [str(x) for x in a.layers] if a.layers else []),
           "03_grid.log")

    # --- 4. ablation della novita'
    stadio("4/5 ablation alpha (novita')",
           ["train_downstream.py", "--sweep-alpha", "--variant", a.variant]
           + (["--layers"] + [str(x) for x in a.layers] if a.layers else []),
           "04_sweep_alpha.log")

    # --- 5. verifica
    stadio("5/5 verifica dei numeri", ["verify_claims.py", "--variant", a.variant],
           "05_verifica.log")

    print(f"\n{'=' * 70}")
    print(f"Pipeline completata in {(time.time() - t0)/60:.1f} min")
    print("Tabella finale:")
    with open(os.path.join(LOGS, "05_verifica.log"), encoding="utf-8", errors="replace") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
