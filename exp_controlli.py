"""
CONFRONTO — PR-AUC PAI 5, la novita' contro i suoi controlli a budget uguale.

I due controlli che attribuiscono il risultato della novita' - obiettivo 4.

La griglia dice che `balanced_tokens` ha la PR-AUC piu' alta sull'encoder
casuale. Non dice PERCHE', e ci sono due spiegazioni alternative che quella
misura da sola non esclude. Un'ablation che non le esclude non ha ablato
niente.

--------------------------------------------------------------------------
CONTROLLO A - E' il ribilanciamento o e' l'augmentation?

`balanced_token_sampling` fa due cose insieme:
  (a) genera viste, cioe' sottoinsiemi diversi di token della stessa
      lesione: e' augmentation
  (b) ne assegna di piu' alle classi rare: e' ribilanciamento
La novita' rivendica (b). Ma (a) da sola potrebbe bastare, e in quel caso
la novita' sarebbe un'augmentation con un nome altisonante.

`random_tokens` tiene (a) e toglie (b): stesso numero totale di viste,
generate dalla stessa procedura, ma distribuite senza guardare la classe.
La lettura e' netta:
  random ~ none            -> l'augmentation da sola non serve
  balanced ~ random        -> la novita' non ribilancia niente
  balanced > random > none -> entrambe contribuiscono, e si sa di quanto

--------------------------------------------------------------------------
CONTROLLO B - Ha vinto perche' ha fatto piu' passi di gradiente?

A parita' di EPOCHE i metodi non vedono lo stesso numero di istanze:

    none / class_weighted / focal / oversample   4719 per epoca
    balanced_tokens, alpha 0.50                  6894 per epoca  (+46%)
    balanced_tokens, alpha 1.00                 10015 per epoca  (+112%)

La novita' riceve il 46% di passi in piu' delle baseline con cui viene
confrontata. Puo' non essere il motivo del vantaggio, ma finche' non si
pareggia il budget non e' escluso, e non escluderlo e' esattamente il tipo
di obiezione che si prende in sede di discussione.

Qui si fissa il numero TOTALE di istanze di training e si ricavano le
epoche di conseguenza, cosi' ogni metodo vede la stessa quantita' di
gradiente. Nota bene: questo peggiora deliberatamente il confronto per la
novita', perche' le toglie un vantaggio che nella griglia aveva. Se
sopravvive, il risultato e' piu' forte di prima; se non sopravvive, era
quello.

--------------------------------------------------------------------------
Entrambi i controlli girano sull'encoder CASUALE con testa `flat`, che e'
la configurazione dove la novita' ha il suo massimo: si testa l'ipotesi
dove e' piu' favorevole, non dove e' comoda.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluation import evaluate_split
from globals import HEAD_EPOCHS, OUT_DIR
from imbalance import class_counts
from train_downstream import istanze_per_epoca, load_latents, train_head
from utils import Freno, save_json

CARICO = 70
METODI = ["none", "class_weighted", "oversample", "random_tokens",
          "balanced_tokens"]
CHIAVI = ("macro_f1", "pr_auc_pai5", "recall_pai5", "precision_pai5",
          "f1_pai5", "quadratic_kappa")


def misura(cached, method, seeds, alpha, budget, freno, head="flat"):
    per_seed = []
    for s in seeds:
        t0 = time.perf_counter()
        clf, _ = train_head(cached, method, head, seed=s, bts_alpha=alpha,
                            budget_istanze=budget)
        per_seed.append(evaluate_split(clf, cached["data"]["test"], head))
        del clf
        torch.cuda.empty_cache()
        freno.pausa(t0)
    return {k: (float(np.mean([m[k] for m in per_seed])),
                float(np.std([m[k] for m in per_seed]))) for k in CHIAVI}


def riga(nome, m, extra=""):
    return (f"  {nome:24s} {m['pr_auc_pai5'][0]:.4f}+-{m['pr_auc_pai5'][1]:.4f}"
            f"  {m['macro_f1'][0]:.4f}  {m['recall_pai5'][0]:.4f}"
            f"  {m['precision_pai5'][0]:.3f}  {m['f1_pai5'][0]:.3f}{extra}")


def intestazione():
    return (f"  {'metodo':24s} {'PR-AUC5':>13s}  {'macroF1':>6s}  "
            f"{'rec5':>6s}  {'prec5':>5s}  {'F1_5':>5s}")


def scarto(a, b, n):
    """Differenza fra due medie in unita' di errore standard combinato."""
    d = a[0] - b[0]
    se = float(np.sqrt(a[1] ** 2 + b[1] ** 2) / np.sqrt(n))
    return d, (d / se if se > 0 else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_casuale")
    ap.add_argument("--variant", default="vit_small")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 7, 11])
    ap.add_argument("--head", default="flat")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--carico", type=int, default=CARICO)
    a = ap.parse_args()

    freno = Freno(a.carico)
    print(f"[freno] {freno}", flush=True)
    cached = load_latents(a.variant, layers=a.layers, tag=a.tag)
    train_labels = cached["data"]["train"]["labels"]
    counts = class_counts(train_labels)

    print(f"\nencoder{a.tag}, testa {a.head}, alpha {a.alpha}, "
          f"{len(a.seeds)} seed, misure sul TEST")
    print(f"train PAI3/4/5 = {counts.int().tolist()}")

    print("\nIstanze viste per epoca, a parita' di epoche:")
    per_ep = {m: istanze_per_epoca(train_labels, m, a.alpha) for m in METODI}
    for m in METODI:
        print(f"  {m:24s} {per_ep[m]:8.0f}"
              f"  ({per_ep[m] / per_ep['none'] - 1:+.0%} rispetto a none)")

    # Il budget di riferimento e' quello che la novita' consumava nella
    # griglia: si pareggia verso l'ALTO, dando alle baseline le epoche che
    # servono, invece di togliere epoche alla novita'. Togliergliele
    # cambierebbe anche il suo risultato e non si saprebbe piu' a quale
    # delle due modifiche attribuire la differenza.
    budget = HEAD_EPOCHS * per_ep["balanced_tokens"]
    print(f"\nBudget comune: {budget:,.0f} istanze "
          f"(= {HEAD_EPOCHS} epoche della novita' ad alpha {a.alpha})")
    print("Epoche assegnate a ciascun metodo per raggiungerlo:")
    for m in METODI:
        print(f"  {m:24s} {round(budget / per_ep[m]):4d} epoche")

    fuori = {"tag": a.tag, "head": a.head, "alpha": a.alpha,
             "seeds": a.seeds, "budget": budget,
             "istanze_per_epoca": per_ep, "libero": {}, "pareggiato": {}}
    percorso = os.path.join(OUT_DIR, f"controlli_{a.variant}{a.tag}.json")

    # RIPARTENZA. Dieci celle da misurare, ognuna cinque addestramenti: e'
    # l'esperimento piu' lungo dei cinque, e su Colab la sessione cade da
    # sola. Si riprende solo se il protocollo coincide - seed, testa,
    # alpha: riprendere un file misurato con altri parametri mescolerebbe
    # due esperimenti in una tabella sola, in silenzio.
    if os.path.isfile(percorso):
        with open(percorso, encoding="utf-8") as f:
            vecchio = json.load(f)
        stesso = all(vecchio.get(k) == v for k, v in
                     (("seeds", a.seeds), ("head", a.head), ("alpha", a.alpha)))
        if stesso:
            fuori = vecchio
            n_fatte = len(fuori["libero"]) + len(fuori["pareggiato"])
            print(f"\nRiprendo: {n_fatte}/{2 * len(METODI)} celle gia' su disco")
        else:
            print(f"\n{os.path.basename(percorso)} esiste ma con un altro "
                  f"protocollo (seed/testa/alpha diversi): riparto da capo")

    for nome, bud in (("libero", None), ("pareggiato", budget)):
        titolo = (f"EPOCHE FISSE ({HEAD_EPOCHS}), budget diverso"
                  if bud is None else
                  "BUDGET PAREGGIATO, epoche diverse")
        print(f"\n{'=' * 78}\n{titolo}\n{'=' * 78}")
        print(intestazione())
        for m in METODI:
            ep = HEAD_EPOCHS if bud is None else round(bud / per_ep[m])
            if m in fuori[nome]:
                print(riga(m, fuori[nome][m], f"   [{ep} ep, da disco]"))
                continue
            fuori[nome][m] = misura(cached, m, a.seeds, a.alpha, bud, freno,
                                    a.head)
            print(riga(m, fuori[nome][m], f"   [{ep} ep]"), flush=True)
            save_json(fuori, percorso)

    n = len(a.seeds)
    print(f"\n{'=' * 78}\nLETTURA\n{'=' * 78}")
    for nome in ("libero", "pareggiato"):
        r = fuori[nome]
        print(f"\n{nome.upper()}  (PR-AUC su PAI 5)")
        for x, y in (("balanced_tokens", "random_tokens"),
                     ("random_tokens", "none"),
                     ("balanced_tokens", "none"),
                     ("balanced_tokens", "class_weighted"),
                     ("balanced_tokens", "oversample")):
            d, z = scarto(r[x]["pr_auc_pai5"], r[y]["pr_auc_pai5"], n)
            print(f"  {x:16s} - {y:16s} {d:+.4f}   {z:+.2f} err.std")

    print(f"\nCome leggere il primo confronto:")
    print("  balanced - random  isola il RIBILANCIAMENTO (stesso budget di viste)")
    print("  random   - none    isola l'AUGMENTATION (stesso budget, nessun")
    print("                     ribilanciamento)")
    print("  la loro somma e' balanced - none, che e' il numero della griglia.")
    print(f"\nRisultati in {percorso}")
