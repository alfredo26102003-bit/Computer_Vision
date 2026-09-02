"""
PIPELINE — metriche, verificate contro scikit-learn a 4.4e-16.

Evaluation - metriche per il grading PAI sbilanciato e ordinale.

Sezione "Evaluation" della struttura richiesta dal corso.

Il brief richiede metriche threshold-agnostic sulla minoritaria - Macro-F1,
Precision-Recall AUC, confusion matrix - e implicitamente vieta l'accuracy
globale. Ha ragione: con il 61% di PAI 3, un classificatore che predice
sempre PAI 3 ottiene 61% di accuracy e zero utilita' clinica.

In piu' c'e' il kappa quadratico pesato, che il brief non chiede ma che e'
la metrica giusta per una scala ORDINALE: confondere PAI 3 con PAI 5 e'
clinicamente peggio che confondere PAI 4 con PAI 5, e la Macro-F1 li pesa
uguale.
"""

import numpy as np
import torch

from globals import DEVICE, FIG_DIR, NUM_CLASSES, PAI_GRADES, e_ordinale


# ==========================================================================
# Metriche di base
# ==========================================================================
def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm):
    """Precision, recall e F1 per classe dalla confusion matrix."""
    out = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"precision": prec, "recall": rec, "f1": f1, "support": int(cm[c].sum())})
    return out


def macro_f1(cm):
    return float(np.mean([m["f1"] for m in per_class_prf(cm)]))


def balanced_accuracy(cm):
    return float(np.mean([m["recall"] for m in per_class_prf(cm)]))


def pr_auc(scores, binary_labels):
    """
    Area sotto la curva precision-recall, per una classe contro il resto.

    Threshold-agnostic e sensibile alla minoritaria: e' la metrica giusta
    per PAI 5, dove la ROC-AUC sarebbe ottimisticamente alta solo perche'
    i negativi sono tantissimi.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(binary_labels, dtype=int)
    if y.sum() == 0:
        return float("nan")

    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def pr_curve(scores, binary_labels):
    """
    La curva precision-recall di cui `pr_auc` restituisce solo l'area.

    L'area comprime la curva in un numero, e due metodi con la stessa area
    possono avere forme molto diverse: uno preciso ad alta soglia e inutile
    a recall alta, l'altro il contrario. Su una minoritaria clinica la forma
    conta piu' dell'area, perche' il punto di lavoro utile non e' "tutta la
    curva" ma "la precisione che resta quando si pretende di trovare l'80%
    dei PAI 5".

    Restituisce (recall, precision) crescenti in recall, con il punto
    iniziale (0, 1) aggiunto perche' la curva parta dall'asse e le aree
    sotto curve diverse siano confrontabili a vista.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(binary_labels, dtype=int)
    if y.sum() == 0:
        return np.array([0.0]), np.array([1.0])

    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    return (np.concatenate([[0.0], recall]),
            np.concatenate([[1.0], precision]))


def curva_media(curve, griglia=None):
    """
    Media di piu' curve PR sui seed, interpolata su una griglia di recall.

    Mediare le curve punto per punto e' impossibile: ogni seed produce un
    numero diverso di punti, a valori di recall diversi. Si interpola quindi
    ciascuna su una griglia comune di recall e si media in verticale, che e'
    il modo standard (vertical averaging) e l'unico che conserva il
    significato dell'asse: "a questa recall, questa precisione".

    Restituisce (griglia, media, deviazione).
    """
    if griglia is None:
        griglia = np.linspace(0.0, 1.0, 201)
    righe = []
    for rec, prec in curve:
        # `np.interp` pretende x crescente. La recall lo e' gia' per
        # costruzione, ma con punteggi a pari merito puo' ripetersi: si
        # tiene per ogni recall la precisione MASSIMA, che e' la
        # convenzione della curva PR interpolata.
        r, p = np.asarray(rec), np.asarray(prec)
        ordine = np.argsort(r, kind="stable")
        r, p = r[ordine], p[ordine]
        righe.append(np.interp(griglia, r, p))
    m = np.stack(righe)
    return griglia, m.mean(0), m.std(0)


def quadratic_weighted_kappa(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Cohen's kappa con pesi quadratici - la metrica corretta per il PAI.

    Penalizza gli errori in proporzione al QUADRATO della distanza sulla
    scala: sbagliare di due gradi (3 -> 5) costa quattro volte sbagliare di
    uno. E' esattamente la struttura di gravita' clinica del problema, e la
    Macro-F1 non la coglie.

    1.0 = accordo perfetto, 0.0 = accordo casuale, < 0 = peggio del caso.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")

    i, j = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing="ij")
    w = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom > 0 else float("nan")


def ordinal_mae(y_true, y_pred):
    """Errore assoluto medio sulla scala ordinale, in gradi PAI."""
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())


# ==========================================================================
# Valutazione di un modello
# ==========================================================================
@torch.no_grad()
def evaluate_split(clf, split_data, head_type="flat", batch_size=256,
                   use_geom=False, con_punteggi=False):
    """
    Valuta la testa su uno split di latenti cachati.

    `con_punteggi` aggiunge al risultato i punteggi grezzi per classe e le
    etichette vere. Servono a disegnare la curva PR, che dai soli aggregati
    non si puo' ricostruire: l'area e' un integrale e non si inverte.
    Restano fuori per difetto perche' sono un array per lesione e
    finirebbero dentro i JSON dell'ablation, moltiplicandone la dimensione
    per nessun uso.
    """
    clf.eval()
    tokens, mask, labels = split_data["tokens"], split_data["mask"], split_data["labels"]

    probs, preds = [], []
    for i in range(0, len(labels), batch_size):
        tok = tokens[i:i + batch_size].float().to(DEVICE)
        msk = mask[i:i + batch_size].to(DEVICE)
        gm = (split_data["geom"][i:i + batch_size].to(DEVICE)
              if use_geom else None)
        logits, _, _ = clf(tok, token_mask=msk, geom=gm)

        if e_ordinale(head_type):
            cum = torch.sigmoid(logits)                    # (B, K-1)
            pred = (cum > 0.5).sum(dim=1)
            # probabilita' per classe dai cumulativi: P(k) = P(>k-1) - P(>k)
            p = torch.zeros(cum.shape[0], NUM_CLASSES, device=cum.device)
            p[:, 0] = 1 - cum[:, 0]
            for k in range(1, NUM_CLASSES - 1):
                p[:, k] = cum[:, k - 1] - cum[:, k]
            p[:, -1] = cum[:, -1]
            p = p.clamp(min=0)
        else:
            p = torch.softmax(logits, dim=-1)
            pred = p.argmax(dim=-1)

        probs.append(p.cpu())
        preds.append(pred.cpu())

    probs = torch.cat(probs).numpy()
    preds = torch.cat(preds).numpy()
    y = labels.numpy()
    cm = confusion_matrix(y, preds)
    pc = per_class_prf(cm)

    res = {
        "macro_f1": macro_f1(cm),
        "balanced_acc": balanced_accuracy(cm),
        "accuracy": float((preds == y).mean()),
        "quadratic_kappa": quadratic_weighted_kappa(y, preds),
        "ordinal_mae": ordinal_mae(y, preds),
        "confusion_matrix": cm.tolist(),
    }
    for c, g in enumerate(PAI_GRADES):
        res[f"recall_pai{g}"] = pc[c]["recall"]
        res[f"precision_pai{g}"] = pc[c]["precision"]
        res[f"f1_pai{g}"] = pc[c]["f1"]
        res[f"pr_auc_pai{g}"] = pr_auc(probs[:, c], (y == c).astype(int))
    if con_punteggi:
        res["scores"] = probs
        res["y_true"] = y
    return res


def print_report(res, title=""):
    print(f"\n=== {title} ===")
    print(f"  Macro-F1            : {res['macro_f1']:.4f}")
    print(f"  Balanced accuracy   : {res['balanced_acc']:.4f}")
    print(f"  Kappa quadratico    : {res['quadratic_kappa']:.4f}")
    print(f"  MAE ordinale        : {res['ordinal_mae']:.4f}")
    print(f"  Accuracy globale    : {res['accuracy']:.4f}  "
          f"(NON riportatela da sola: predire sempre PAI 3 da' 0.612)")

    print(f"\n  {'classe':8s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'PR-AUC':>8s}")
    for g in PAI_GRADES:
        print(f"  PAI {g:<4d} {res[f'precision_pai{g}']:7.4f} "
              f"{res[f'recall_pai{g}']:7.4f} {res[f'f1_pai{g}']:7.4f} "
              f"{res[f'pr_auc_pai{g}']:8.4f}")

    cm = np.array(res["confusion_matrix"])
    print("\n  Confusion matrix (righe = vero, colonne = predetto)")
    print("            " + "".join(f"PAI{g:<5d}" for g in PAI_GRADES))
    for c, g in enumerate(PAI_GRADES):
        print(f"    PAI {g}   " + "".join(f"{v:<8d}" for v in cm[c]))

    off2 = cm[0, 2] + cm[2, 0]
    if off2:
        print(f"\n  Errori a due gradi (3<->5): {off2} - sono i clinicamente")
        print("  piu' gravi, ed e' il motivo per cui c'e' il kappa quadratico.")


def plot_confusion(res, name="confusion_matrix"):
    import os
    import matplotlib.pyplot as plt

    cm = np.array(res["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(PAI_GRADES)), [f"PAI {g}" for g in PAI_GRADES])
    ax.set_yticks(range(len(PAI_GRADES)), [f"PAI {g}" for g in PAI_GRADES])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Vero")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    # verifica delle metriche su casi con risultato noto
    print("=== Test delle metriche ===")

    y = np.array([0] * 60 + [1] * 30 + [2] * 10)

    perfetto = y.copy()
    cm = confusion_matrix(y, perfetto)
    print(f"\nPredizione perfetta:")
    print(f"  macro-F1 = {macro_f1(cm):.3f} (atteso 1.000)")
    print(f"  kappa    = {quadratic_weighted_kappa(y, perfetto):.3f} (atteso 1.000)")

    sempre_maggioritaria = np.zeros_like(y)
    cm = confusion_matrix(y, sempre_maggioritaria)
    print(f"\nPredice sempre la maggioritaria (il fallimento tipico):")
    print(f"  accuracy = {(sempre_maggioritaria == y).mean():.3f}  <- sembra buona")
    print(f"  macro-F1 = {macro_f1(cm):.3f}  <- rivela il problema")
    print(f"  kappa    = {quadratic_weighted_kappa(y, sempre_maggioritaria):.3f} (atteso 0.000)")

    # il kappa quadratico distingue errori vicini da errori lontani
    vicino = y.copy(); vicino[y == 2] = 1     # PAI 5 -> PAI 4, sbaglia di 1
    lontano = y.copy(); lontano[y == 2] = 0   # PAI 5 -> PAI 3, sbaglia di 2
    print(f"\nStesso numero di errori, gravita' diversa:")
    print(f"  PAI5 -> PAI4 (1 grado) : macro-F1={macro_f1(confusion_matrix(y,vicino)):.3f}  "
          f"kappa={quadratic_weighted_kappa(y, vicino):.3f}")
    print(f"  PAI5 -> PAI3 (2 gradi) : macro-F1={macro_f1(confusion_matrix(y,lontano)):.3f}  "
          f"kappa={quadratic_weighted_kappa(y, lontano):.3f}")
    print("  -> il kappa quadratico penalizza di piu' l'errore clinicamente peggiore")

    print(f"\nPR-AUC separazione perfetta = "
          f"{pr_auc(np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(10), np.zeros(90)]):.3f} (atteso 1.000)")
