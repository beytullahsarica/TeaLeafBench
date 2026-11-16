"""Classification metrics. Macro averages are the headline numbers: tea disease datasets are imbalanced."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score, top_k_accuracy_score)


def classification_metrics(probs, labels, classes=None):
    """probs (N, C) softmax outputs, labels (N,) ints -> dict of scalars (+ per-class lists and confusion matrix)."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    n_cls = probs.shape[1]
    idx = np.arange(n_cls)
    pred = probs.argmax(1)
    p, r, f, support = precision_recall_fscore_support(labels, pred, labels=idx, zero_division=0)
    m = {
        'accuracy': accuracy_score(labels, pred),
        'balanced_accuracy': balanced_accuracy_score(labels, pred),
        'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
        'weighted_f1': f1_score(labels, pred, labels=idx, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(labels, pred, labels=idx),
        'mcc': matthews_corrcoef(labels, pred),
        'nll': log_loss(labels, _normalise(probs), labels=idx),
        'ece': expected_calibration_error(probs, labels),
        'n': int(len(labels)),
    }
    if n_cls > 3:
        m['top3_accuracy'] = top_k_accuracy_score(labels, probs, k=3, labels=idx)
    present = np.unique(labels)
    try:          # needs every class present in labels
        if len(present) == n_cls:
            m['macro_auc'] = roc_auc_score(labels, _normalise(probs), multi_class='ovr', average='macro', labels=idx) \
                if n_cls > 2 else roc_auc_score(labels, probs[:, 1])
    except ValueError:
        pass
    m = {k: float(v) if k != 'n' else v for k, v in m.items()}
    m['per_class'] = {'class': list(classes) if classes else [str(i) for i in idx],
                      'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(), 'support': support.tolist()}
    m['confusion_matrix'] = confusion_matrix(labels, pred, labels=idx).tolist()
    return m


def _normalise(probs):
    """Rows summing exactly to 1 (float32 softmax / averaged TTA views drift by ~1e-7; sklearn >= 1.5 rejects that)."""
    p = np.clip(probs, 1e-12, 1.0)
    return p / p.sum(1, keepdims=True)


def scalar_metrics(m):
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)
