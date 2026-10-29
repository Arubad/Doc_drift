from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def summarise(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p)
    out = {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
           "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan}
    yhat = (p >= thr).astype(int)
    out["f1"] = float(f1_score(y, yhat, zero_division=0))
    out["precision"] = float(precision_score(y, yhat, zero_division=0))
    out["recall"] = float(recall_score(y, yhat, zero_division=0))
    return out


def recall_at_fpr(y, p, fpr=0.05):
    """Recall achieved when the threshold is set so that FPR on negatives == fpr."""
    y = np.asarray(y); p = np.asarray(p)
    neg = p[y == 0]
    if len(neg) == 0:
        return np.nan
    thr = np.quantile(neg, 1 - fpr)
    return float((p[y == 1] >= thr).mean())
