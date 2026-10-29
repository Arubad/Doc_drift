"""Experiment 3b -- Which negatives are the detectors separating from?

Re-scores the Exp 3 out-of-fold predictions with two negative sets:
 * clean originals only (the realistic deployment negative), and
 * benign-edit copies only (hard negatives that carry *their own* generator
   fingerprints, e.g. the renamed-local name pool).
A large gap between the two AUROCs means the detector has learned the benign
operators' fingerprints as a *negative* cue -- artifact learning on the
negative side, the mirror image of what Exp 7 measures on the positive side.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import save_csv

pred = pd.read_pickle("results/exp3_predictions.pkl")
dets = [c for c in pred.columns if c not in ("uid", "mutation", "label", "kind")]
rows = []
for d in dets:
    pos = pred[pred.label == 1]
    for neg_name in ["clean", "benign"]:
        neg = pred[pred.kind == neg_name]
        y = [1] * len(pos) + [0] * len(neg)
        p = list(pos[d]) + list(neg[d])
        rows.append({"detector": d, "negatives": neg_name, "auroc": roc_auc_score(y, p)})
res = pd.DataFrame(rows).pivot(index="detector", columns="negatives", values="auroc").reset_index()
res["benign_minus_clean"] = res.benign - res.clean
save_csv(res, "exp3b_negative_type_auroc.csv")
print(res.round(3).to_string())
