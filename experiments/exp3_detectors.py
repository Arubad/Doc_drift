"""Experiment 3 -- Detector comparison (function-level split).

Five detectors, from a zero-training heuristic to a hybrid TF-IDF+structural
gradient boosting model. Split is *grouped by function uid* (all variants of a
function go to the same fold), 5-fold. Reports AUROC / AUPRC / F1 /
recall@5%FPR overall and per mutation kind, plus false-positive rate on the
benign hard negatives.

Hypothesis H1: structural-alignment features (32 numbers) match or beat a
20k-dim bag-of-tokens model, because the task is about *interface agreement*
rather than content.
Hypothesis H2: semantic mutations are near chance for every detector -> the
benchmark exposes the ceiling of static, execution-free drift detection.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import load_benchmark, save_csv
from docdrift.models import DETECTORS
from docdrift.metrics import summarise, recall_at_fpr

bench = load_benchmark()
print(f"benchmark rows: {len(bench)}  functions: {bench.uid.nunique()}")
gkf = GroupKFold(n_splits=5)
rows, per_kind = [], []
preds = {}
for dname, cls in DETECTORS.items():
    p_all = np.zeros(len(bench))
    for fold, (tr, te) in enumerate(gkf.split(bench, bench.label, groups=bench.uid)):
        d = cls()
        d.fit(bench.source.values[tr], bench.docstring.values[tr], bench.label.values[tr])
        p_all[te] = d.predict_proba(bench.source.values[te], bench.docstring.values[te])
    preds[dname] = p_all
    y = bench.label.values
    s = summarise(y, p_all)
    s["recall@5%fpr"] = recall_at_fpr(y, p_all, 0.05)
    s["fpr_benign"] = float((p_all[bench.kind.values == "benign"] >= 0.5).mean())
    s["fpr_clean"] = float((p_all[bench.kind.values == "clean"] >= 0.5).mean())
    rows.append({"detector": dname, **s})
    # per mutation kind: positives of that kind vs all negatives
    neg = bench.label.values == 0
    for kind in ["surface", "behavioural", "semantic"]:
        m = (bench.kind.values == kind) | neg
        sk = summarise(y[m], p_all[m])
        sk["recall@5%fpr"] = recall_at_fpr(y[m], p_all[m], 0.05)
        per_kind.append({"detector": dname, "kind": kind, **sk})
    for mut in sorted(bench.mutation.unique()):
        if mut in ("none", "rename_local", "rewrite_augassign", "add_noop"):
            continue
        m = (bench.mutation.values == mut) | neg
        per_kind.append({"detector": dname, "kind": mut, **summarise(y[m], p_all[m]),
                         "recall@5%fpr": recall_at_fpr(y[m], p_all[m], 0.05)})
    print(dname, {k: round(v, 3) for k, v in s.items()})

res = pd.DataFrame(rows); save_csv(res, "exp3_detectors.csv")
pk = pd.DataFrame(per_kind); save_csv(pk, "exp3_per_mutation.csv")
pd.DataFrame(preds).assign(uid=bench.uid.values, mutation=bench.mutation.values, label=bench.label.values,
                           kind=bench.kind.values).to_pickle("results/exp3_predictions.pkl")
pd.set_option("display.width", 200)
print(res.round(3).to_string())
print(pk.pivot_table(index="kind", columns="detector", values="auroc").round(3).to_string())
