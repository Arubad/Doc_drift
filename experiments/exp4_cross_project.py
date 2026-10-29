"""Experiment 4 -- Cross-project generalisation.

Grouped split by *project* (top-level package): train on some packages, test
on packages never seen. Documentation conventions differ across packages
(:param: style, Google style, bare prose), so this measures whether detectors
learn alignment or package-specific vocabulary.

Hypothesis H3: the TF-IDF model drops substantially under project shift while
the structural model is stable, because alignment features are
convention-agnostic.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import load_benchmark, save_csv
from docdrift.models import DETECTORS
from docdrift.metrics import summarise, recall_at_fpr

bench = load_benchmark()
gkf = GroupKFold(n_splits=5)
rows = []
for dname, cls in DETECTORS.items():
    if dname in ("heuristic", "hybrid-GBM"):
        continue
    for fold, (tr, te) in enumerate(gkf.split(bench, bench.label, groups=bench.project)):
        d = cls().fit(bench.source.values[tr], bench.docstring.values[tr], bench.label.values[tr])
        p = d.predict_proba(bench.source.values[te], bench.docstring.values[te])
        y = bench.label.values[te]
        s = summarise(y, p); s["recall@5%fpr"] = recall_at_fpr(y, p, 0.05)
        rows.append({"detector": dname, "split": "cross-project", "fold": fold, **s})
    for fold, (tr, te) in enumerate(gkf.split(bench, bench.label, groups=bench.uid)):
        d = cls().fit(bench.source.values[tr], bench.docstring.values[tr], bench.label.values[tr])
        p = d.predict_proba(bench.source.values[te], bench.docstring.values[te])
        y = bench.label.values[te]
        s = summarise(y, p); s["recall@5%fpr"] = recall_at_fpr(y, p, 0.05)
        rows.append({"detector": dname, "split": "within-project", "fold": fold, **s})
    print("done", dname)

res = pd.DataFrame(rows)
save_csv(res, "exp4_cross_project.csv")
summ = res.groupby(["detector", "split"])[["auroc", "auprc", "f1", "recall@5%fpr"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]
save_csv(summ.reset_index(), "exp4_summary.csv")
pd.set_option("display.width", 200)
print(summ.round(3).to_string())
