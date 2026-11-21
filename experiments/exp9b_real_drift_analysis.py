"""Experiment 9b -- Secondary analyses of real-drift results (no retraining).

Uses data/exp9_real_scores.pkl (per-row scores of every variant from exp9),
data/real_pairs_clean.pkl and the manual annotation file.

* H3: pair accuracy per edit category (param / return / raise / type / default / prose).
* Annotation-verified subsets: pairs annotated factual / incomplete (real drift)
  vs clarity / noise (not drift). On not-drift pairs both docstrings are
  acceptable, so ~0.5 is the expected accuracy of any detector.
* Ceiling: with label precision p, a perfect detector scores p + (1 - p)/2.
* Figure: synthetic metric vs real pair accuracy, one point per variant.

Annotation file: $DOCDRIFT_ANNOTATIONS (default ../runs/real_drift/exp8b_annotation_A.csv).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, RESULTS, save_csv, fig_path

ANN = os.environ.get("DOCDRIFT_ANNOTATIONS",
                     os.path.join(os.path.dirname(__file__), "..", "runs", "real_drift", "exp8b_annotation_A.csv"))
real = pd.read_pickle(os.path.join(DATA, "real_pairs_clean.pkl"))
scores = pd.read_pickle(os.path.join(DATA, "exp9_real_scores.pkl"))
variants = pd.read_csv(os.path.join(RESULTS, "exp9_variants.csv"))
n = len(real)
# exp9 stacks rows as [old (label 1) for each pair] + [new (label 0) for each pair]
pid = real.pair_id.values


def correct(score):
    d = score[:n] - score[n:]
    return pd.Series((d > 0) + 0.5 * (d == 0), index=pid)


C = pd.DataFrame({v: correct(s) for v, s in scores.items()})       # pairs x variants

# ---- H3: per edit category
rows = []
for cat in ["param", "return", "raise", "type", "default", "prose"]:
    ids = real.pair_id[real.categories.map(lambda c: cat in c)].values
    for v in C.columns:
        rows.append({"category": cat, "variant": v, "n_pairs": len(ids), "pair_acc": float(C.loc[ids, v].mean())})
per_cat = pd.DataFrame(rows)
save_csv(per_cat, "exp9b_per_category.csv")
print(per_cat.pivot_table(index="variant", columns="category", values="pair_acc").round(3).to_string())

# ---- annotation-verified subsets + ceilings
if os.path.exists(ANN):
    ann = pd.read_csv(ANN).merge(real[["pair_id", "interface"]], on="pair_id")
    ann["drift"] = ann.annotation.isin(["factual", "incomplete"])
    rows = []
    for name, m in [("annotated drift", ann.drift), ("annotated factual", ann.annotation == "factual"),
                    ("annotated not-drift", ~ann.drift)]:
        ids = ann.pair_id[m].values
        for v in C.columns:
            rows.append({"subset": name, "variant": v, "n_pairs": len(ids), "pair_acc": float(C.loc[ids, v].mean())})
    ann_res = pd.DataFrame(rows)
    save_csv(ann_res, "exp9b_annotated.csv")
    print(ann_res.pivot_table(index="variant", columns="subset", values="pair_acc").round(3).to_string())
    ceil = []
    for name, m in [("all", np.ones(len(ann), bool)), ("interface", ann.interface.values)]:
        p = ann.drift[m].mean()
        ceil.append({"subset": name, "annotated": int(m.sum()), "label_precision": p, "perfect_detector_pair_acc": p + (1 - p) / 2})
    save_csv(pd.DataFrame(ceil), "exp9b_ceiling.csv")
    print(pd.DataFrame(ceil).round(3).to_string(index=False))

# ---- figure: synthetic metric vs real accuracy
V = variants[variants.family != "heuristic"]
fam_marker = {"struct-LR": "o", "struct-GBM": "s", "tfidf-LR": "^", "hybrid-GBM": "D"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, col, title in [(axes[0], "indist_auroc", "in-distribution AUROC"), (axes[1], "lomo_auroc", "LOMO AUROC")]:
    for fam, g in V.groupby("family"):
        ax.scatter(g[col], g.real_pair_acc_interface, marker=fam_marker[fam], label=fam, s=40, alpha=.8)
    rho = spearmanr(V[col], V.real_pair_acc_interface).statistic
    ax.axhline(0.5, color="grey", lw=.8, ls=":")
    ax.set_xlabel(f"synthetic {title}"); ax.set_title(f"Spearman ρ = {rho:.2f}")
axes[0].set_ylabel("real doc-fix pair accuracy (interface subset)")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(fig_path("exp9_synthetic_vs_real.png"), dpi=150)
print("[saved] exp9_synthetic_vs_real.png")

# ---- mechanism: how do structural features change from the old to the new docstring?
# Synthetic drift only ever edits CODE; real fixes edit the DOCSTRING. Compare feature deltas
# (new - old) with the direction each feature pushes the struct-LR|all model.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_benchmark
from docdrift.features import extract, FEATURE_NAMES

bench = load_benchmark()
Fb = np.stack([extract(s, d) for s, d in zip(bench.source, bench.docstring)])
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Fb, bench.label.values)
coef = m[-1].coef_[0] / m[0].scale_                       # per raw-feature-unit effect on logit
Fo = np.stack([extract(s, d) for s, d in zip(real.source_old, real.doc_old)])
Fn = np.stack([extract(s, d) for s, d in zip(real.source_new, real.doc_new)])
delta = (Fn - Fo).mean(0)
mech = pd.DataFrame({"feature": FEATURE_NAMES, "mean_new_minus_old": delta, "coef_per_unit": coef,
                     "logit_contribution": delta * coef}).sort_values("logit_contribution", key=abs, ascending=False)
save_csv(mech, "exp9b_mechanism.csv")
print("\nmean logit(new) - logit(old) under struct-LR|all:", round(float((delta * coef).sum()), 3),
      "(positive = new docstring looks MORE drifted)")
print(mech.head(10).round(3).to_string(index=False))
