"""Experiment 7 -- Leave-One-Mutation-Out (LOMO): do detectors learn
*alignment* or *mutation artifacts*?

Mutation-generated benchmarks (ours and the JIT comment-inconsistency
literature) risk teaching detectors the fingerprints of the mutation
operators (e.g. the literal "must not be None" inserted by add_raise, the
wrapper call inserted by wrap_return) instead of genuine docstring/code
disagreement. LOMO trains on all mutation types but one and tests on the
held-out type (positives) vs all negatives. A detector that learned
alignment transfers; one that memorised artifacts collapses.

We also report a "cross-kind" protocol: train on surface mutations only,
test on behavioural ones (and vice versa).

Hypothesis H5: the bag-of-tokens model loses far more under LOMO than the
structural model, and its in-distribution advantage (Exp 3) is largely
artifact learning. This makes LOMO AUROC the metric to optimise for
practical drift detectors.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import load_benchmark, save_csv
from docdrift.models import StructuralGBM, TfidfLR, Hybrid, StructuralLR
from docdrift.metrics import summarise, recall_at_fpr
from docdrift.mutations import SURFACE, BEHAVIOURAL, SEMANTIC

bench = load_benchmark()
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr, te = next(gss.split(bench, bench.label, groups=bench.uid))
S, D, Y, M = bench.source.values, bench.docstring.values, bench.label.values, bench.mutation.values
neg_te = te[Y[te] == 0]
rows = []
muts = sorted(SURFACE | BEHAVIOURAL | SEMANTIC)
for cls in [StructuralLR, StructuralGBM, TfidfLR]:
    # in-distribution reference (same split)
    d = cls().fit(S[tr], D[tr], Y[tr])
    p_te = np.full(len(bench), np.nan); p_te[te] = d.predict_proba(S[te], D[te])
    for mut in muts:
        pos = te[M[te] == mut]
        idx = np.r_[pos, neg_te]
        s = summarise(Y[idx], p_te[idx]); s["recall@5%fpr"] = recall_at_fpr(Y[idx], p_te[idx], 0.05)
        rows.append({"detector": cls.name, "protocol": "in-dist", "held_out": mut, **s})
    # LOMO
    for mut in muts:
        keep = tr[M[tr] != mut]
        d = cls().fit(S[keep], D[keep], Y[keep])
        pos = te[M[te] == mut]
        idx = np.r_[pos, neg_te]
        p = d.predict_proba(S[idx], D[idx])
        s = summarise(Y[idx], p); s["recall@5%fpr"] = recall_at_fpr(Y[idx], p, 0.05)
        rows.append({"detector": cls.name, "protocol": "LOMO", "held_out": mut, **s})
    # cross-kind
    for name, train_kinds, test_kinds in [("surface->behavioural", SURFACE, BEHAVIOURAL),
                                          ("behavioural->surface", BEHAVIOURAL, SURFACE),
                                          ("surface+behavioural->semantic", SURFACE | BEHAVIOURAL, SEMANTIC)]:
        keep = tr[(Y[tr] == 0) | np.isin(M[tr], list(train_kinds))]
        d = cls().fit(S[keep], D[keep], Y[keep])
        pos = te[np.isin(M[te], list(test_kinds))]
        idx = np.r_[pos, neg_te]
        p = d.predict_proba(S[idx], D[idx])
        s = summarise(Y[idx], p); s["recall@5%fpr"] = recall_at_fpr(Y[idx], p, 0.05)
        rows.append({"detector": cls.name, "protocol": "cross-kind", "held_out": name, **s})
    print("done", cls.name)

res = pd.DataFrame(rows)
save_csv(res, "exp7_lomo.csv")
pd.set_option("display.width", 220)
piv = res[res.protocol.isin(["in-dist", "LOMO"])].pivot_table(index="held_out", columns=["detector", "protocol"], values="auroc")
print(piv.round(3).to_string())
drop = res[res.protocol == "LOMO"].groupby("detector").auroc.mean() - res[res.protocol == "in-dist"].groupby("detector").auroc.mean()
print("\nmean AUROC change in-dist -> LOMO:\n", drop.round(3))
print(res[res.protocol == "cross-kind"].pivot_table(index="held_out", columns="detector", values="auroc").round(3).to_string())
save_csv(piv.reset_index(), "exp7_lomo_pivot.csv")
