"""Experiment 11c (exploratory, post hoc) -- Training-data contamination check for the LLM judges.

Qwen2.5-Coder was trained on public GitHub code and released in September 2024, so it may
have seen (and prefer) the *fixed* docstring of a pre-release pair. Pairs committed on or after
2024-10-01 cannot be in its training data. Compare pair accuracy before vs after that date.
Cluster bootstrap over commits (B=2000).
"""
import os

import numpy as np
import pandas as pd

RUNS = os.environ.get("DOCDRIFT_RUNS", os.path.join(os.path.dirname(__file__), "..", "runs"))
CUTOFF = pd.Timestamp("2024-10-01", tz="UTC")
DATASET = os.path.join(os.path.dirname(__file__), "..", "dataset", "docfix_pairs.jsonl.gz")
real = pd.read_json(DATASET, lines=True, dtype={"commit": str}).sort_values("pair_id").reset_index(drop=True)
n = len(real)
post = (pd.to_datetime(real.date, utc=True) >= CUTOFF).values
wo, wn = real.doc_old.str.split().str.len(), real.doc_new.str.split().str.len()
lm = (real.interface & ((wn - wo).abs() <= 0.10 * wo)).values
L = np.r_[wo.values, wn.values].astype(float)
length_c = (L[:n] < L[n:]) + 0.5 * (L[:n] == L[n:])
rng = np.random.default_rng(0)


def ci(c, idx):
    cm = real.commit.values[idx]
    cs = np.unique(cm)
    by = {x: idx[cm == x] for x in cs}
    b = [c[np.concatenate([by[x] for x in rng.choice(cs, len(cs))])].mean() for _ in range(2000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


rows = []
for model in ["qwen-1.5b", "qwen-7b", "length-baseline"]:
    if model == "length-baseline":
        c = length_c
    else:
        s = pd.read_pickle(os.path.join(RUNS, "llm_judge", f"exp11_scores_{model}.pkl"))["real"]
        c = (s[:n] > s[n:]) + 0.5 * (s[:n] == s[n:])
    for sub, m in [("interface", real.interface.values), ("interface-length-matched", lm)]:
        for period, pm in [("before 2024-10", ~post), ("after 2024-10 (unseen)", post)]:
            idx = np.where(m & pm)[0]
            lo, hi = ci(c, idx)
            rows.append({"model": model, "subset": sub, "period": period, "pairs": len(idx),
                         "pair_acc": c[idx].mean(), "ci_lo": lo, "ci_hi": hi})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RUNS, "llm_judge", "exp11c_contamination_check.csv"), index=False)
print(out.round(3).to_string(index=False))
