"""Experiment 11b (exploratory, post hoc) -- Is the LLM judges' length-matched accuracy a length effect?

Within the length-matched interface subset, split pairs by whether the new docstring is
longer, shorter, or exactly the same length (in words) as the old one. On same-length pairs
the length baseline is 0.5 by construction, so accuracy above 0.5 there cannot come from
length. Cluster bootstrap over commits (B=2000).

Inputs: real_pairs_clean.pkl and exp11 score files (DOCDRIFT_RUNS, default ../runs).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUNS = os.environ.get("DOCDRIFT_RUNS", os.path.join(os.path.dirname(__file__), "..", "runs"))
real = pd.read_pickle(os.path.join(RUNS, "real_drift", "real_pairs_clean.pkl")).reset_index(drop=True)
n = len(real)
wo, wn = real.doc_old.str.split().str.len(), real.doc_new.str.split().str.len()
lm = (real.interface & ((wn - wo).abs() <= 0.10 * wo)).values
d = (wn - wo).values
groups = {"new longer": lm & (d > 0), "new shorter": lm & (d < 0), "same length": lm & (d == 0), "all length-matched": lm}
length_base = {"new longer": 1.0, "new shorter": 0.0, "same length": 0.5}
rng = np.random.default_rng(0)
rows = []
for model in ["qwen-1.5b", "qwen-7b"]:
    s = pd.read_pickle(os.path.join(RUNS, "llm_judge", f"exp11_scores_{model}.pkl"))["real"]
    c = (s[:n] > s[n:]) + 0.5 * (s[:n] == s[n:])
    for g, k in groups.items():
        idx = np.where(k)[0]
        cm = real.commit.values[idx]
        cs = np.unique(cm)
        by = {x: idx[cm == x] for x in cs}
        boot = [c[np.concatenate([by[x] for x in rng.choice(cs, len(cs))])].mean() for _ in range(2000)]
        rows.append({"model": model, "group": g, "pairs": len(idx), "pair_acc": c[idx].mean(),
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                     "length_baseline": length_base.get(g, np.nan)})
    rows.append({"model": model, "group": "spearman(score diff, length diff)", "pairs": int(lm.sum()),
                 "pair_acc": spearmanr(s[:n][lm] - s[n:][lm], d[lm]).statistic})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RUNS, "llm_judge", "exp11b_length_check.csv"), index=False)
print(out.round(3).to_string(index=False))
