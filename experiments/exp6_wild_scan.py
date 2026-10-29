"""Experiment 6 -- Scan the *unmutated* corpus for real drift candidates.

Train the structural model on the benchmark, score every clean function in
the corpus, and list the top-scored functions together with the structural
cues that fired. These are candidates for genuine documentation drift in
CPython to be verified manually (and, if confirmed, reported upstream --
an accepted bug/doc fix is strong evidence for a paper).
"""
import os
import numpy as np
import pandas as pd

from common import load_benchmark, DATA, save_csv
from docdrift.models import StructuralGBM
from docdrift.features import extract, FEATURE_NAMES

bench = load_benchmark()
corpus = pd.read_pickle(os.path.join(DATA, "corpus.pkl"))
d = StructuralGBM().fit(bench.source.values, bench.docstring.values, bench.label.values)
p = d.predict_proba(corpus.source.values, corpus.docstring.values)
F = np.stack([extract(s, dd) for s, dd in zip(corpus.source, corpus.docstring)])
feat = pd.DataFrame(F, columns=FEATURE_NAMES)
corpus["drift_score"] = p
cue_cols = ["n_sig_params_unmentioned", "n_doc_params_unmatched", "return_yield_mismatch",
            "n_code_exceptions_undocumented", "n_doc_exceptions_unmatched", "n_doc_numbers_unmatched_defaults"]
corpus["cues"] = [", ".join(f"{c}={int(v)}" for c, v in zip(cue_cols, row) if v > 0) for row in feat[cue_cols].values]
top = corpus.sort_values("drift_score", ascending=False).head(40)[["file", "name", "lineno", "drift_score", "cues"]]
save_csv(top, "exp6_wild_candidates.csv")
print(f"fraction of clean corpus scored > 0.5: {(p > 0.5).mean():.3f}")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
print(top.head(20).round(3).to_string())
# save sources of top 10 for manual inspection
with open(os.path.join("results", "exp6_top10_sources.txt"), "w") as fh:
    for _, r in corpus.sort_values("drift_score", ascending=False).head(10).iterrows():
        fh.write(f"### {r.file}:{r.lineno} {r.name}  score={r.drift_score:.3f}  cues: {r.cues}\n{r.source}\n\n")
print("done")
