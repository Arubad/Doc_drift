"""Experiment 1 -- Mine the corpus and characterise it.

Mines every documented function from the local CPython standard library
(offline, reproducible on any machine) and reports corpus statistics: number of
functions, projects (top-level packages), docstring length, parameter counts,
and how often docstrings make *checkable claims* (mention parameters,
return/yield, exceptions, numeric defaults). The claim-rate determines the
ceiling of any lexical/structural detector."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA, save_csv, fig_path
from docdrift.corpus import mine
from docdrift.features import extract, FEATURE_NAMES

root = os.environ.get("DOCDRIFT_CORPUS")  # optional: point at any repo
df = mine(root)
print(f"mined {len(df)} documented functions from {df.project.nunique()} projects")
df.to_pickle(os.path.join(DATA, "corpus.pkl"))

F = np.stack([extract(s, d) for s, d in zip(df.source, df.docstring)])
feat = pd.DataFrame(F, columns=FEATURE_NAMES)
stats = {
    "n_functions": len(df), "n_projects": int(df.project.nunique()),
    "median_doc_words": float(feat.doc_len.median()), "median_code_lines": float(feat.code_len.median()),
    "frac_mentions_all_params": float((feat.frac_params_mentioned == 1).mean()),
    "frac_mentions_return": float(feat.doc_mentions_return.mean()),
    "frac_mentions_raise": float(feat.doc_mentions_raise.mean()),
    "frac_has_numeric_default": float((feat.n_defaults > 0).mean()),
    "frac_defaults_mentioned": float((feat.n_doc_numbers_matched_defaults > 0).mean()),
}
print(pd.Series(stats))
save_csv(pd.DataFrame([stats]), "exp1_corpus_stats.csv")
top = df.project.value_counts().head(25).reset_index()
top.columns = ["project", "n_functions"]
save_csv(top, "exp1_top_projects.csv")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
axes[0].hist(feat.doc_len, bins=50, range=(0, 200)); axes[0].set_title("docstring words")
axes[1].hist(feat.n_params, bins=range(0, 12)); axes[1].set_title("# parameters")
axes[2].hist(feat.frac_params_mentioned, bins=20); axes[2].set_title("fraction of params mentioned in doc")
fig.tight_layout(); fig.savefig(fig_path("exp1_corpus.png"), dpi=130)
print("done")
