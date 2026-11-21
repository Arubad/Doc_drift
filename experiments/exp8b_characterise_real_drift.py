"""Experiment 8b -- Clean and characterise the mined doc-fix pairs.

1. Re-apply the (stricter) trivial-edit filter: markup, version directives,
   apostrophes and single-token typos are not drift.
2. Tag each pair with the interface claims its docstring edit touches
   (param / return / raise / type / default, else prose). Tags are metadata
   for analysis and subset definition; they never decide a label.
3. Draw a stratified random sample for manual annotation of label precision
   (results/exp8b_annotation_sample.csv; fill the `annotation` column with
   one of: factual / incomplete / clarity / noise).

Writes data/real_pairs_clean.pkl.
"""
import difflib
import os

import numpy as np
import pandas as pd

from common import DATA, save_csv
from docdrift.realdrift import is_trivial_edit, edit_categories, INTERFACE_CATEGORIES

raw = pd.read_pickle(os.path.join(DATA, "real_pairs.pkl"))
keep = ~np.array([is_trivial_edit(o, n) for o, n in zip(raw.doc_old, raw.doc_new)])
df = raw[keep].copy()
# a function edited by several commits contributes several pairs; keep them (different docstrings),
# but drop exact duplicate (code, old doc, new doc) triples (cherry-picks / backports)
df = df.drop_duplicates(subset=["source_new", "doc_old", "doc_new"]).reset_index(drop=True)
df["categories"] = [edit_categories(o, n, s) for o, n, s in zip(df.doc_old, df.doc_new, df.source_new)]
df["interface"] = df.categories.map(lambda c: any(x in INTERFACE_CATEGORIES for x in c))
df["pair_id"] = range(len(df))
df.to_pickle(os.path.join(DATA, "real_pairs_clean.pkl"))

print(f"raw pairs {len(raw)} -> clean {len(df)} ({len(raw) - keep.sum()} trivial, "
      f"{keep.sum() - len(df)} duplicates); interface subset {int(df.interface.sum())}")
per_repo = df.groupby("repo").agg(pairs=("pair_id", "size"), commits=("commit", "nunique"),
                                  interface=("interface", "sum")).sort_values("pairs", ascending=False)
save_csv(per_repo.reset_index(), "exp8b_per_repo.csv")
cats = df.explode("categories").categories.value_counts().rename_axis("category").reset_index(name="pairs")
cats["share"] = cats.pairs / len(df)
save_csv(cats, "exp8b_categories.csv")
print(cats.round(3).to_string(index=False))

# stratified annotation sample: 150 pairs, at most 8 per repo, fixed seed
rng = np.random.default_rng(0)
sample = (df.sample(frac=1, random_state=0).groupby("repo", group_keys=False).head(8)
            .sample(n=min(150, len(df)), random_state=0))


def doc_diff(o, n):
    return "\n".join(l for l in difflib.unified_diff(o.splitlines(), n.splitlines(), lineterm="", n=1)
                     if not l.startswith(("---", "+++")))


ann = pd.DataFrame({"pair_id": sample.pair_id, "repo": sample.repo, "commit": sample.commit.str[:10],
                    "message": sample.message, "qualname": sample.qualname,
                    "categories": sample.categories.map(",".join),
                    "signature": sample.source_new.map(lambda s: s.splitlines()[0]),
                    "doc_diff": [doc_diff(o, n) for o, n in zip(sample.doc_old, sample.doc_new)],
                    "annotation": ""})
save_csv(ann, "exp8b_annotation_sample.csv")
