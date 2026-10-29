"""Experiment 2 -- Build the mutation benchmark.

For every corpus function we attempt every mutation (with a per-function
seed). Applicability rates per mutation are a result in themselves: they tell
us which kinds of drift the corpus can express. The benchmark is balanced per
function: one clean copy (label 0), up to `K_POS` mutated copies (label 1)
and `K_NEG` benign-edit copies (label 0, hard negatives).

Every source -- clean, mutated, benign -- is normalised through
ast.unparse so that formatting cannot leak the label."""
import ast
import os
import hashlib

import numpy as np
import pandas as pd

from common import DATA, save_csv
from docdrift.mutations import apply_mutation, ALL_MUTATIONS, BENIGN, kind_of

K_POS, K_NEG = 3, 1
df = pd.read_pickle(os.path.join(DATA, "corpus.pkl"))
rng = np.random.default_rng(0)

rows, applic = [], {m: [0, 0] for m in ALL_MUTATIONS + sorted(BENIGN)}
for i, r in df.iterrows():
    seed = int(hashlib.md5(r.uid.encode()).hexdigest()[:8], 16)
    try:
        clean = ast.unparse(ast.parse(r.source))
    except Exception:
        continue
    rows.append({"uid": r.uid, "project": r.project, "name": r.name, "source": clean,
                 "docstring": r.docstring, "label": 0, "mutation": "none", "kind": "clean"})
    muts = list(ALL_MUTATIONS); rng.shuffle(muts)
    n_pos = 0
    for m in muts:
        out = apply_mutation(r.source, m, seed)
        applic[m][1] += 1
        if out is None:
            continue
        applic[m][0] += 1
        if n_pos < K_POS:
            rows.append({"uid": r.uid, "project": r.project, "name": r.name, "source": out,
                         "docstring": r.docstring, "label": 1, "mutation": m, "kind": kind_of(m)})
            n_pos += 1
    bens = sorted(BENIGN); rng.shuffle(bens)
    n_neg = 0
    for m in bens:
        out = apply_mutation(r.source, m, seed)
        applic[m][1] += 1
        if out is None:
            continue
        applic[m][0] += 1
        if n_neg < K_NEG:
            rows.append({"uid": r.uid, "project": r.project, "name": r.name, "source": out,
                         "docstring": r.docstring, "label": 0, "mutation": m, "kind": "benign"})
            n_neg += 1

bench = pd.DataFrame(rows)
bench.to_pickle(os.path.join(DATA, "benchmark.pkl"))
print(bench.groupby(["kind", "label"]).size())
print(bench.mutation.value_counts())
ap = pd.DataFrame([{"mutation": m, "kind": kind_of(m), "applicable": a, "attempted": n,
                    "rate": a / n if n else 0} for m, (a, n) in applic.items()])
save_csv(ap.sort_values("rate", ascending=False), "exp2_applicability.csv")
save_csv(bench.groupby(["kind", "mutation", "label"]).size().reset_index(name="n"), "exp2_benchmark_composition.csv")
print(ap.round(3))
