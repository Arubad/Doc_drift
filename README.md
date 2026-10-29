# DocDrift — Documentation Drift Detection

**Research question:** can we detect that a function's docstring no longer describes
its code (*documentation drift*) from the (code, docstring) pair alone — no diff, no
execution, no LLM — and how should such detectors be evaluated so that they learn
*alignment* rather than the fingerprints of the benchmark generator?

Fully offline: the corpus is mined from the Python standard library installed on the
machine (21 732 documented functions, 146 packages); drift is synthesised with a
12-operator AST mutation taxonomy.

```
pip install -r requirements.txt
bash run_all.sh     # ~40 min CPU; writes ./data (corpus, benchmark) and ./results
DOCDRIFT_CORPUS=/path/to/any/python/repo bash run_all.sh   # run on your own code base
```

| File | Content |
|---|---|
| `docdrift/corpus.py` | mines documented functions (AST), groups by top-level package |
| `docdrift/mutations.py` | mutation taxonomy: 4 surface, 4 behavioural, 2 semantic, 2 benign (hard negatives) |
| `docdrift/features.py` | 32 structural-alignment features (parameter / return / raise / default claims checked against the AST) |
| `docdrift/models.py` | heuristic, structural LR/GBM, TF-IDF LR, hybrid GBM |
| `experiments/exp1..exp7` | corpus stats, benchmark build, detector comparison, cross-package transfer, feature/noise/learning-curve analysis, wild scan of CPython, **leave-one-mutation-out** |

See **REPORT.md** for the narrative, results and the research plan.
