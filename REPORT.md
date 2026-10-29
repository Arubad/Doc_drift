# DocDrift: A Mutation Benchmark and Leave-One-Mutation-Out Protocol for Documentation-Drift Detection

*Research report — all numbers are read from `results/` produced by `run_all.sh`.*

---

## 1. Problem and motivation

Documentation drift — a docstring that no longer describes the function under it —
is one of the most common and least-studied forms of technical debt. Stale
documentation misleads API users, confuses code-review, and (increasingly) poisons
the training data and retrieval context of code LLMs, which read docstrings as ground
truth about behaviour.

Existing work on *comment–code inconsistency* (Panthaplackel et al. 2021 "Deep JIT
inconsistency detection", Liu et al. 2018 "Just-in-time obsolete comment detection",
Wen et al. 2019) treats the problem as a **just-in-time** task: given a code *change*
(diff), predict whether the comment should also change. That framing needs the diff
and the commit history. It cannot answer the question a maintainer, a linter, or an
LLM-based agent actually faces: *given this (code, docstring) pair as it exists right
now, is the docstring still true?* We call this **static documentation-drift
detection** and study it end to end.

Research questions:

* **RQ1** How well can a detector that sees only the (code, docstring) pair separate
  drifted from consistent pairs, and which kinds of drift are detectable without
  executing the code?
* **RQ2** Does a cheap, interpretable *structural-alignment* model (32 features that
  check the docstring's interface claims against the AST) match a lexical
  bag-of-identifiers model?
* **RQ3** Do detectors trained on synthesised drift learn *alignment*, or the
  fingerprints of the mutation operators that synthesised it? (This is the question
  the JIT literature, which also trains on mutated data, has not asked.)
* **RQ4** Do detectors transfer across packages with different documentation
  conventions, how robust are they to label noise and small training sets, and do they
  find real drift in the wild?

## 2. Contribution (honestly scoped)

1. **DocDrift corpus and benchmark.** 21 732 documented functions mined offline from
   the Python standard library and 270 installed third-party packages, and a
   12-operator AST mutation taxonomy (4 surface, 4 behavioural, 2 semantic, 2 benign
   hard negatives) producing a 103 640-pair benchmark with controlled drift *kind*.
   The pipeline runs on any Python repository (`DOCDRIFT_CORPUS=path`).
2. **A 32-feature structural-alignment detector** that is interpretable, needs no
   vocabulary, and degrades gracefully under label noise (−0.03 AUROC at 40 % noise
   vs −0.19 for TF-IDF).
3. **The Leave-One-Mutation-Out (LOMO) protocol, and a negative result that
   reframes the task.** In-distribution, detectors reach 0.81–0.86 AUROC. Held out one
   mutation type at a time, *every* detector — lexical and structural — drops by
   0.15–0.18 AUROC on average, and on several drift kinds falls to or *below* chance
   (e.g. `wrap_return`: 0.99 → 0.41). Cross-kind transfer (train on surface drift, test
   on behavioural) is 0.49–0.65. Detectors trained on mutated data learn
   per-operator cues, not alignment. We argue LOMO AUROC, not in-distribution AUROC,
   is the number that predicts real-world usefulness and should be reported by any
   inconsistency-detection paper that trains on synthetic drift.
4. **Negative-side artifact learning.** Against benign-edit negatives the TF-IDF
   model scores 0.92 AUROC; against untouched originals, 0.80. The 0.12 gap is the model
   learning the benign operators' fingerprints as a "consistent" cue. The structural
   model shows no gap (0.807 vs 0.808).
5. **Wild-scan evidence.** Trained on the benchmark and run over the unmutated corpus,
   the structural detector surfaces real drift in shipped libraries: e.g.
   `jsonschema._utils.find_additional_properties` and `pymdownx.highlight._wrap_code`
   document "Return …" but are generators (`yield`), `pathspec.match_tree_files`
   documents two parameters that do not exist. These are candidates for upstream
   fixes — accepted fixes are strong evidence of practical value.

## 3. Related work

* **JIT comment inconsistency** (Liu et al. ASE 2018; Panthaplackel et al. AAAI 2021;
  Steiner & Zhang 2022): diff-conditioned, neural, trained on mined commits with
  heuristic labels. We remove the diff requirement and show the artifact-learning
  risk of mutation-trained detectors.
* **Code–comment coherence / comment quality** (Corazza et al. 2018; Rabbi & Siddik
  2020; Steidl et al. 2013): similarity-based, no notion of drift *kind*.
* **Mutation testing for ML-on-code** (e.g. Jain et al. 2020 for code models): our
  taxonomy borrows the operator style but targets documentation contracts (parameters,
  return/yield, exceptions, defaults) rather than test adequacy.
* **LLM-based docstring checking** (2023–): prompt-based, expensive, unevaluated on
  controlled drift kinds; our benchmark is a drop-in evaluation set for them (see §7).

## 4. Method

### 4.1 Corpus (`docdrift/corpus.py`, Exp 1)

AST-mined functions with a docstring of ≥ 8 words, 3–80 lines, ≤ 40 per file.
Statistics: median 32 docstring words, 23 code lines; **68 %** of docstrings mention
every parameter, 51 % mention a return value, 10 % mention raising, only 3 % mention a
numeric default. These claim rates bound what any static detector can verify: drift in
a *never-mentioned* aspect is undetectable by construction.

### 4.2 Mutation taxonomy (`docdrift/mutations.py`, Exp 2)

Every operator edits the **code** and leaves the docstring untouched; all sources
(clean, mutated, benign) are re-rendered through `ast.unparse` so formatting cannot leak
the label.

| kind | operator | applicability | what drifts |
|---|---|---|---|
| surface | `rename_param` | 84 % | docstring names a parameter that no longer exists |
| surface | `remove_param` | 28 % | documented defaulted parameter deleted |
| surface | `add_param` | 100 % | undocumented keyword parameter added and used |
| surface | `change_default` | 30 % | literal default changed |
| behavioural | `add_raise` | 84 % | undocumented exception guard inserted |
| behavioural | `remove_raise` | 23 % | documented raise removed |
| behavioural | `return_to_yield` | 78 % | function becomes a generator |
| behavioural | `wrap_return` | 78 % | return value wrapped in `list`/`str`/`tuple` |
| semantic | `negate_condition` | 70 % | an `if` test negated |
| semantic | `swap_binop` | 47 % | arithmetic / comparison operator swapped |
| benign | `rename_local` | 80 % | local (non-parameter) variable renamed (hard negative) |
| benign | `rewrite_augassign` | 7 % | `x += y` ↔ `x = x + y` (hard negative) |

Benchmark: 21 732 clean (label 0) + 17 458 benign (label 0) + 64 450 mutated (label 1):
25 252 surface, 27 454 behavioural, 11 744 semantic.

### 4.3 Detectors (`docdrift/models.py`)

* `heuristic` — zero-training: fraction of unmentioned parameters + return/yield flag.
* `struct-LR`, `struct-GBM` — 32 structural-alignment features (`docdrift/features.py`):
  parameters mentioned/unmentioned/unmatched, return vs yield claims, documented vs
  raised exceptions, numeric defaults vs numbers in the docstring, identifier Jaccard,
  wrapper-call flags, and size counts; logistic regression / gradient boosting.
* `tfidf-LR` — separate TF-IDF over code identifiers (docstring stripped) and
  docstring tokens, 40 k dims, logistic regression.
* `hybrid-GBM` — SVD(64) of the TF-IDF block + structural features, boosting.

Splits are **grouped by function** (all variants of a function fall in the same fold);
Exp 4 additionally groups by **package**.

## 5. Experiments and results

### Exp 3 — Detector comparison, 5-fold grouped CV (`exp3_detectors.py`)

| detector | AUROC | AUPRC | F1 | recall @ 5 % FPR | FPR on clean originals |
|---|---|---|---|---|---|
| heuristic | 0.644 | 0.716 | 0.303 | 0.187 | 0.078 |
| struct-LR | 0.756 | 0.837 | 0.779 | 0.322 | 0.472 |
| struct-GBM | 0.808 | 0.876 | 0.798 | 0.398 | 0.402 |
| tfidf-LR | **0.856** | 0.905 | **0.825** | 0.481 | 0.444 |
| hybrid-GBM | 0.850 | **0.913** | 0.808 | **0.566** | 0.346 |

Per drift kind (AUROC, positives of that kind vs all negatives):

| kind / operator | heuristic | struct-GBM | tfidf-LR | hybrid-GBM |
|---|---|---|---|---|
| **surface** | 0.715 | 0.801 | 0.865 | 0.861 |
| `rename_param` | 0.809 | 0.828 | 0.831 | 0.841 |
| `add_param` | 0.719 | 0.887 | 0.972 | 0.984 |
| `remove_param` | 0.556 | 0.525 | 0.676 | 0.594 |
| `change_default` | 0.521 | 0.583 | 0.676 | 0.617 |
| **behavioural** | 0.639 | 0.912 | 0.903 | 0.945 |
| `return_to_yield` | 0.739 | 0.987 | 0.951 | 0.990 |
| `wrap_return` | 0.468 | 0.991 | 0.807 | 0.993 |
| `add_raise` | 0.741 | 0.837 | 0.955 | 0.935 |
| `remove_raise` | 0.508 | 0.610 | **0.881** | 0.617 |
| **semantic** | 0.501 | 0.579 | 0.726 | 0.605 |
| `negate_condition` | 0.502 | 0.603 | 0.783 | 0.637 |
| `swap_binop` | 0.500 | 0.541 | 0.634 | 0.552 |

Findings (RQ1, RQ2). (i) Interface-level drift (parameters, return/yield) is highly
detectable statically; `remove_param`, `change_default` and `remove_raise` are hard
because only 3–10 % of docstrings make the corresponding claim (§4.1 ceiling).
(ii) Semantic drift is near chance for the structural model and 0.63–0.78 for TF-IDF —
but see Exp 7 for why that TF-IDF number is suspicious. (iii) The 32-feature structural
GBM is within 0.05 AUROC of the 40 k-dimensional lexical model and ahead on behavioural
drift; the hybrid gives the best recall at a fixed 5 % false-positive budget (0.566),
which is the operating point a linter would use.

### Exp 3b — Which negatives are being separated? (`exp3b_negative_type_analysis.py`)

| detector | AUROC vs benign-edit negatives | AUROC vs clean originals | gap |
|---|---|---|---|
| struct-LR | 0.757 | 0.756 | +0.001 |
| struct-GBM | 0.807 | 0.808 | −0.001 |
| hybrid-GBM | 0.859 | 0.843 | +0.017 |
| **tfidf-LR** | **0.922** | **0.803** | **+0.118** |

The lexical model's headline AUROC is inflated by the benign edits: it has learned
that a renamed local drawn from the benign pool (`result`, `tmp`, `cur`, …) signals
"consistent". Against the only negatives that exist in deployment — untouched code —
TF-IDF (0.803) and structural GBM (0.808) are equal. **Hard negatives generated by
operators carry their own fingerprints and must be reported separately from clean
negatives.**

### Exp 7 — Leave-One-Mutation-Out (`exp7_leave_one_mutation_out.py`)  ← key result

Train on all mutation types but one; test on the held-out type vs all negatives.

| held-out operator | struct-GBM in-dist → LOMO | struct-LR in-dist → LOMO | tfidf-LR in-dist → LOMO |
|---|---|---|---|
| `add_param` | 0.888 → 0.772 | 0.742 → 0.676 | 0.972 → 0.763 |
| `add_raise` | 0.833 → 0.561 | 0.786 → 0.526 | 0.955 → 0.714 |
| `rename_param` | 0.826 → 0.643 | 0.775 → 0.716 | 0.823 → 0.586 |
| `return_to_yield` | 0.986 → 0.531 | 0.982 → 0.599 | 0.953 → 0.648 |
| `wrap_return` | 0.991 → **0.412** | 0.989 → **0.363** | 0.809 → 0.567 |
| `remove_raise` | 0.615 → 0.583 | 0.550 → 0.532 | 0.875 → 0.662 |
| `negate_condition` | 0.598 → 0.554 | 0.518 → 0.505 | 0.783 → 0.655 |
| `swap_binop` | 0.535 → 0.516 | 0.508 → 0.499 | 0.632 → 0.604 |
| **mean change** | **−0.179** | **−0.148** | **−0.167** |

Cross-kind transfer (AUROC): surface → behavioural 0.49 / 0.48 / 0.64;
behavioural → surface 0.55 / 0.53 / 0.65; surface+behavioural → semantic
0.52 / 0.50 / 0.63 (struct-GBM / struct-LR / tfidf-LR).

Findings (RQ3). (i) **No detector generalises to an unseen drift type.** The structural
model, despite features designed to encode alignment (e.g. `return_yield_mismatch`,
`ret_wrapped_*`), learns *which direction* each feature points only from the operator
that produced it: with `wrap_return` held out, the `ret_wrapped_list` feature has never
been positive in training and the model scores wrapped returns as *more* consistent
than chance (0.41). (ii) The lexical model's in-distribution advantage on semantic
drift (0.78 on `negate_condition`) shrinks to 0.66 under LOMO — part of what it
learned was the inserted `not` token co-occurring with the label. (iii) The gap between
in-distribution and LOMO AUROC is the quantity that measures artifact learning; it is
large for every model here and, we conjecture, for the mutation-trained JIT detectors
in the literature. This turns the project from "build a better detector" into "build a
detector whose LOMO curve is flat" — a crisper and, to our knowledge, unaddressed
problem.

### Exp 4 — Cross-package generalisation (`exp4_cross_project.py`)

5-fold CV grouped by top-level package (train on some packages, test on unseen ones)
vs grouped by function.

| detector | within-package AUROC | cross-package AUROC | Δ | recall@5 %FPR within → cross |
|---|---|---|---|---|
| struct-LR | 0.757 ± 0.002 | 0.755 ± 0.023 | −0.002 | 0.321 → 0.321 |
| struct-GBM | 0.808 ± 0.002 | 0.805 ± 0.024 | −0.003 | 0.398 → 0.394 |
| tfidf-LR | 0.856 ± 0.002 | 0.843 ± 0.016 | −0.013 | 0.480 → 0.448 |

Findings (RQ4). Package shift is mild for every detector; the structural model is
essentially invariant to documentation convention (Sphinx `:param:`, Google, NumPy,
bare prose) while the lexical model loses a little vocabulary. Package shift is
therefore *not* the hard generalisation axis — drift-kind shift (Exp 7) is.

### Exp 5 — Interpretability, label noise, learning curve (`exp5_feature_analysis.py`)

Top logistic coefficients (positive = drift): `ret_wrapped_str` +0.61,
`ret_wrapped_list` +0.59, `code_has_yield` +0.54, `ret_wrapped_tuple` +0.51,
`return_yield_mismatch` +0.39; `frac_params_mentioned` −0.63 (more parameters
mentioned ⇒ more consistent), `code_has_return_value` −0.28, `doc_mentions_yield` −0.26.
These are exactly the cues a human reviewer uses, and also exactly the per-operator
cues that Exp 7 shows do not transfer.

| training label noise | struct-GBM AUROC | tfidf-LR AUROC |
|---|---|---|
| 0 % | 0.806 | 0.856 |
| 10 % | 0.804 | 0.836 |
| 20 % | 0.801 | 0.809 |
| 30 % | 0.794 | 0.755 |
| 40 % | **0.774** | **0.661** |

| training functions | 50 | 100 | 200 | 500 | 1 000 | 2 000 | 15 212 |
|---|---|---|---|---|---|---|---|
| struct-GBM | 0.679 | 0.711 | 0.773 | 0.789 | 0.794 | 0.800 | 0.806 |
| tfidf-LR | 0.754 | 0.795 | 0.813 | 0.825 | 0.834 | 0.841 | 0.856 |

Findings. The structural model is far more noise-robust (loses 0.03 vs 0.19 AUROC at
40 % noise) — important because real-world drift labels mined from commits are noisy —
and saturates by ~500 training functions, so a per-project model is cheap to train.

### Exp 6 — Wild scan of the unmutated corpus (`exp6_wild_scan.py`)

The structural GBM scores 39.6 % of untouched functions above 0.5 (its FPR on clean
code, consistent with Exp 3) — too many to review blindly, but the top of the ranking
is precise. Top-20 candidates (`results/exp6_wild_candidates.csv`,
`exp6_top10_sources.txt`) include, verified by reading the source:

* `jsonschema/_utils.py::find_additional_properties` — docstring says *"Return the set
  of additional properties"*; the function is a generator (`yield property`).
* `pymdownx/highlight.py::_wrap_code` — *"Return source"*; generator.
* `pathspec/pathspec.py::match_tree_files` — documents two parameters and an exception
  that are not in the signature/body.
* `unittest/case.py::subTest` (CPython) — documents a return while being a context
  manager generator; undocumented exception path.
* `matplotlib/__init__.py::rc_context`, `networkx …::strategy_connected_sequential`,
  `sklearn/utils/estimator_checks.py::_generate_sparse_data` — return/yield mismatch.

Several of these are genuine documentation defects (generator described as returning
a collection is a user-visible contract difference), others are stylistic ("Return"
used loosely for "yield"). Deciding which is which is itself a labelling task for the
real-data phase (§7).

## 6. Limitations

* Drift is synthesised. The mutation operators are plausible but not drawn from the
  empirical distribution of real drift; §7 step 1 addresses this.
* Only interface-level claims are checked; semantic drift (what the function computes)
  is out of reach for any static, execution-free detector and we show it is near chance.
* The corpus is Python only and skewed toward large scientific packages.
* No LLM baseline yet (the environment was offline); the benchmark is designed to
  evaluate one.

## 7. Path to a publishable result

1. **Real drift labels.** Mine commit pairs where the signature or return/raise
   structure changed and the docstring did *not* (automatic, high-precision labels),
   plus commits that later fixed the docstring (confirming drift). Measure the
   in-distribution → real-data gap for each detector; LOMO AUROC should predict it
   better than in-distribution AUROC — that correlation is the paper's central claim.
2. **Flat-LOMO detectors.** Two candidates: (a) *claim-extraction* — parse the docstring
   into explicit claims (`param X exists`, `returns Y`, `raises Z`) and verify each
   against the AST symbolically, so there is nothing to learn per operator; (b) an LLM
   judge (code + docstring → "consistent?") evaluated under LOMO, which it trivially
   satisfies because it is not trained on the mutations — the question becomes its FPR
   on clean code, which Exp 6 suggests is the binding constraint.
3. **Report hard negatives separately** (Exp 3b) and **LOMO** (Exp 7) as standard
   evaluation for any mutation-trained inconsistency detector; re-evaluate one public
   JIT model under both to show the effect is not specific to our detectors.
4. **Upstream the wild-scan hits** (jsonschema, pathspec, pymdownx, CPython `subTest`)
   as documentation PRs; accepted fixes are the practical-impact evidence reviewers ask for.
5. **Tooling**: ship the structural detector as a `flake8`/`ruff`-style plugin that
   flags only the high-precision claim types (parameter existence, return/yield,
   documented-but-unraised exceptions), with the measured FPR per claim type.

Target venues: MSR (benchmark + mining angle), ICSME/SANER (tool + wild-scan evidence),
or an ICSE/FSE NIER/short paper on the LOMO evaluation protocol.

## 8. Reproduction

```
bash run_all.sh                       # stdlib + installed packages, ~40 min CPU
DOCDRIFT_CORPUS=/path/to/repo bash run_all.sh   # any Python project
```
Experiments are independent after `exp1`/`exp2`; seeds are fixed.
