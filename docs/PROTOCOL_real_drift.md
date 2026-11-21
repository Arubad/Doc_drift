# Study protocol: does synthetic-benchmark performance predict real documentation-drift detection?

*Written 2026-11-21, after mining began and **before** any detector was evaluated on real pairs.
Changes after that point are listed under "Deviations" at the bottom.*

## Question

Documentation-drift (comment–code inconsistency) detectors are trained and evaluated on
synthetic drift produced by mutation operators. The DocDrift study (REPORT.md, replicated on
an independent pinned corpus in `runs/colab_pinned_corpus/`) shows these detectors lose
0.15–0.18 AUROC when the mutation type is held out (LOMO). Whether that matters depends on
an untested assumption: **which synthetic number tells you how a detector will do on real drift?**

## Hypotheses

* **H1 (primary).** Across detector variants, LOMO AUROC on the synthetic benchmark has a
  higher Spearman correlation with accuracy on real drift than in-distribution AUROC has
  (ρ_LOMO − ρ_in-dist > 0, 95 % cluster-bootstrap CI excluding 0).
* **H2.** Detectors' accuracy on real drift is well below their in-distribution synthetic
  AUROC, i.e. the synthetic benchmark overstates real-world performance.
* **H3.** Real drift that touches interface claims (parameters, return/yield, exceptions,
  types, defaults) is detected far better than prose-only drift.

A result against H1 (in-distribution predicts better, or neither predicts) is reported the same way.

## Real drift data (exp8, exp8b)

* **Source.** Git histories (since 2012) of the GitHub repositories of 33 of the 35 pinned corpus
  packages (`experiments/exp8_mine_real_drift.py`; HEAD commits recorded in `results/exp8_repos.csv`).
* **Doc-fix pair.** A function whose docstring changed in a commit while its code — the AST with
  the docstring removed — is identical. Old (code, docstring) = label 1, new = label 0. Because
  the code is identical, the pair can only be separated by judging docstring against code.
  The labels come from developers, not from our operators.
* **Exclusions** (decided before evaluation, all independent of detector features): merges;
  commits touching > 3 non-test `.py` files or > 60 changed lines; commits editing > 3
  docstrings (style sweeps); test/doc/example/benchmark paths; trivial edits (whitespace,
  punctuation, apostrophes, reST roles and link targets, `versionadded/versionchanged/deprecated`
  directives, single-token typo fixes with edit distance ≤ 2); docstrings < 8 words; functions
  outside 3–80 lines; exact duplicate (code, old, new) triples.
* **Edit categories** (metadata, never used to assign labels): which interface claims the edit
  touches — param / return / raise / type / default — else prose.
* **Subsets.** `interface` (any interface category; **primary**) and `all` (secondary).
* **Label precision.** A stratified random sample of 150 pairs (≤ 8 per repository) is annotated
  as *factual* (old docstring was wrong about the code), *incomplete* (old docstring omitted
  behaviour the code has), *clarity* (rewording; old docstring not wrong), or *noise*.
  Precision = (factual + incomplete) / annotated, reported per subset.

## Detectors and synthetic metrics (exp9)

* 4 families (struct-LR, struct-GBM, tfidf-LR, hybrid-GBM; hyper-parameters as in
  `docdrift/models.py`) × 7 training regimes (operator subsets: all, surface, behavioural,
  semantic, surface+behavioural, interface-only, all with clean-only negatives) = **28 variants**;
  the zero-training heuristic is reported but excluded from correlations.
* **In-distribution AUROC:** 70/30 split grouped by function; test positives from the regime's operators.
* **LOMO AUROC:** mean over the regime's operators of AUROC when that operator is held out of training.
* **Real-drift metrics:** pair accuracy P(score(old) > score(new)) (ties ½; primary) and AUROC
  over all real rows.
* **Leakage control:** real-pair scores come from models trained without any synthetic
  function of the pair's package (5 package folds).

## Analysis

* Spearman ρ between each synthetic metric and real pair accuracy, across the 28 variants.
* 95 % CIs by cluster bootstrap over commits (B = 1000), variants fixed; report
  P(ρ_LOMO > ρ_in-dist) under the bootstrap.
* Per-category pair accuracy for H3.

## Planned extensions (GPU stage)

* LLM judges (open-weights code models) on the same real pairs — not trained on mutations,
  so they show what a mutation-free detector achieves on real drift.
* LLM-assisted annotation of all pairs, validated against the 150 manual labels.

## Deviations

*(none yet)*
