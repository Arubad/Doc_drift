# DocDrift — working notes for Claude Code

Research repo: does a docstring still describe its code? Studies documentation-drift detectors
and shows synthetic (mutation-based) benchmarks don't transfer to real drift. Start with
`README.md` (layout, key findings, reproduce commands), then `docs/paper.md` (draft) and
`docs/PROTOCOL_real_drift.md` (pre-registered protocol, addenda A/B, deviations).

Work happens on branch `real-drift-study` (not yet merged to `main`, no PR opened).

## Status (as of 2026-11-24)

- All experiments exp1–exp11c are **done**; outputs in `runs/`. Paper draft is complete
  (all sections, 2 figures). Dataset released in `dataset/`. Unit tests pass:
  `python -m unittest discover -s tests`.
- Work is **paused** waiting on the human annotation pass below.

## Next steps, in priority order

1. **Human annotation (annotator B) — waiting on the user.**
   - File: `runs/real_drift/ANNOTATE_ME.md`, 50 pairs, each with a blank `LABEL:` line.
     Labels: `factual` / `incomplete` / `clarity` / `noise` (definitions at the top of the file).
   - Annotator A (an LLM assistant) labelled 150 pairs: `dataset/annotations_annotator_A.csv`
     (same as `runs/real_drift/exp8b_annotation_A.csv`). Paper reports precision
     (factual+incomplete) 0.55 overall / 0.65 interface subset, ceiling ≈ 0.83.
   - **Do not show or hint at annotator A's labels, or suggest labels for specific pairs,
     before the user has finished annotating** — that would invalidate the agreement measure.
   - Check progress with: `grep -c '^LABEL: *$' runs/real_drift/ANNOTATE_ME.md` (blank = remaining).
2. **After annotation:** write a script (not yet written; e.g. `experiments/exp8c_agreement.py`)
   that parses the labels from `ANNOTATE_ME.md` (pair ids are in the `## N. pair <id>` headings),
   joins with annotator A on `pair_id`, and reports Cohen's κ (4-way and binary drift vs not-drift),
   plus precision recomputed from annotator B. Then update the paper (§2.2 label quality,
   §5 Limitations, §6) and close Deviation #3 in the protocol.
3. Verify and expand references in `docs/paper.md` (Liu et al. 2018 flagged; only 4 refs total).
4. Optional extensions listed in paper §6: larger LLM judges, prompt-robustness checks (prompts
   fixed in advance), a public length-matched benchmark split.
5. Open a PR `real-drift-study` → `main` when the user asks.

## Conventions

- **Pre-registration discipline:** any new hypothesis or analysis is written into
  `docs/PROTOCOL_real_drift.md` (addendum or Deviations list) and committed **before** running
  it. Post-hoc analyses are labelled exploratory. Keep this up.
- Commit messages are short and prefixed by area, e.g. `exp11: ...`, `protocol: ...`.
- Heavy runs (exp8 mining clones 33 repos; exp11 needs a GPU) were done on free Google Colab;
  don't rerun them locally unless asked.
- The paper discloses AI assistance; annotator A is labelled as an LLM in the data.
