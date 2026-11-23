"""Paper figures from saved results in runs/ (no recomputation)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS, OUT = os.path.join(ROOT, "runs"), os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Figure 1: synthetic metric vs real pair accuracy (28 code-side variants)
V = pd.read_csv(os.path.join(RUNS, "real_drift", "exp9_variants.csv"))
V = V[V.family != "heuristic"]
mk = {"struct-LR": "o", "struct-GBM": "s", "tfidf-LR": "^", "hybrid-GBM": "D"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, col, lab in [(axes[0], "indist_auroc", "in-distribution AUROC"), (axes[1], "lomo_auroc", "leave-one-mutation-out AUROC")]:
    for fam, g in V.groupby("family"):
        ax.scatter(g[col], g.real_pair_acc_interface, marker=mk[fam], label=fam, s=42, alpha=.85)
    ax.axhline(0.5, color="grey", lw=.8, ls=":")
    ax.set_xlabel(f"synthetic {lab}")
    ax.set_title(f"Spearman ρ = {spearmanr(V[col], V.real_pair_acc_interface).statistic:.2f}", fontsize=11)
axes[0].set_ylabel("real doc-fix pair accuracy\n(interface subset, n = 4,082)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(OUT, "fig1_synthetic_vs_real.png"), dpi=160)

# Figure 2: synthetic vs real (length-matched) per detector type
d10v = pd.read_csv(os.path.join(RUNS, "doc_side", "exp10_variants.csv"))
c9 = pd.read_csv(os.path.join(RUNS, "real_drift", "exp9_variants.csv"))
labels, syn, real = [], [], []
for fam in mk:
    labels.append(f"{fam}\ncode-side"); syn.append(c9[(c9.variant == f"{fam}|all")].indist_auroc.iloc[0])
    real.append(c9[(c9.variant == f"{fam}|all")].real_pair_acc_interface.iloc[0])
labels.append("struct-LR\ndoc-side"); syn.append(d10v[d10v.variant == "struct-LR|doc-all"].indist_auroc.iloc[0])
real.append(d10v[d10v.variant == "struct-LR|doc-all"].pair_acc_interface.iloc[0])
for m in ["qwen-1.5b", "qwen-7b"]:
    r = pd.read_csv(os.path.join(RUNS, "llm_judge", f"exp11_{m}.csv")).set_index("subset")
    labels.append(f"{m}\nzero-shot"); syn.append(r.loc["interface", "synthetic_auroc"])
    real.append(r.loc["interface", "pair_acc"])
labels.append("docstring\nlength only"); syn.append(float("nan")); real.append(0.781)
x = range(len(labels)); w = .38
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar([i - w / 2 for i in x], syn, w, label="synthetic benchmark (AUROC)", color="#8da0cb")
ax.bar([i + w / 2 for i in x], real, w, label="real doc-fix pairs (pair accuracy, interface)", color="#fc8d62")
ax.axhline(0.5, color="grey", lw=.8, ls=":")
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(.3, 1.0)
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_ylabel("score (0.5 = chance)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_synthetic_vs_real_by_detector.png"), dpi=160)
print("saved figures to", OUT)
