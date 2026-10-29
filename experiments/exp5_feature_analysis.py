"""Experiment 5 -- What do the detectors use? Feature importance, label-noise
robustness and the learning curve.

(a) Logistic coefficients of the structural model (interpretability).
(b) Label noise: flip 0/10/20/30/40 % of training labels; does AUROC degrade
    gracefully? (Mutation-generated labels are clean; real-world mined labels
    would not be, so this bounds the transfer risk.)
(c) Learning curve: AUROC vs number of training *functions* -- how small can
    a project-specific training set be?
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from common import load_benchmark, save_csv, fig_path
from docdrift.models import StructuralLR, StructuralGBM, TfidfLR
from docdrift.features import FEATURE_NAMES
from docdrift.metrics import summarise

bench = load_benchmark()
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr, te = next(gss.split(bench, bench.label, groups=bench.uid))
S, D, Y = bench.source.values, bench.docstring.values, bench.label.values

# (a)
m = StructuralLR().fit(S[tr], D[tr], Y[tr])
coef = pd.DataFrame({"feature": FEATURE_NAMES, "coef": m.coef()}).sort_values("coef", key=np.abs, ascending=False)
save_csv(coef, "exp5_coefficients.csv")
print(coef.head(12).round(3).to_string())

# (b) label noise
rows = []
rng = np.random.default_rng(0)
for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
    for seed in range(3):
        y_noisy = Y[tr].copy()
        flip = rng.random(len(y_noisy)) < noise
        y_noisy[flip] = 1 - y_noisy[flip]
        for cls in [StructuralGBM, TfidfLR]:
            d = cls().fit(S[tr], D[tr], y_noisy)
            s = summarise(Y[te], d.predict_proba(S[te], D[te]))
            rows.append({"noise": noise, "seed": seed, "detector": cls.name, **s})
noise_df = pd.DataFrame(rows)
save_csv(noise_df, "exp5_label_noise.csv")
print(noise_df.groupby(["detector", "noise"]).auroc.mean().round(3))

# (c) learning curve
rows = []
uids_tr = bench.uid.values[tr]
uniq = np.unique(uids_tr)
for n_fn in [50, 100, 200, 500, 1000, 2000, len(uniq)]:
    for seed in range(3):
        sel = set(np.random.default_rng(seed).choice(uniq, min(n_fn, len(uniq)), replace=False))
        idx = np.array([i for i in tr if bench.uid.values[i] in sel])
        for cls in [StructuralGBM, TfidfLR]:
            d = cls().fit(S[idx], D[idx], Y[idx])
            s = summarise(Y[te], d.predict_proba(S[te], D[te]))
            rows.append({"n_functions": min(n_fn, len(uniq)), "seed": seed, "detector": cls.name, **s})
lc = pd.DataFrame(rows)
save_csv(lc, "exp5_learning_curve.csv")
print(lc.groupby(["detector", "n_functions"]).auroc.mean().round(3))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for det, c in [("struct-GBM", "tab:red"), ("tfidf-LR", "tab:blue")]:
    g = noise_df[noise_df.detector == det].groupby("noise").auroc.mean()
    axes[0].plot(g.index, g.values, marker="o", color=c, label=det)
    g = lc[lc.detector == det].groupby("n_functions").auroc.mean()
    axes[1].plot(g.index, g.values, marker="o", color=c, label=det)
axes[0].set_xlabel("training label noise"); axes[0].set_ylabel("test AUROC"); axes[0].legend()
axes[1].set_xscale("log"); axes[1].set_xlabel("# training functions"); axes[1].set_ylabel("test AUROC")
fig.suptitle("Exp5: robustness to label noise and data size")
fig.tight_layout(); fig.savefig(fig_path("exp5_noise_and_curve.png"), dpi=130)
print("done")
