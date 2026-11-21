"""Experiment 9 -- Does synthetic-benchmark performance predict REAL drift detection?

Pre-registered hypothesis (written before looking at real-pair results):
  H-real: across detector variants, LOMO AUROC on the synthetic benchmark is a
  better predictor (higher Spearman rho) of accuracy on real doc-fix pairs than
  in-distribution AUROC is.

Detector variants = 4 model families x 7 training regimes (which mutation
operators supply the positives) = 28 trained variants, + the zero-training
heuristic. For each variant we compute on the synthetic benchmark
  * in-dist AUROC  -- held-out functions, same operators as training;
  * LOMO AUROC     -- mean over operators in the regime, each held out in turn;
and on the real doc-fix pairs (exp8)
  * pair accuracy  -- P(score(old docstring) > score(new docstring)), ties = 0.5;
  * AUROC          -- old (label 1) vs new (label 0) over all real rows.

Leakage control: the real pairs come from packages that are also in the
synthetic corpus. Real-pair scores are produced by models trained on the
synthetic benchmark *minus every function of the pair's package* (5 folds
over packages), so no model has seen the fixed version of a function.

Real pairs are evaluated on two subsets (exp8b): `interface` -- the edit
touches a parameter / return / raise / type / default claim (headline, since a
static detector can only verify interface claims) -- and `all` clean pairs.

Uncertainty: cluster bootstrap over commits (B=1000) for pair accuracy and for
the Spearman correlations (variants fixed, real pairs resampled).

Implementation note: features / TF-IDF / SVD are computed once (unsupervised,
fit on all synthetic text) and cached; model hyper-parameters match
docdrift/models.py.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from common import DATA, load_benchmark, save_csv
from docdrift.features import extract, code_tokens
from docdrift.mutations import SURFACE, BEHAVIOURAL, SEMANTIC

B = int(os.environ.get("DOCDRIFT_BOOT", "1000"))
RNG = np.random.default_rng(0)

REGIMES = {
    "all": SURFACE | BEHAVIOURAL | SEMANTIC,
    "surface": SURFACE,
    "behavioural": BEHAVIOURAL,
    "semantic": SEMANTIC,
    "surface+behavioural": SURFACE | BEHAVIOURAL,
    "interface": {"rename_param", "add_param", "remove_param", "return_to_yield", "add_raise", "remove_raise"},
    "all-no-benign-neg": SURFACE | BEHAVIOURAL | SEMANTIC,   # negatives = clean only
}
FAMILIES = ["struct-LR", "struct-GBM", "tfidf-LR", "hybrid-GBM"]

# ----------------------------------------------------------------- data
bench = load_benchmark().reset_index(drop=True)
real = pd.read_pickle(os.path.join(DATA, "real_pairs_clean.pkl")).reset_index(drop=True)
if os.environ.get("DOCDRIFT_QUICK"):          # smoke test: tiny subsample, 2 families
    keep_uid = bench.uid.drop_duplicates().sample(1500, random_state=0)
    bench = bench[bench.uid.isin(keep_uid)].reset_index(drop=True)
    real = real.sample(min(300, len(real)), random_state=0).reset_index(drop=True)
    FAMILIES = ["struct-LR", "tfidf-LR"]
    REGIMES = {k: REGIMES[k] for k in ["all", "surface", "behavioural"]}
    B = 20
print(f"synthetic rows {len(bench)}, real pairs {len(real)} from {real.repo.nunique()} repos")
R = pd.concat([
    pd.DataFrame({"pair_id": real.pair_id, "commit": real.commit, "project": real.project,
                  "source": real.source_old, "docstring": real.doc_old, "label": 1}),
    pd.DataFrame({"pair_id": real.pair_id, "commit": real.commit, "project": real.project,
                  "source": real.source_new, "docstring": real.doc_new, "label": 0}),
], ignore_index=True)


def feats(src, doc):
    return np.stack([extract(s, d) for s, d in zip(src, doc)])


print("features ...", flush=True)
Fb, Fr = feats(bench.source, bench.docstring), feats(R.source, R.docstring)
vc = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
vd = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
Tb = hstack([vc.fit_transform([code_tokens(s) for s in bench.source]), vd.fit_transform(bench.docstring)]).tocsr()
Tr = hstack([vc.transform([code_tokens(s) for s in R.source]), vd.transform(R.docstring)]).tocsr()
svd = TruncatedSVD(64, random_state=0).fit(Tb)
Hb, Hr = np.hstack([svd.transform(Tb), Fb]), np.hstack([svd.transform(Tr), Fr])
X = {"struct": (Fb, Fr), "tfidf": (Tb, Tr), "hybrid": (Hb, Hr)}


def make(family):
    if family == "struct-LR":
        return "struct", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if family == "struct-GBM":
        return "struct", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
    if family == "tfidf-LR":
        return "tfidf", LogisticRegression(max_iter=3000, C=2.0)
    if family == "hybrid-GBM":
        return "hybrid", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)
    raise KeyError(family)


def fit_predict(family, train_idx, Xtest):
    key, m = make(family)
    m.fit(X[key][0][train_idx], Yb[train_idx])
    return m.predict_proba(Xtest)[:, 1], key


Yb, Mb = bench.label.values, bench.mutation.values
neg_mask = (Yb == 0)
clean_mask = bench.kind.values == "clean"


def train_rows(idx, regime, ops):
    """Rows of idx usable for training under a regime restricted to ops."""
    neg = clean_mask[idx] if regime == "all-no-benign-neg" else neg_mask[idx]
    return idx[neg | np.isin(Mb[idx], list(ops))]


gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr, te = next(gss.split(bench, Yb, groups=bench.uid))
neg_te = te[neg_mask[te]]

# package folds for leakage-free real-pair scoring
real_projects = np.array(sorted(R.project.unique()))
RNG.shuffle(real_projects)
proj_fold = {p: i % 5 for i, p in enumerate(real_projects)}
R_fold = R.project.map(proj_fold).values


SUBSETS = {"interface": set(real.pair_id[real.interface]), "all": set(real.pair_id)}


def pair_correct(score):
    """Per-pair correctness: 1 if old (drifted) scores above new, 0.5 on ties."""
    s = pd.DataFrame({"pair_id": R.pair_id, "label": R.label, "s": score})
    w = s.pivot_table(index="pair_id", columns="label", values="s")
    d = w[1] - w[0]
    return (d > 0) + 0.5 * (d == 0)


def pair_acc(score, subset="interface"):
    c = pair_correct(score)
    return float(c[c.index.isin(SUBSETS[subset])].mean())


def sub_auroc(score, subset="interface"):
    m = R.pair_id.isin(SUBSETS[subset]).values
    return float(roc_auc_score(R.label[m], score[m]))


# ----------------------------------------------------------------- run variants
rows, real_scores = [], {}
for regime, ops in REGIMES.items():
    for fam in FAMILIES:
        name = f"{fam}|{regime}"
        key = make(fam)[0]
        # in-distribution
        p, _ = fit_predict(fam, train_rows(tr, regime, ops), X[key][0][te])
        pt = pd.Series(p, index=te)
        pos = te[np.isin(Mb[te], list(ops))]
        indist = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg_te))], np.r_[pt[pos], pt[neg_te]])
        # LOMO within the regime
        lomo = []
        for op in sorted(ops):
            p, _ = fit_predict(fam, train_rows(tr, regime, ops - {op}), X[key][0][te])
            pt = pd.Series(p, index=te)
            pos = te[Mb[te] == op]
            if len(pos):
                lomo.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg_te))], np.r_[pt[pos], pt[neg_te]]))
        # real pairs, package-held-out
        score = np.zeros(len(R))
        all_idx = np.arange(len(bench))
        for k in range(5):
            held = set(p for p, f in proj_fold.items() if f == k)
            trk = train_rows(all_idx[~bench.project.isin(held).values], regime, ops)
            mk = R_fold == k
            if mk.any():
                score[mk] = fit_predict(fam, trk, X[key][1][mk])[0]
        real_scores[name] = score
        rows.append({"variant": name, "family": fam, "regime": regime, "indist_auroc": indist,
                     "lomo_auroc": float(np.mean(lomo)),
                     **{f"real_pair_acc_{k}": pair_acc(score, k) for k in SUBSETS},
                     **{f"real_auroc_{k}": sub_auroc(score, k) for k in SUBSETS}})
        print(f"{name:40s} in-dist {indist:.3f}  LOMO {np.mean(lomo):.3f}  "
              f"real-pair interface {rows[-1]['real_pair_acc_interface']:.3f} all {rows[-1]['real_pair_acc_all']:.3f}",
              flush=True)

# zero-training heuristic (reference only, not in the correlation)
from docdrift.models import LexicalHeuristic
h = LexicalHeuristic().predict_proba(R.source.values, R.docstring.values)
real_scores["heuristic"] = h
rows.append({"variant": "heuristic", "family": "heuristic", "regime": "-", "indist_auroc": np.nan,
             "lomo_auroc": np.nan, **{f"real_pair_acc_{k}": pair_acc(h, k) for k in SUBSETS},
             **{f"real_auroc_{k}": sub_auroc(h, k) for k in SUBSETS}})

res = pd.DataFrame(rows)
save_csv(res, "exp9_variants.csv")
pd.to_pickle(real_scores, os.path.join(DATA, "exp9_real_scores.pkl"))

# ----------------------------------------------------------------- correlation + bootstrap
V = res[res.family != "heuristic"].reset_index(drop=True)
W_all = pd.DataFrame({n: pair_correct(real_scores[n]) for n in V.variant})   # pairs x variants
pair_commit = real.set_index("pair_id").commit
ci = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
summ_rows, acc_rows = [], []
for subset, ids_set in SUBSETS.items():
    W = W_all.loc[sorted(ids_set)]
    pc = pair_commit.loc[W.index]
    commits = pc.unique()
    by_commit = {c: pc.index[pc.values == c].values for c in commits}
    y = V[f"real_pair_acc_{subset}"].values
    rho_in = spearmanr(V.indist_auroc, y).statistic
    rho_lo = spearmanr(V.lomo_auroc, y).statistic
    boot = []
    for _ in range(B):
        cs = RNG.choice(commits, size=len(commits), replace=True)
        ids = np.concatenate([by_commit[c] for c in cs])
        acc_b = W.loc[ids].mean().reindex(V.variant).values
        r_in = spearmanr(V.indist_auroc, acc_b).statistic
        r_lo = spearmanr(V.lomo_auroc, acc_b).statistic
        boot.append((r_in, r_lo, r_lo - r_in, *acc_b))
    boot = np.array(boot)
    for stat, val, col in [("spearman(indist, real)", rho_in, 0), ("spearman(lomo, real)", rho_lo, 1),
                           ("rho_lomo - rho_indist", rho_lo - rho_in, 2)]:
        lo, hi = ci(boot[:, col])
        summ_rows.append({"subset": subset, "n_pairs": len(W), "n_commits": len(commits),
                          "stat": stat, "value": val, "ci_lo": lo, "ci_hi": hi})
    summ_rows.append({"subset": subset, "n_pairs": len(W), "n_commits": len(commits),
                      "stat": "P(rho_lomo > rho_indist)", "value": float((boot[:, 2] > 0).mean()),
                      "ci_lo": np.nan, "ci_hi": np.nan})
    for j, v in enumerate(V.variant):
        acc_rows.append({"subset": subset, "variant": v, "real_pair_acc": y[j],
                         "ci_lo": np.percentile(boot[:, 3 + j], 2.5), "ci_hi": np.percentile(boot[:, 3 + j], 97.5)})
summary = pd.DataFrame(summ_rows)
save_csv(summary, "exp9_correlation.csv")
save_csv(pd.DataFrame(acc_rows), "exp9_pair_acc_ci.csv")
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
print(summary.round(3).to_string(index=False))
