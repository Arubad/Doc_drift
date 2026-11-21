"""Experiment 10 -- Doc-side synthetic drift (protocol Addendum A, hypothesis H4).

Build a benchmark whose drift lives in the DOCSTRING (code fixed), train the same
four detector families on it, and evaluate on the real doc-fix pairs (exp8b) with the
same leakage control and bootstrap as exp9. H4 must hold on the length-matched subset.

Outputs: data/doc_benchmark.pkl, results/exp10_*.csv
"""
import hashlib
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, load_benchmark, save_csv
from docdrift.docmutations import (DOC_DRIFT, DOC_BENIGN, DOC_INTERFACE, apply_doc_mutation,
                                   build_pools, render)
from docdrift.features import extract, code_tokens

B = int(os.environ.get("DOCDRIFT_BOOT", "1000"))
QUICK = bool(os.environ.get("DOCDRIFT_QUICK"))
K_POS, K_NEG = 3, 2
RNG = np.random.default_rng(0)
FAMILIES = ["struct-LR", "struct-GBM", "tfidf-LR", "hybrid-GBM"]

# ----------------------------------------------------------------- 1. doc-side benchmark
corpus = pd.read_pickle(os.path.join(DATA, "corpus.pkl"))
if QUICK:
    corpus = corpus.sample(1500, random_state=0)
pools = build_pools(corpus.source)
rows, applic = [], {op: [0, 0] for op in DOC_DRIFT + DOC_BENIGN}
for r in corpus.itertuples():
    seed = int(hashlib.md5(r.uid.encode()).hexdigest()[:8], 16)
    clean = render(r.source)
    if clean is None:
        continue
    import ast as _ast
    clean_doc = _ast.get_docstring(_ast.parse(clean).body[0])
    rows.append({"uid": r.uid, "project": r.project, "source": clean, "docstring": clean_doc,
                 "label": 0, "mutation": "none", "kind": "clean"})
    local = np.random.default_rng(seed)
    for ops, k, label, kind in [(DOC_DRIFT, K_POS, 1, "doc-drift"), (DOC_BENIGN, K_NEG, 0, "doc-benign")]:
        order = list(ops); local.shuffle(order)
        n = 0
        for op in order:
            out = apply_doc_mutation(r.source, op, seed, pools)
            applic[op][1] += 1
            if out is None:
                continue
            applic[op][0] += 1
            if n < k:
                rows.append({"uid": r.uid, "project": r.project, "source": out[0], "docstring": out[1],
                             "label": label, "mutation": op, "kind": kind})
                n += 1
docb = pd.DataFrame(rows).reset_index(drop=True)
docb.to_pickle(os.path.join(DATA, "doc_benchmark.pkl"))
save_csv(pd.DataFrame([{"mutation": op, "applicable": a, "attempted": t, "rate": a / t if t else 0}
                       for op, (a, t) in applic.items()]), "exp10_applicability.csv")
print(docb.groupby(["kind", "mutation"]).size().to_string(), flush=True)
words = docb.docstring.str.split().str.len()
print("mean docstring words by label:", words.groupby(docb.label).mean().round(1).to_dict(), flush=True)

# ----------------------------------------------------------------- 2. real pairs
real = pd.read_pickle(os.path.join(DATA, "real_pairs_clean.pkl")).reset_index(drop=True)
wo, wn = real.doc_old.str.split().str.len(), real.doc_new.str.split().str.len()
real["length_matched"] = (wn - wo).abs() <= 0.10 * wo
SUBSETS = {"interface": set(real.pair_id[real.interface]),
           "interface-length-matched": set(real.pair_id[real.interface & real.length_matched]),
           "all": set(real.pair_id)}
n_real = len(real)
R = pd.concat([pd.DataFrame({"pair_id": real.pair_id, "project": real.project, "source": real.source_old,
                             "docstring": real.doc_old, "label": 1}),
               pd.DataFrame({"pair_id": real.pair_id, "project": real.project, "source": real.source_new,
                             "docstring": real.doc_new, "label": 0})], ignore_index=True)

# ----------------------------------------------------------------- 3. training pools
code = load_benchmark().reset_index(drop=True)
if QUICK:
    code = code[code.uid.isin(docb.uid)].reset_index(drop=True)
code = code.assign(kind=np.where(code.kind == "clean", "code-clean", code.kind))
TRAIN = pd.concat([docb.assign(side="doc"), code.assign(side="code")], ignore_index=True)
print(f"training rows: doc {len(docb)}, code {len(code)}; real pairs {n_real}", flush=True)


def feats(src, doc):
    return np.stack([extract(s, d) for s, d in zip(src, doc)])


F_T, F_R = feats(TRAIN.source, TRAIN.docstring), feats(R.source, R.docstring)
vc = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
vd = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
T_T = hstack([vc.fit_transform([code_tokens(s) for s in TRAIN.source]), vd.fit_transform(TRAIN.docstring)]).tocsr()
T_R = hstack([vc.transform([code_tokens(s) for s in R.source]), vd.transform(R.docstring)]).tocsr()
svd = TruncatedSVD(64, random_state=0).fit(T_T)
X = {"struct": (F_T, F_R), "tfidf": (T_T, T_R),
     "hybrid": (np.hstack([svd.transform(T_T), F_T]), np.hstack([svd.transform(T_R), F_R]))}
Y, SIDE, MUT, PROJ = TRAIN.label.values, TRAIN.side.values, TRAIN.mutation.values, TRAIN.project.values


def make(f):
    return {"struct-LR": ("struct", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            "struct-GBM": ("struct", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)),
            "tfidf-LR": ("tfidf", lambda: LogisticRegression(max_iter=3000, C=2.0)),
            "hybrid-GBM": ("hybrid", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0)),
            }[f]


REGIMES = {"doc-all": SIDE == "doc", "doc+code-all": np.ones(len(TRAIN), bool)}

real_projects = np.array(sorted(R.project.unique())); RNG.shuffle(real_projects)
fold_of = {p: i % 5 for i, p in enumerate(real_projects)}
R_fold = R.project.map(fold_of).values


def pair_correct(score):
    d = score[:n_real] - score[n_real:]
    return pd.Series((d > 0) + 0.5 * (d == 0), index=real.pair_id.values)


# ----------------------------------------------------------------- 4. train / evaluate
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
doc_idx = np.where(SIDE == "doc")[0]
tr_uid, te_uid = next(gss.split(doc_idx, Y[doc_idx], groups=TRAIN.uid.values[doc_idx]))
d_tr, d_te = doc_idx[tr_uid], doc_idx[te_uid]
neg_te = d_te[Y[d_te] == 0]
results, scores = [], {}
for regime, mask in REGIMES.items():
    for fam in FAMILIES:
        key, ctor = make(fam)
        name = f"{fam}|{regime}"
        # synthetic in-distribution on doc-side test functions
        tr = np.setdiff1d(np.where(mask)[0], d_te)
        tr = tr[~np.isin(TRAIN.uid.values[tr], TRAIN.uid.values[d_te])]
        m = ctor().fit(X[key][0][tr], Y[tr])
        p = pd.Series(m.predict_proba(X[key][0][d_te])[:, 1], index=d_te)
        indist = roc_auc_score(Y[d_te], p[d_te])
        # LOMO over doc-side drift operators (doc-all only)
        lomo = []
        if regime == "doc-all" and not QUICK:
            for op in DOC_DRIFT:
                keep = tr[MUT[tr] != op]
                mm = ctor().fit(X[key][0][keep], Y[keep])
                pos = d_te[MUT[d_te] == op]
                pp = mm.predict_proba(X[key][0][np.r_[pos, neg_te]])[:, 1]
                lomo.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg_te))], pp))
        # real pairs, package-held-out
        score = np.zeros(len(R))
        for k in range(5):
            held = {p for p, f in fold_of.items() if f == k}
            trk = np.where(mask & ~np.isin(PROJ, list(held)))[0]
            mk = R_fold == k
            if mk.any():
                score[mk] = ctor().fit(X[key][0][trk], Y[trk]).predict_proba(X[key][1][mk])[:, 1]
        scores[name] = score
        c = pair_correct(score)
        row = {"variant": name, "family": fam, "regime": regime, "indist_auroc": indist,
               "lomo_auroc": float(np.mean(lomo)) if lomo else np.nan}
        for s, ids in SUBSETS.items():
            row[f"pair_acc_{s}"] = float(c[c.index.isin(ids)].mean())
        results.append(row)
        print(f"{name:28s} in-dist {indist:.3f}  LOMO {row['lomo_auroc']:.3f}  real interface "
              f"{row['pair_acc_interface']:.3f}  len-matched {row['pair_acc_interface-length-matched']:.3f}", flush=True)

# baselines: doc length (shorter = drifted), zero-training heuristic
L = R.docstring.str.split().str.len().values.astype(float)
from docdrift.models import LexicalHeuristic
for name, sc in [("baseline:shorter-doc", -L),
                 ("heuristic", LexicalHeuristic().predict_proba(R.source.values, R.docstring.values))]:
    scores[name] = sc
    c = pair_correct(sc)
    results.append({"variant": name, "family": name, "regime": "-", "indist_auroc": np.nan, "lomo_auroc": np.nan,
                    **{f"pair_acc_{s}": float(c[c.index.isin(ids)].mean()) for s, ids in SUBSETS.items()}})
res = pd.DataFrame(results)
save_csv(res, "exp10_variants.csv")
pd.to_pickle(scores, os.path.join(DATA, "exp10_real_scores.pkl"))

# ----------------------------------------------------------------- 5. bootstrap: CI + paired diff vs best code-side
code_scores = pd.read_pickle(os.path.join(DATA, "exp9_real_scores.pkl")) if not QUICK else {}
code_var = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "results", "exp9_variants.csv")) if not QUICK else None
commit_of = real.set_index("pair_id").commit
out = []
for s, ids in SUBSETS.items():
    ids = np.array(sorted(ids))
    commits = commit_of.loc[ids].unique()
    by_c = {c: ids[commit_of.loc[ids].values == c] for c in commits}
    draws = [np.concatenate([by_c[c] for c in RNG.choice(commits, len(commits))]) for _ in range(B)]
    for name in res.variant:
        c = pair_correct(scores[name])
        boot = np.array([c.loc[d].mean() for d in draws])
        rec = {"subset": s, "variant": name, "n_pairs": len(ids), "pair_acc": float(c.loc[ids].mean()),
               "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))}
        fam = name.split("|")[0]
        if code_var is not None and fam in FAMILIES:
            best = code_var[code_var.family == fam].sort_values("real_pair_acc_interface").variant.iloc[-1]
            cb = pair_correct(code_scores[best])
            diff = np.array([c.loc[d].mean() - cb.loc[d].mean() for d in draws])
            rec.update({"best_code_variant": best, "diff_vs_best_code": float(c.loc[ids].mean() - cb.loc[ids].mean()),
                        "diff_ci_lo": float(np.percentile(diff, 2.5)), "diff_ci_hi": float(np.percentile(diff, 97.5))})
        out.append(rec)
ci = pd.DataFrame(out)
save_csv(ci, "exp10_real_ci.csv")
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
print(ci.round(3).to_string(index=False))
