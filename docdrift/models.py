"""Detectors. Every detector exposes fit(X_src, X_doc, y) / predict_proba(X_src, X_doc)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract, code_tokens


class LexicalHeuristic:
    """Zero-training baseline: fraction of signature params NOT mentioned in the
    docstring plus a return/yield mismatch flag."""
    name = "heuristic"

    def fit(self, src, doc, y):
        return self

    def predict_proba(self, src, doc):
        F = np.stack([extract(s, d) for s, d in zip(src, doc)])
        score = (1 - F[:, 4]) + F[:, 9] + 0.5 * (F[:, 2] > 0) + 0.5 * (F[:, 14] > 0)
        return score / score.max() if score.max() > 0 else score


class StructuralLR:
    name = "struct-LR"

    def __init__(self):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    def _X(self, src, doc):
        return np.stack([extract(s, d) for s, d in zip(src, doc)])

    def fit(self, src, doc, y):
        self.m.fit(self._X(src, doc), y)
        return self

    def predict_proba(self, src, doc):
        return self.m.predict_proba(self._X(src, doc))[:, 1]

    def coef(self):
        return self.m[-1].coef_[0]


class StructuralGBM(StructuralLR):
    name = "struct-GBM"

    def __init__(self, seed=0):
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                                random_state=seed)


class TfidfLR:
    """Bag-of-identifiers over code tokens and docstring tokens (separately
    vectorised), linear model -- the 'neural-free' stand-in for a
    sequence model that sees both sides but has no alignment inductive bias."""
    name = "tfidf-LR"

    def __init__(self):
        self.vc = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
        self.vd = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3, max_features=20000, sublinear_tf=True)
        self.m = LogisticRegression(max_iter=3000, C=2.0)

    def _X(self, src, doc, fit=False):
        c = [code_tokens(s) for s in src]
        if fit:
            return hstack([self.vc.fit_transform(c), self.vd.fit_transform(doc)]).tocsr()
        return hstack([self.vc.transform(c), self.vd.transform(doc)]).tocsr()

    def fit(self, src, doc, y):
        self.m.fit(self._X(src, doc, fit=True), y)
        return self

    def predict_proba(self, src, doc):
        return self.m.predict_proba(self._X(src, doc))[:, 1]


class Hybrid(TfidfLR):
    """TF-IDF + structural features in one gradient-boosted model."""
    name = "hybrid-GBM"

    def __init__(self, seed=0):
        super().__init__()
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                                random_state=seed)
        from sklearn.decomposition import TruncatedSVD
        self.svd = TruncatedSVD(64, random_state=seed)

    def _X2(self, src, doc, fit=False):
        T = self._X(src, doc, fit=fit)
        Z = self.svd.fit_transform(T) if fit else self.svd.transform(T)
        S = np.stack([extract(s, d) for s, d in zip(src, doc)])
        return np.hstack([Z, S])

    def fit(self, src, doc, y):
        self.m.fit(self._X2(src, doc, fit=True), y)
        return self

    def predict_proba(self, src, doc):
        return self.m.predict_proba(self._X2(src, doc))[:, 1]


DETECTORS = {"heuristic": LexicalHeuristic, "struct-LR": StructuralLR, "struct-GBM": StructuralGBM,
             "tfidf-LR": TfidfLR, "hybrid-GBM": Hybrid}
