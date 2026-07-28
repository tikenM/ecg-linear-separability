"""
Leakage-safe modelling (fix #1).

Every state-bearing step -- imputation, scaling, mutual-information +
RFE selection, the PCA-augmentation, and SMOTE-ENN balancing -- is wrapped in
a single `imblearn` Pipeline. The Pipeline guarantees these are fit on the
TRAINING fold only. Cross-validation is always grouped by patient (record).

This replaces the original "fit SMOTE/PCA/selection on the whole dataset then
split" flow, which is the most likely source of the inflated headline number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours

from .config import RANDOM_STATE, AAMI_CLASSES


class AdaptiveSMOTEENN(BaseEstimator):
    """SMOTE-ENN whose SMOTE k_neighbors adapts to the rarest class actually
    present in the data it receives.

    Under patient-grouped CV a rare class (e.g. Fusion in MIT-BIH) can have
    only a handful of samples in a given training fold, which breaks vanilla
    SMOTE (default k_neighbors=6). Behaviour:
      * rarest class >= 2  -> SMOTE with k = min(max_k, rarest-1), then ENN
      * rarest class  < 2  -> cannot synthesise; fall back to duplicate
        oversampling (RandomOverSampler) so the fold still trains and no class
        is silently dropped.
    Duck-typed on `fit_resample`, so it slots straight into imblearn Pipeline.
    """

    def __init__(self, random_state: int = RANDOM_STATE, max_k: int = 5,
                 enn_neighbors: int = 3):
        self.random_state = random_state
        self.max_k = max_k
        self.enn_neighbors = enn_neighbors

    def fit_resample(self, X, y):
        from collections import Counter
        counts = Counter(y)
        min_count = min(counts.values())
        if min_count < 2:
            ros = RandomOverSampler(random_state=self.random_state)
            return ros.fit_resample(X, y)
        k = max(1, min(self.max_k, min_count - 1))
        sampler = SMOTEENN(
            smote=SMOTE(k_neighbors=k, random_state=self.random_state),
            enn=EditedNearestNeighbours(n_neighbors=self.enn_neighbors),
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)

    def fit(self, X, y=None):
        return self


# --------------------------------------------------------------------------- #
# Custom transformer: MI + RFE selection, then PCA on the selected subset,
# concatenated back to the full (scaled) matrix. Fit on train fold only.
# --------------------------------------------------------------------------- #
class HybridFeatureAugmenter(BaseEstimator, TransformerMixin):
    """MI + RFE + PCA augmentation with robust handling of small feature sets
    (important for ablation studies where only one family is kept).
    """

    def __init__(self, k_mi: int = 50, k_rfe: int = 30, n_pca: int = 5,
                 random_state: int = RANDOM_STATE):
        self.k_mi = k_mi
        self.k_rfe = k_rfe
        self.n_pca = n_pca
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_feat = X.shape[1]

        k_mi = min(self.k_mi, n_feat)
        self.mi_ = SelectKBest(mutual_info_classif, k=k_mi).fit(X, y)
        mi_idx = np.where(self.mi_.get_support())[0]

        k_rfe = min(self.k_rfe, len(mi_idx))
        rfe_est = LinearSVC(C=0.1, dual="auto", random_state=self.random_state, max_iter=5000)
        self.rfe_ = RFE(rfe_est, n_features_to_select=k_rfe).fit(X[:, mi_idx], y)
        self.rfe_idx_ = mi_idx[self.rfe_.support_]

        n_rfe = len(self.rfe_idx_)

        # Safe n_components for arpack: must be < min(n_samples, n_features)
        n_pca = min(self.n_pca, n_rfe - 1) if n_rfe > 1 else 0

        if n_pca >= 1:
            X_rfe = X[:, self.rfe_idx_]
            self.pca_ = PCA(
                n_components=n_pca,
                random_state=self.random_state,
                svd_solver="arpack",
                tol=1e-4
            )
            self.pca_.fit(X_rfe)
            self._use_pca = True
        else:
            self.pca_ = None
            self._use_pca = False

        self.n_features_in_ = n_feat
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if getattr(self, "_use_pca", False) and self.pca_ is not None:
            pcs = self.pca_.transform(X[:, self.rfe_idx_])
            return np.hstack([X, pcs])
        else:
            return X


def make_classifier(name: str):
    name = name.lower()
    if name in ("lr", "logreg", "logistic"):
        # liblinear (as in the manuscript) needs an explicit OvR wrapper for
        # multiclass in recent sklearn; OvR also yields clean per-class coef_.
        from sklearn.multiclass import OneVsRestClassifier
        return OneVsRestClassifier(
            LogisticRegression(solver="liblinear", class_weight="balanced",
                               C=1.0, max_iter=2000, random_state=RANDOM_STATE))
    if name in ("svc", "linsvc", "linearsvc"):
        return LinearSVC(C=0.1, class_weight="balanced", dual="auto",
                         random_state=RANDOM_STATE, max_iter=5000)
    if name in ("dt", "tree"):
        return DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    raise ValueError(name)


def build_pipeline(classifier_name: str = "svc", *, use_smote: bool = True,
                   use_augment: bool = True, k_mi: int = 50, k_rfe: int = 30,
                   n_pca: int = 5) -> ImbPipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if use_augment:
        steps.append(("augment", HybridFeatureAugmenter(k_mi, k_rfe, n_pca)))
    if use_smote:
        steps.append(("balance", AdaptiveSMOTEENN(random_state=RANDOM_STATE)))
    steps.append(("clf", make_classifier(classifier_name)))
    return ImbPipeline(steps)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true, y_pred, classes=AAMI_CLASSES) -> dict:
    present = [c for c in classes if c in set(y_true)]
    rec = recall_score(y_true, y_pred, labels=present, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": {c: float(r) for c, r in zip(present, rec)},
        "confusion": confusion_matrix(y_true, y_pred, labels=present).tolist(),
        "labels": present,
    }


@dataclass
class EvalResult:
    name: str
    metrics: dict
    model_size_kb: float | None = None


def evaluate_interpatient(pipe, X_tr, y_tr, X_te, y_te, name="model") -> EvalResult:
    """Fit on the (inter-patient) training set, score on the disjoint test set."""
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return EvalResult(name=name, metrics=_metrics(y_te, pred),
                      model_size_kb=classifier_size_kb(pipe))


def cross_val_grouped(make_pipe, X, y, groups, n_splits: int = 5) -> dict:
    """Patient-grouped CV. `make_pipe` is a zero-arg factory returning a fresh
    pipeline each fold (so nothing leaks across folds)."""
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    accs, f1w, f1m = [], [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipe()
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1w.append(f1_score(y[te], pred, average="weighted", zero_division=0))
        f1m.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return {
        "acc": _summ(accs), "f1_weighted": _summ(f1w), "f1_macro": _summ(f1m),
        "folds": n_splits,
    }


def _summ(vals) -> dict:
    from scipy import stats
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t = stats.t.ppf(0.975, n - 1)
        half = t * sd / np.sqrt(n)
    else:
        half = 0.0
    return {"mean": mean, "std": sd, "ci95": [mean - half, mean + half]}


# --------------------------------------------------------------------------- #
# Model size (the transparency/footprint story, fix #2)
# --------------------------------------------------------------------------- #
def classifier_size_kb(pipe) -> float:
    """Serialized size of the *decision* parameters only (coef_/intercept_ or
    tree), i.e. what must live on the device once features are computed."""
    import io, joblib
    clf = pipe.named_steps.get("clf", pipe)
    buf = io.BytesIO()
    # store just the fitted estimator's learned arrays where possible
    payload = {}
    for attr in ("coef_", "intercept_", "classes_", "tree_"):
        if hasattr(clf, attr):
            payload[attr] = getattr(clf, attr)
    joblib.dump(payload if payload else clf, buf)
    return len(buf.getvalue()) / 1024.0