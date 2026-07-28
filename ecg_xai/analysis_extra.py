"""
Extra experiments that supply empirical support for the formal results in the
Methodology. Each function maps to a proposition or lemma.

  leakage_stress_test     -> Proposition 4 (leakage-free estimation)
  nonlinear_headroom      -> Proposition 3 (kernel-gap certificate)
  causal_vs_acausal_graph -> Proposition 1 (causality of the feature map)
  runtime_profile         -> Lemma (complexity)
  window_size_sweep       -> parameter W justification and latency tradeoff
  alignment_significance  -> robustness of Definition 3 (explanation alignment)

The dataset is not bundled. Run these on a machine with the PhysioNet records.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split

from .config import RANDOM_STATE
from .config import GRAPH_WINDOW_BEATS
from . import features as featmod
from .pipeline import build_pipeline, evaluate_interpatient, _metrics


# --------------------------------------------------------------------------- #
# Proposition 4: leakage stress test
# --------------------------------------------------------------------------- #
def leakage_stress_test(X_tr, y_tr, X_te, y_te, X_all, y_all,
                        classifier: str = "svc") -> dict:
    """Quantify the optimism removed by the leakage-free, inter-patient
    protocol. Two numbers are produced on the SAME features.

    safe  : pipeline fit on the inter-patient training partition only,
            evaluated on the patient-disjoint test partition.
    leaky : the flawed protocol. Standardization, selection, PCA, and
            SMOTE-ENN are fit on the FULL pooled dataset, and the split is a
            random beat-level (intra-patient) split. This reproduces the
            common practice that inflates reported accuracy.
    """
    safe = evaluate_interpatient(build_pipeline(classifier), X_tr, y_tr,
                                 X_te, y_te, name="safe").metrics

    # leaky: fit the whole transform cascade on all data, then random split
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.impute import SimpleImputer
    from .pipeline import HybridFeatureAugmenter, AdaptiveSMOTEENN, make_classifier
    pre = ImbPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("augment", HybridFeatureAugmenter()),
        ("balance", AdaptiveSMOTEENN(random_state=RANDOM_STATE)),
    ])
    Xb, yb = pre.fit_resample(X_all, y_all)          # leakage: fit on everything
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        Xb, yb, test_size=0.2, stratify=yb, random_state=RANDOM_STATE)
    clf = make_classifier(classifier).fit(Xtr2, ytr2)
    leaky = _metrics(yte2, clf.predict(Xte2))

    # return {
    #     "safe": {k: safe[k] for k in ("acc", "f1_weighted", "f1_macro")},
    #     "leaky": {k: leaky[k] for k in ("acc", "f1_weighted", "f1_macro")},
    #     "inflation": {
    #         "acc": round(leaky["acc"] - safe["acc"], 4),
    #         "f1_weighted": round(leaky["f1_weighted"] - safe["f1_weighted"], 4),
    #         "f1_macro": round(leaky["f1_macro"] - safe["f1_macro"], 4),
    #     },
    # }

    return {
        "safe": {k: safe[k] for k in ("acc", "f1_weighted", "f1_macro", "f1_macro_4")},
        "leaky": {k: leaky[k] for k in ("acc", "f1_weighted", "f1_macro", "f1_macro_4")},
        "inflation": {
            "acc": round(leaky["acc"] - safe["acc"], 4),
            "f1_weighted": round(leaky["f1_weighted"] - safe["f1_weighted"], 4),
            "f1_macro": round(leaky["f1_macro"] - safe["f1_macro"], 4),
            "f1_macro_4": round(leaky["f1_macro_4"] - safe["f1_macro_4"], 4),
        },
    }


# --------------------------------------------------------------------------- #
# Proposition 3: nonlinear headroom over the linear baseline
# --------------------------------------------------------------------------- #
def nonlinear_headroom(X, y, groups, n_splits: int = 5,
                       gammas=("scale", 0.01, 0.1, 1.0)) -> dict:
    """Grouped-CV accuracy of several nonlinear models and an RBF gamma grid,
    each reported as a gap over the linear classifier. Small gaps across all of
    them support the near-linear-separability claim (Proposition 3)."""
    Xs = StandardScaler().fit_transform(np.asarray(X, float))
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    def cv_acc(make_model):
        accs = []
        for tr, te in gkf.split(Xs, y, groups):
            m = make_model().fit(Xs[tr], y[tr])
            accs.append(accuracy_score(y[te], m.predict(Xs[te])))
        return float(np.mean(accs))

    lin = cv_acc(lambda: LinearSVC(C=1.0, dual="auto", max_iter=10000))
    out = {"linear_acc": round(lin, 4), "models": {}}
    for g in gammas:
        out["models"][f"rbf_gamma_{g}"] = cv_acc(
            lambda g=g: SVC(kernel="rbf", C=1.0, gamma=g))
    out["models"]["knn_k15"] = cv_acc(lambda: KNeighborsClassifier(n_neighbors=15))
    out["models"]["random_forest"] = cv_acc(
        lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    out["models"]["mlp"] = cv_acc(
        lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                              random_state=RANDOM_STATE))
    out["gap_over_linear"] = {k: round(v - lin, 4) for k, v in out["models"].items()}
    out["max_gap"] = round(max(out["gap_over_linear"].values()), 4)
    return out


# --------------------------------------------------------------------------- #
# Proposition 1: causal vs acausal graph features
# --------------------------------------------------------------------------- #
# def causal_vs_acausal_graph(train_beats, test_beats, classifier="svc") -> dict:
#     """Re-featurize with causal and acausal graph/HRV context, then run the
#     inter-patient evaluation for each. A positive acausal-minus-causal gap is
#     the leakage that the causal construction prevents (Proposition 1)."""
#     Xc_tr, *_ = featmod.build_feature_matrix(train_beats, causal=True)
#     Xc_te, *_ = featmod.build_feature_matrix(test_beats, causal=True)
#     Xa_tr, *_ = featmod.build_feature_matrix(train_beats, causal=False)
#     Xa_te, *_ = featmod.build_feature_matrix(test_beats, causal=False)
#     ytr, yte = train_beats.labels, test_beats.labels

#     causal = evaluate_interpatient(build_pipeline(classifier), Xc_tr, ytr,
#                                    Xc_te, yte, name="causal").metrics
#     acausal = evaluate_interpatient(build_pipeline(classifier), Xa_tr, ytr,
#                                     Xa_te, yte, name="acausal").metrics
#     return {
#         "causal": {k: causal[k] for k in ("acc", "f1_weighted", "f1_macro")},
#         "acausal": {k: acausal[k] for k in ("acc", "f1_weighted", "f1_macro")},
#         "leakage_gap_f1_weighted": round(
#             acausal["f1_weighted"] - causal["f1_weighted"], 4),
#     }

# --------------------------------------------------------------------------- #
# Proposition 1: causal vs acausal graph features
# --------------------------------------------------------------------------- #
def causal_vs_acausal_graph(train_beats, test_beats, classifier="svc",
                            window: int = 64) -> dict:
    """
    Improved causal vs acausal comparison using a symmetric window.

    Instead of using the full record (which is O(n²) and too slow on real data),
    we use a symmetric window of past + future beats for the acausal case.
    This still demonstrates the effect of future-beat leakage while remaining
    computationally feasible.
    """
    print("[extra] causal vs acausal graph (Prop. 1) ...")

    # Causal version (trailing window only - no future leakage)
    print(f"  Computing causal features (window={window})...")
    Xc_tr, *_ = featmod.build_feature_matrix(train_beats, causal=True, window_beats=window)
    Xc_te, *_ = featmod.build_feature_matrix(test_beats, causal=True, window_beats=window)
    ytr, yte = train_beats.labels, test_beats.labels

    causal = evaluate_interpatient(build_pipeline(classifier), Xc_tr, ytr,
                                   Xc_te, yte, name="causal").metrics

    # Acausal version using symmetric window (includes future beats)
    print(f"  Computing acausal features with symmetric window (±{window} beats)...")
    Xa_tr, *_ = featmod.build_feature_matrix(train_beats, causal=False, window_beats=window)
    Xa_te, *_ = featmod.build_feature_matrix(test_beats, causal=False, window_beats=window)

    acausal = evaluate_interpatient(build_pipeline(classifier), Xa_tr, ytr,
                                    Xa_te, yte, name="acausal").metrics

    # return {
    #     "causal": {k: causal[k] for k in ("acc", "f1_weighted", "f1_macro")},
    #     "acausal": {k: acausal[k] for k in ("acc", "f1_weighted", "f1_macro")},
    #     "leakage_gap_f1_weighted": round(
    #         acausal["f1_weighted"] - causal["f1_weighted"], 4),
    #     "window_size": window,
    #     "note": f"Acausal used symmetric window of ±{window} beats"
    # }

    return {
        "causal": {k: causal[k] for k in ("acc", "f1_weighted", "f1_macro", "f1_macro_4")},
        "acausal": {k: acausal[k] for k in ("acc", "f1_weighted", "f1_macro", "f1_macro_4")},
        "leakage_gap_f1_weighted": round(
            acausal["f1_weighted"] - causal["f1_weighted"], 4),
        "leakage_gap_f1_macro_4": round(
            acausal["f1_macro_4"] - causal["f1_macro_4"], 4),
        "window_size": window,
        "note": f"Acausal used symmetric window of ±{window} beats"
    }


# --------------------------------------------------------------------------- #
# Complexity lemma: measured runtime and footprint
# --------------------------------------------------------------------------- #
def runtime_profile(beats, n_beats: int = 500) -> dict:
    """Measure per-beat feature time by family and per-beat classifier
    inference time, to support the complexity lemma with measured numbers."""
    from .features import (_time_features, _freq_features, _morph_features,
                           _wavelet_features, _compact_descriptor, _graph_window,
                           _hrv_window)
    n = min(n_beats, len(beats))
    segs = beats.segments[:n]

    def timed(fn, reps=1):
        t0 = time.perf_counter()
        for _ in range(reps):
            for s in segs:
                fn(s)
        return (time.perf_counter() - t0) / (reps * n) * 1e3  # ms/beat

    prof = {
        "time_ms": timed(_time_features),
        "freq_ms": timed(lambda s: _freq_features(s)),
        "morph_ms": timed(_morph_features),
        "wavelet_ms": timed(lambda s: _wavelet_features(s)),
    }
    # graph cost on a window of descriptors
    descr = np.stack([_compact_descriptor(s) for s in segs], axis=0)
    # W = min(32, n)
    W = min(GRAPH_WINDOW_BEATS, n)
    t0 = time.perf_counter()
    for i in range(n):
        lo = max(0, i - W + 1)
        _graph_window(descr[lo:i + 1])
    prof["graph_ms"] = (time.perf_counter() - t0) / n * 1e3
    prof["feature_total_ms"] = round(sum(prof.values()), 4)

    # classifier inference time and footprint
    X, names, family, _ = featmod.build_feature_matrix(beats.subset(
        np.arange(min(len(beats), 3000))))
    y = beats.labels[:X.shape[0]]
    pipe = build_pipeline("svc").fit(X, y)
    t0 = time.perf_counter()
    for _ in range(5):
        pipe.predict(X[:n])
    prof["classifier_ms_per_beat"] = (time.perf_counter() - t0) / (5 * n) * 1e3
    prof = {k: round(v, 5) for k, v in prof.items()}
    return prof


# --------------------------------------------------------------------------- #
# Graph window-size sweep
# --------------------------------------------------------------------------- #
def window_size_sweep(train_beats, test_beats, windows=(8, 16, 32, 64),
                      classifier="svc") -> dict:
    """Performance and graph latency as a function of the window size W."""
    from .features import _compact_descriptor, _graph_window
    ytr, yte = train_beats.labels, test_beats.labels
    out = {}
    descr = np.stack([_compact_descriptor(s) for s in test_beats.segments[:500]],
                     axis=0)
    for W in windows:
        Xtr, *_ = featmod.build_feature_matrix(train_beats, window_beats=W)
        Xte, *_ = featmod.build_feature_matrix(test_beats, window_beats=W)
        res = evaluate_interpatient(build_pipeline(classifier), Xtr, ytr,
                                    Xte, yte, name=f"W{W}").metrics
        # latency of the graph descriptor at this window
        t0 = time.perf_counter()
        for i in range(len(descr)):
            lo = max(0, i - W + 1)
            _graph_window(descr[lo:i + 1])
        lat = (time.perf_counter() - t0) / len(descr) * 1e3
        out[f"W={W}"] = {"f1_weighted": round(res["f1_weighted"], 4),
                         "graph_ms_per_beat": round(lat, 5)}
    return out


# --------------------------------------------------------------------------- #
# Definition 3 robustness: alignment significance
# --------------------------------------------------------------------------- #
def alignment_significance(linear_band_vec, attribution, n_boot: int = 2000,
                           n_perm: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """Bootstrap CI and permutation null for the alignment cosine.

    linear_band_vec : (B,) normalized linear band importance.
    attribution     : (n_beats, n_leads, seg_len) CNN attribution magnitudes.
    Per-beat CNN band-importance vectors are formed, the cosine to the linear
    vector is computed per beat, the mean cosine is bootstrapped for a CI, and a
    permutation null is built by shuffling the band assignment of the linear
    vector.
    """
    from .xai_alignment import cnn_band_importance
    rng = np.random.default_rng(seed)
    a_lin = np.asarray(linear_band_vec, float)

    # per-beat CNN band importance
    per_beat = np.stack(
        [cnn_band_importance(attribution[i:i + 1]) for i in range(len(attribution))],
        axis=0)

    def cos(u, v):
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

    cos_beat = np.array([cos(a_lin, per_beat[i]) for i in range(len(per_beat))])
    mean_cos = float(cos_beat.mean())

    boot = np.array([cos_beat[rng.integers(0, len(cos_beat), len(cos_beat))].mean()
                     for _ in range(n_boot)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    mean_cnn = per_beat.mean(axis=0)
    obs = cos(a_lin, mean_cnn)
    null = np.array([cos(a_lin[rng.permutation(len(a_lin))], mean_cnn)
                     for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))

    return {
        "mean_cosine": round(mean_cos, 4),
        "bootstrap_ci95": [round(c, 4) for c in ci],
        "observed_cosine": round(obs, 4),
        "permutation_p_value": round(p, 4),
        "null_mean": round(float(null.mean()), 4),
    }

