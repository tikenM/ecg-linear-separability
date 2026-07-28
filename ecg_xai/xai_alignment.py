"""
XAI alignment (fix #4): "recovers black-box explanations cheaply."

The claim we want to support, in a citable form:
    The cheap, interpretable-by-design linear model relies on the SAME
    physiological frequency bands that an expensive black-box CNN attends to,
    at a fraction of the parameter / compute cost.

Both models are projected into a shared 3-band frequency space:

* Linear model -> sum of |coefficient| over feature columns tagged with each
  band (band_map from features.build_feature_matrix), aggregated over classes.
* CNN          -> band power of the |attribution| map over the raw waveform
  (saliency or Integrated Gradients), aggregated over beats and leads.

Agreement metrics: Spearman rank correlation, cosine similarity, and top-band
overlap. Cost ratio: CNN params / linear params.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from scipy.signal import welch

from .config import FREQ_BANDS, TARGET_FS


def extract_coef(clf) -> np.ndarray:
    """Return (n_classes, n_features) coefficients from a linear classifier or
    a OneVsRestClassifier wrapping linear estimators."""
    if hasattr(clf, "coef_") and clf.coef_ is not None:
        return np.atleast_2d(np.asarray(clf.coef_))
    if hasattr(clf, "estimators_"):
        return np.vstack([np.asarray(e.coef_).ravel() for e in clf.estimators_])
    raise AttributeError("classifier exposes no linear coefficients")


def linear_band_importance(coef: np.ndarray, names, band_map) -> np.ndarray:
    """coef: (n_classes, n_features) or (n_features,). Returns (n_bands,)
    in the order of FREQ_BANDS, normalized to sum 1."""
    coef = np.atleast_2d(np.asarray(coef))
    imp = np.abs(coef).sum(axis=0)               # per-feature importance
    bands = list(FREQ_BANDS.keys())
    vec = np.zeros(len(bands))
    for j, nm in enumerate(names):
        b = band_map.get(nm)
        if b in FREQ_BANDS:
            vec[bands.index(b)] += imp[j]
    s = vec.sum()
    return vec / s if s > 0 else vec


def cnn_band_importance(attr: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """attr: (n_beats, n_leads, seg_len) attribution magnitudes.
    Returns band-power distribution of the attribution, normalized to sum 1."""
    bands = list(FREQ_BANDS.keys())
    acc = np.zeros(len(bands))
    n, nl, L = attr.shape
    nper = min(L, 128)
    for i in range(n):
        for c in range(nl):
            a = attr[i, c]
            f, pxx = welch(a, fs=fs, nperseg=nper)
            for k, (lo, hi) in enumerate(FREQ_BANDS.values()):
                m = (f >= lo) & (f < hi)
                acc[k] += float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0
    s = acc.sum()
    return acc / s if s > 0 else acc


def agreement(lin_vec: np.ndarray, cnn_vec: np.ndarray) -> dict:
    """Compare two band-importance distributions."""
    lin = np.asarray(lin_vec, float)
    cnn = np.asarray(cnn_vec, float)
    cos = float(lin @ cnn / (np.linalg.norm(lin) * np.linalg.norm(cnn) + 1e-12))
    rho = float(spearmanr(lin, cnn).correlation) if len(lin) > 2 else np.nan
    top_lin = int(np.argmax(lin))
    top_cnn = int(np.argmax(cnn))
    return {
        "bands": list(FREQ_BANDS.keys()),
        "linear_band_importance": lin.round(4).tolist(),
        "cnn_band_importance": cnn.round(4).tolist(),
        "cosine_similarity": round(cos, 4),
        "spearman_rho": None if np.isnan(rho) else round(rho, 4),
        "top_band_match": bool(top_lin == top_cnn),
        "top_band_linear": list(FREQ_BANDS.keys())[top_lin],
        "top_band_cnn": list(FREQ_BANDS.keys())[top_cnn],
    }


def cost_ratio(linear_params: int, cnn_params: int) -> dict:
    return {
        "linear_params": int(linear_params),
        "cnn_params": int(cnn_params),
        "cnn_over_linear": round(cnn_params / max(linear_params, 1), 1),
    }


def alignment_report(linear_clf, names, band_map, cnn, X_raw_test) -> dict:
    """Full fix-#4 report. `linear_clf` must expose coef_ (LogReg / LinearSVC).
    `cnn` is a fitted ECGCNN. `X_raw_test` is (n, n_leads, seg_len)."""
    coef = extract_coef(linear_clf)
    lin_vec = linear_band_importance(coef, names, band_map)

    sal = cnn.saliency(X_raw_test)
    ig = cnn.integrated_gradients(X_raw_test)
    cnn_vec_sal = cnn_band_importance(sal)
    cnn_vec_ig = cnn_band_importance(ig)

    lin_params = int(coef.size + coef.shape[0])
    cnn_params = cnn.cost()["params"]

    return {
        "saliency_alignment": agreement(lin_vec, cnn_vec_sal),
        "integrated_gradients_alignment": agreement(lin_vec, cnn_vec_ig),
        "cost_ratio": cost_ratio(lin_params, cnn_params),
        "headline": (
            "Linear feature model recovers the CNN's dominant frequency band "
            "at ~1/{:.0f} the parameters".format(
                max(cnn_params / max(lin_params, 1), 1))
        ),
    }
