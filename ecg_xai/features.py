# """
# Feature extraction, organised by *family* so families can be switched off for
# the ablation study (fix #3). Returns a dense matrix plus, crucially, a
# column -> family map and a column -> frequency-band map (the latter feeds the
# XAI alignment common space, fix #4).

# Streaming honesty (fix #2)
# --------------------------
# 'hrv' and 'graph' features are computed only from a *trailing causal window*
# of previous beats within the SAME record. They never use future beats and
# never cross record (patient) boundaries. This both prevents leakage and lets
# the paper state the true per-decision latency (you must buffer
# GRAPH_WINDOW_BEATS beats before the graph features are defined).
# """
# from __future__ import annotations

# import numpy as np
# from scipy.stats import skew, kurtosis
# from scipy.signal import welch
# import pywt

# from .config import (
#     TARGET_FS, FREQ_BANDS, FEATURE_FAMILIES,
#     GRAPH_WINDOW_BEATS, GRAPH_TAU,
# )
# from .data_io import BeatSet
# from .preprocessing import bandpass

# WAVELET = "db4"
# WAVELET_LEVEL = 4


# # --------------------------------------------------------------------------- #
# # Per-beat families (no temporal context needed)
# # --------------------------------------------------------------------------- #
# def _time_features(seg: np.ndarray) -> dict[str, float]:
#     """seg: (n_leads, L)."""
#     out = {}
#     for c in range(seg.shape[0]):
#         x = seg[c]
#         out[f"td_mean_ch{c}"] = float(np.mean(x))
#         out[f"td_std_ch{c}"] = float(np.std(x))
#         out[f"td_skew_ch{c}"] = float(skew(x))
#         out[f"td_kurt_ch{c}"] = float(kurtosis(x))
#         out[f"td_rms_ch{c}"] = float(np.sqrt(np.mean(x ** 2)))
#         zc = np.sum(np.abs(np.diff(np.sign(x)))) / (2.0 * len(x))
#         out[f"td_zcr_ch{c}"] = float(zc)
#     return out


# def _freq_features(seg: np.ndarray, fs: int = TARGET_FS) -> tuple[dict, dict]:
#     """Return (features, band_map). band_map tags band-power columns with their
#     Hz band so the XAI alignment can aggregate to the same space."""
#     out, band_map = {}, {}
#     nper = min(len(seg[0]), 128)
#     for c in range(seg.shape[0]):
#         f, pxx = welch(seg[c], fs=fs, nperseg=nper)
#         total = np.trapezoid(pxx, f) + 1e-12
#         for name, (lo, hi) in FREQ_BANDS.items():
#             m = (f >= lo) & (f < hi)
#             bp = float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0
#             key = f"fd_{name}_ch{c}"
#             out[key] = bp / total
#             band_map[key] = name
#         # spectral entropy
#         p = pxx / (pxx.sum() + 1e-12)
#         out[f"fd_spec_entropy_ch{c}"] = float(-np.sum(p * np.log(p + 1e-12)))
#     return out, band_map


# def _morph_features(seg: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
#     """Cheap QRS-centric morphology proxies on the band-passed beat."""
#     out = {}
#     for c in range(seg.shape[0]):
#         x = bandpass(seg[c], fs=fs)
#         r_idx = int(np.argmax(np.abs(x)))
#         out[f"mp_r_amp_ch{c}"] = float(x[r_idx])
#         # QRS width proxy: samples around R above 30% of peak |amp|
#         thr = 0.3 * np.abs(x[r_idx])
#         above = np.where(np.abs(x) >= thr)[0]
#         qrs = (above.max() - above.min()) / fs * 1000.0 if above.size else 0.0
#         out[f"mp_qrs_dur_ms_ch{c}"] = float(qrs)
#         # energy ratio: window around R vs whole beat
#         w = int(0.06 * fs)
#         a, b = max(0, r_idx - w), min(len(x), r_idx + w)
#         e_qrs = float(np.sum(x[a:b] ** 2))
#         e_tot = float(np.sum(x ** 2)) + 1e-12
#         out[f"mp_energy_ratio_ch{c}"] = e_qrs / e_tot
#     return out


# def _wavelet_features(seg: np.ndarray) -> tuple[dict, dict]:
#     """Daubechies sub-band log-energies. Sub-bands map approximately to Hz
#     bands (cA/detail levels) -> used for XAI alignment too."""
#     out, band_map = {}, {}
#     level_to_band = {
#         "A4": "low_0_5Hz", "D4": "mid_5_15Hz", "D3": "high_15_40Hz",
#         "D2": "high_15_40Hz", "D1": "high_15_40Hz",
#     }
#     for c in range(seg.shape[0]):
#         coeffs = pywt.wavedec(seg[c], WAVELET, level=WAVELET_LEVEL)
#         names = [f"A{WAVELET_LEVEL}"] + [f"D{WAVELET_LEVEL - i}" for i in range(WAVELET_LEVEL)]
#         for nm, cf in zip(names, coeffs):
#             key = f"wv_{nm}_energy_ch{c}"
#             out[key] = float(np.log1p(np.sum(np.asarray(cf) ** 2)))
#             if nm in level_to_band:
#                 band_map[key] = level_to_band[nm]
#     return out, band_map


# # --------------------------------------------------------------------------- #
# # Temporal-context families (causal trailing window, within-record only)
# # --------------------------------------------------------------------------- #
# def _hrv_window(rr: np.ndarray) -> dict[str, float]:
#     """HRV from a trailing RR window (seconds). rr may contain NaN."""
#     rr = rr[~np.isnan(rr)]
#     if rr.size < 3:
#         return {"hrv_sdnn": 0.0, "hrv_pnn50": 0.0, "hrv_rmssd": 0.0,
#                 "hrv_lfhf": 0.0, "hrv_mean_rr": float(np.nan_to_num(rr.mean()) if rr.size else 0.0)}
#     d = np.diff(rr)
#     sdnn = float(np.std(rr))
#     pnn50 = float(np.mean(np.abs(d) > 0.05))
#     rmssd = float(np.sqrt(np.mean(d ** 2)))
#     if rr.size >= 8:
#         f, pxx = welch(rr - rr.mean(), fs=4.0, nperseg=min(rr.size, 8))
#         lf = np.trapezoid(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]) + 1e-9
#         hf = np.trapezoid(pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]) + 1e-9
#         lfhf = float(lf / hf)
#     else:
#         lfhf = 0.0
#     return {"hrv_sdnn": sdnn, "hrv_pnn50": pnn50, "hrv_rmssd": rmssd,
#             "hrv_lfhf": lfhf, "hrv_mean_rr": float(np.mean(rr))}


# def _graph_window(base_window: np.ndarray, tau: float = GRAPH_TAU,
#                   cur: int | None = None) -> dict[str, float]:
#     """Graph-theoretic descriptors for the current beat given a window of
#     compact per-beat descriptors `base_window` of shape (w, d). Edges:
#         w_ij = cos(f_i, f_j) * exp(-|i-j|/tau).
#     `cur` is the index of the current node within the window.
#     """
#     import networkx as nx
#     w = base_window.shape[0]
#     if cur is None:
#         cur = w - 1
#     if w < 3:
#         return {"gr_pagerank": 0.0, "gr_clustering": 0.0, "gr_wdeg": 0.0}

#     F = np.asarray(base_window, dtype=np.float64)
#     F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

#     row_norms = np.linalg.norm(F, axis=1, keepdims=True)
#     safe_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
#     Fn = F / safe_norms
#     Fn = np.nan_to_num(Fn, nan=0.0, posinf=0.0, neginf=0.0)

#     cos = Fn @ Fn.T
#     cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)

#     idx = np.arange(w)
#     decay = np.exp(-np.abs(idx[:, None] - idx[None, :]) / tau)
#     A = np.clip(cos, 0.0, None) * decay
#     np.fill_diagonal(A, 0.0)

#     G = nx.from_numpy_array(A)
#     try:
#         pr = nx.pagerank(G, alpha=0.85, weight="weight").get(cur, 0.0)
#     except Exception:
#         pr = 0.0
#     cl = nx.clustering(G, nodes=cur, weight="weight")
#     wdeg = float(np.sum(A[cur]))
#     return {"gr_pagerank": float(pr), "gr_clustering": float(cl), "gr_wdeg": wdeg}


# def _compact_descriptor(seg: np.ndarray) -> np.ndarray:
#     """Small, fast vector used to build the beat-similarity graph."""
#     parts = []
#     for c in range(seg.shape[0]):
#         x = np.nan_to_num(seg[c], nan=0.0, posinf=0.0, neginf=0.0)
#         s = float(np.std(x))
#         rms = float(np.sqrt(np.mean(x ** 2)))
#         sk = float(skew(x)) if np.isfinite(skew(x)) else 0.0
#         ku = float(kurtosis(x)) if np.isfinite(kurtosis(x)) else 0.0
#         peak = float(np.max(np.abs(x)))
#         parts += [s, rms, sk, ku, peak]
#     return np.asarray(parts, dtype=np.float64)


# # --------------------------------------------------------------------------- #
# # Assembler
# # --------------------------------------------------------------------------- #
# def build_feature_matrix(
#     beats: BeatSet,
#     window_beats: int = GRAPH_WINDOW_BEATS,
#     causal: bool = True,
# ) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, str]]:
#     """
#     Build the engineered feature matrix.

#     causal=True  : HRV and graph features use only the trailing window (recommended).
#     causal=False : Uses full record (only for causal-vs-acausal leakage experiments).
#     """
#     n = len(beats)
#     rows, family, band_map = [], {}, {}
#     names_ref: list[str] | None = None

#     descr = np.stack([_compact_descriptor(beats.segments[i]) for i in range(n)], axis=0)

#     for rec in np.unique(beats.groups):
#         ridx = np.where(beats.groups == rec)[0]
#         for pos, i in enumerate(ridx):
#             seg = beats.segments[i]
#             feat = {}
#             ft = _time_features(seg);                         feat.update(ft)
#             ff, bf = _freq_features(seg);                     feat.update(ff)
#             fm = _morph_features(seg);                        feat.update(fm)
#             fw, bw = _wavelet_features(seg);                  feat.update(fw)

#             if causal:
#                 lo = max(0, pos - window_beats + 1)
#                 win_idx = ridx[lo:pos + 1]
#                 cur = len(win_idx) - 1
#             else:
#                 win_idx = ridx
#                 cur = pos

#             rr_win = beats.rr_pre[win_idx]
#             feat.update(_hrv_window(rr_win))
#             feat.update(_graph_window(descr[win_idx], cur=cur))

#             if names_ref is None:
#                 names_ref = list(feat.keys())
#                 for k in names_ref:
#                     if k.startswith("td_"):   family[k] = "time"
#                     elif k.startswith("fd_"): family[k] = "freq"
#                     elif k.startswith("mp_"): family[k] = "morph"
#                     elif k.startswith("wv_"): family[k] = "wavelet"
#                     elif k.startswith("hrv_"):family[k] = "hrv"
#                     elif k.startswith("gr_"): family[k] = "graph"
#                 band_map.update(bf)
#                 band_map.update(bw)
#             rows.append([feat[k] for k in names_ref])

#     X = np.asarray(rows, dtype=np.float32)
#     return X, names_ref, family, band_map


# # --------------------------------------------------------------------------- #
# # Saving / Loading Engineered Features
# # --------------------------------------------------------------------------- #

# import json
# from pathlib import Path
# from typing import Dict, Tuple


# def save_feature_set(
#     X: np.ndarray,
#     y: np.ndarray,
#     groups: np.ndarray,
#     names: list[str],
#     family: Dict[str, str],
#     band_map: Dict[str, str],
#     filepath: str | Path,
# ) -> None:
#     """Save the complete engineered feature set + metadata."""
#     filepath = Path(filepath)
#     filepath.parent.mkdir(parents=True, exist_ok=True)

#     save_dict = {
#         "X": X.astype(np.float32),
#         "y": np.asarray(y),
#         "groups": np.asarray(groups),
#         "names": np.array(names, dtype=object),
#         "family": json.dumps(family),
#         "band_map": json.dumps(band_map),
#     }
#     np.savez_compressed(filepath, **save_dict)
#     print(f"[Saved] {filepath} | shape={X.shape} | {filepath.stat().st_size/1024:.1f} KB")


# def load_feature_set(filepath: str | Path) -> Tuple:
#     """Load saved engineered features. Returns: X, y, groups, names, family, band_map"""
#     data = np.load(filepath, allow_pickle=True)
#     return (
#         data["X"],
#         data["y"],
#         data["groups"],
#         data["names"].tolist(),
#         json.loads(str(data["family"])),
#         json.loads(str(data["band_map"])),
#     )


# def save_dataset(
#     beats: BeatSet,
#     name: str,
#     output_dir: str = "data/processed",
#     window_beats: int = GRAPH_WINDOW_BEATS,
#     causal: bool = True,
# ) -> Tuple[np.ndarray, list[str], Dict[str, str], Dict[str, str]]:
#     """High-level helper: Featurize + save a BeatSet automatically."""
#     X, names, family, band_map = build_feature_matrix(
#         beats, window_beats=window_beats, causal=causal
#     )
#     filepath = Path(output_dir) / f"{name}.npz"
#     save_feature_set(X, beats.labels, beats.groups, names, family, band_map, filepath)
#     return X, names, family, band_map


"""
Feature extraction, organised by *family* so families can be switched off for
the ablation study (fix #3). Returns a dense matrix plus, crucially, a
column -> family map and a column -> frequency-band map (the latter feeds the
XAI alignment common space, fix #4).

Streaming honesty (fix #2)
--------------------------
'hrv' and 'graph' features are computed only from a *trailing causal window*
of previous beats within the SAME record. They never use future beats and
never cross record (patient) boundaries. This both prevents leakage and lets
the paper state the true per-decision latency (you must buffer
GRAPH_WINDOW_BEATS beats before the graph features are defined).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt

from .config import (
    TARGET_FS, FREQ_BANDS, FEATURE_FAMILIES,
    GRAPH_WINDOW_BEATS, GRAPH_TAU,
)
from .data_io import BeatSet
from .preprocessing import bandpass

WAVELET = "db4"
WAVELET_LEVEL = 4


# --------------------------------------------------------------------------- #
# Per-beat families (no temporal context needed)
# --------------------------------------------------------------------------- #
def _time_features(seg: np.ndarray) -> dict[str, float]:
    """seg: (n_leads, L)."""
    out = {}
    for c in range(seg.shape[0]):
        x = seg[c]
        out[f"td_mean_ch{c}"] = float(np.mean(x))
        out[f"td_std_ch{c}"] = float(np.std(x))
        out[f"td_skew_ch{c}"] = float(skew(x))
        out[f"td_kurt_ch{c}"] = float(kurtosis(x))
        out[f"td_rms_ch{c}"] = float(np.sqrt(np.mean(x ** 2)))
        zc = np.sum(np.abs(np.diff(np.sign(x)))) / (2.0 * len(x))
        out[f"td_zcr_ch{c}"] = float(zc)
    return out


def _freq_features(seg: np.ndarray, fs: int = TARGET_FS) -> tuple[dict, dict]:
    """Return (features, band_map). band_map tags band-power columns with their
    Hz band so the XAI alignment can aggregate to the same space."""
    out, band_map = {}, {}
    nper = min(len(seg[0]), 128)
    for c in range(seg.shape[0]):
        f, pxx = welch(seg[c], fs=fs, nperseg=nper)
        total = np.trapezoid(pxx, f) + 1e-12
        for name, (lo, hi) in FREQ_BANDS.items():
            m = (f >= lo) & (f < hi)
            bp = float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0
            key = f"fd_{name}_ch{c}"
            out[key] = bp / total
            band_map[key] = name
        # spectral entropy
        p = pxx / (pxx.sum() + 1e-12)
        out[f"fd_spec_entropy_ch{c}"] = float(-np.sum(p * np.log(p + 1e-12)))
    return out, band_map


def _morph_features(seg: np.ndarray, fs: int = TARGET_FS) -> dict[str, float]:
    """Cheap QRS-centric morphology proxies on the band-passed beat."""
    out = {}
    for c in range(seg.shape[0]):
        x = bandpass(seg[c], fs=fs)
        r_idx = int(np.argmax(np.abs(x)))
        out[f"mp_r_amp_ch{c}"] = float(x[r_idx])
        # QRS width proxy: samples around R above 30% of peak |amp|
        thr = 0.3 * np.abs(x[r_idx])
        above = np.where(np.abs(x) >= thr)[0]
        qrs = (above.max() - above.min()) / fs * 1000.0 if above.size else 0.0
        out[f"mp_qrs_dur_ms_ch{c}"] = float(qrs)
        # energy ratio: window around R vs whole beat
        w = int(0.06 * fs)
        a, b = max(0, r_idx - w), min(len(x), r_idx + w)
        e_qrs = float(np.sum(x[a:b] ** 2))
        e_tot = float(np.sum(x ** 2)) + 1e-12
        out[f"mp_energy_ratio_ch{c}"] = e_qrs / e_tot
    return out


def _wavelet_features(seg: np.ndarray) -> tuple[dict, dict]:
    """Daubechies sub-band log-energies. Sub-bands map approximately to Hz
    bands (cA/detail levels) -> used for XAI alignment too."""
    out, band_map = {}, {}
    # rough Hz center per detail level at fs=360: D1~90-180, D2~45-90,
    # D3~22-45, D4~11-22, A4~0-11. Map to our 3 bands.
    level_to_band = {
        "A4": "low_0_5Hz", "D4": "mid_5_15Hz", "D3": "high_15_40Hz",
        "D2": "high_15_40Hz", "D1": "high_15_40Hz",
    }
    for c in range(seg.shape[0]):
        coeffs = pywt.wavedec(seg[c], WAVELET, level=WAVELET_LEVEL)
        names = [f"A{WAVELET_LEVEL}"] + [f"D{WAVELET_LEVEL - i}" for i in range(WAVELET_LEVEL)]
        for nm, cf in zip(names, coeffs):
            key = f"wv_{nm}_energy_ch{c}"
            out[key] = float(np.log1p(np.sum(np.asarray(cf) ** 2)))
            if nm in level_to_band:
                band_map[key] = level_to_band[nm]
    return out, band_map


# --------------------------------------------------------------------------- #
# Temporal-context families (causal trailing window, within-record only)
# --------------------------------------------------------------------------- #
def _hrv_window(rr: np.ndarray) -> dict[str, float]:
    """HRV from a trailing RR window (seconds). rr may contain NaN."""
    rr = rr[~np.isnan(rr)]
    if rr.size < 3:
        return {"hrv_sdnn": 0.0, "hrv_pnn50": 0.0, "hrv_rmssd": 0.0,
                "hrv_lfhf": 0.0, "hrv_mean_rr": float(np.nan_to_num(rr.mean()) if rr.size else 0.0)}
    d = np.diff(rr)
    sdnn = float(np.std(rr))
    pnn50 = float(np.mean(np.abs(d) > 0.05))
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    # crude LF/HF from periodogram of the RR tachogram
    if rr.size >= 8:
        f, pxx = welch(rr - rr.mean(), fs=4.0, nperseg=min(rr.size, 8))
        lf = np.trapezoid(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]) + 1e-9
        hf = np.trapezoid(pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]) + 1e-9
        lfhf = float(lf / hf)
    else:
        lfhf = 0.0
    return {"hrv_sdnn": sdnn, "hrv_pnn50": pnn50, "hrv_rmssd": rmssd,
            "hrv_lfhf": lfhf, "hrv_mean_rr": float(np.mean(rr))}


def _graph_window(base_window: np.ndarray, tau: float = GRAPH_TAU,
                  cur: int | None = None) -> dict[str, float]:
    """Graph-theoretic descriptors for the current beat given a window of
    compact per-beat descriptors `base_window` of shape (w, d). Edges:
        w_ij = cos(f_i, f_j) * exp(-|i-j|/tau).
    `cur` is the index of the current node within the window; it defaults to the
    last position (the causal case). For an acausal full-record graph, the
    caller passes the beat's true position within the record.
    """
    import networkx as nx
    w = base_window.shape[0]
    if cur is None:
        cur = w - 1
    if w < 3:
        return {"gr_pagerank": 0.0, "gr_clustering": 0.0, "gr_wdeg": 0.0}
    F = base_window
    norm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    Fn = F / norm
    cos = Fn @ Fn.T
    idx = np.arange(w)
    decay = np.exp(-np.abs(idx[:, None] - idx[None, :]) / tau)
    A = np.clip(cos, 0, None) * decay
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    try:
        pr = nx.pagerank(G, alpha=0.85, weight="weight").get(cur, 0.0)
    except Exception:
        pr = 0.0
    cl = nx.clustering(G, nodes=cur, weight="weight")
    wdeg = float(A[cur].sum())
    return {"gr_pagerank": float(pr), "gr_clustering": float(cl), "gr_wdeg": wdeg}


def _compact_descriptor(seg: np.ndarray) -> np.ndarray:
    """Small, fast vector used to build the beat-similarity graph."""
    parts = []
    for c in range(seg.shape[0]):
        x = seg[c]
        parts += [np.std(x), np.sqrt(np.mean(x ** 2)),
                  float(skew(x)), float(kurtosis(x)), float(np.max(np.abs(x)))]
    return np.asarray(parts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Assembler
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    beats: BeatSet,
    window_beats: int = GRAPH_WINDOW_BEATS,
    causal: bool = True,
) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, str]]:
    """
    Build the engineered feature matrix.

    causal=True  : HRV and graph features use only the trailing window of the
                   `window_beats` preceding beats (the deployable, leakage-free
                   construction; Proposition 1).
    causal=False : HRV and graph features use the entire record (acausal). This
                   variant exists ONLY for the causal-vs-acausal leakage
                   experiment and must not be used for reported results.

    Returns
    -------
    X         : (n_beats, n_features) float32
    names     : feature column names (len n_features)
    family    : {feature_name -> family}
    band_map  : {feature_name -> freq band name} for band-mappable columns only
    """
    n = len(beats)
    rows, family, band_map = [], {}, {}
    names_ref: list[str] | None = None

    # Precompute compact descriptors per beat for the graph.
    descr = np.stack([_compact_descriptor(beats.segments[i]) for i in range(n)], axis=0)

    # Iterate within each record so temporal context is record-local.
    for rec in np.unique(beats.groups):
        ridx = np.where(beats.groups == rec)[0]
        for pos, i in enumerate(ridx):
            seg = beats.segments[i]
            feat = {}
            ft = _time_features(seg);                         feat.update(ft)
            ff, bf = _freq_features(seg);                     feat.update(ff)
            fm = _morph_features(seg);                        feat.update(fm)
            fw, bw = _wavelet_features(seg);                  feat.update(fw)

            if causal:
                # trailing window ending at the current beat (current = last)
                lo = max(0, pos - window_beats + 1)
                win_idx = ridx[lo:pos + 1]
                cur = len(win_idx) - 1
            else:
                # bounded SYMMETRIC window: W past + W future beats. This is
                # acausal (it lets a beat see the future) but keeps the same
                # O(W^2) per-beat cost as the causal path, so it terminates.
                # Used ONLY for the causal-vs-acausal leakage experiment.
                lo = max(0, pos - window_beats + 1)
                hi = min(len(ridx), pos + window_beats)
                win_idx = ridx[lo:hi]
                cur = pos - lo
            rr_win = beats.rr_pre[win_idx]
            feat.update(_hrv_window(rr_win))
            feat.update(_graph_window(descr[win_idx], cur=cur))

            if names_ref is None:
                names_ref = list(feat.keys())
                # build family + band maps once
                for k in names_ref:
                    if k.startswith("td_"):
                        family[k] = "time"
                    elif k.startswith("fd_"):
                        family[k] = "freq"
                    elif k.startswith("mp_"):
                        family[k] = "morph"
                    elif k.startswith("wv_"):
                        family[k] = "wavelet"
                    elif k.startswith("hrv_"):
                        family[k] = "hrv"
                    elif k.startswith("gr_"):
                        family[k] = "graph"
                band_map.update(bf)
                band_map.update(bw)
            rows.append([feat[k] for k in names_ref])

    X = np.asarray(rows, dtype=np.float32)
    return X, names_ref, family, band_map