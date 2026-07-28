"""
detection_eval.py

Addresses reviewer comment 6: the classification benchmark uses reference
(annotated) R-peaks; a deployable, continuous-monitoring claim needs (a) the
detector's own performance and (b) how classification degrades when reference
peaks are replaced by detected peaks.

This module supplies the harness. It does NOT include a detector -- you plug in
your ensemble detector as a callable `detector(signal_1d, fs) -> peak_samples`.
Two things it computes:

1. score_detection : R-peak detection performance against the reference
   annotations, using the standard AAMI EC57 matching window (default 150 ms):
   sensitivity (Se), positive predictivity (PPV = +P), and F1, plus raw
   TP/FP/FN counts.

2. transfer_labels + build_detected_beatset : re-segment beats around the
   DETECTED peaks and carry each reference beat's AAMI label onto the matched
   detected peak, so the existing feature/classification pipeline can be run
   end-to-end on detected peaks and compared to the reference-peak numbers.

Matching rule
-------------
A detected peak matches a reference peak if it is the nearest detection within
`tol` samples of that reference peak, one-to-one (each reference and each
detection is used at most once). Greedy nearest-first matching, which is the
standard and is stable for the low false-detection rates expected here.
"""
from __future__ import annotations

import numpy as np


def match_peaks(ref: np.ndarray, det: np.ndarray, tol: int) -> dict:
    """One-to-one nearest matching within `tol` samples.

    Returns dict with:
      matched      : list of (ref_idx, det_idx) index pairs into ref / det
      missed_ref   : ref indices with no match (false negatives)
      extra_det    : det indices with no match (false positives)
    """
    ref = np.asarray(ref); det = np.asarray(det)
    if len(ref) == 0 or len(det) == 0:
        return {"matched": [], "missed_ref": list(range(len(ref))),
                "extra_det": list(range(len(det)))}
    # candidate pairs within tolerance, sorted by absolute distance
    cand = []
    for i, r in enumerate(ref):
        lo = np.searchsorted(det, r - tol, side="left")
        hi = np.searchsorted(det, r + tol, side="right")
        for j in range(lo, hi):
            cand.append((abs(int(det[j]) - int(r)), i, j))
    cand.sort()
    used_ref, used_det, matched = set(), set(), []
    for _, i, j in cand:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i); used_det.add(j); matched.append((i, j))
    missed = [i for i in range(len(ref)) if i not in used_ref]
    extra = [j for j in range(len(det)) if j not in used_det]
    return {"matched": matched, "missed_ref": missed, "extra_det": extra}


def score_detection(ref: np.ndarray, det: np.ndarray, fs: int,
                    tol_ms: float = 150.0) -> dict:
    """Sensitivity / positive predictivity / F1 for R-peak detection, AAMI-style.
    ref, det are R-peak sample indices at sampling rate `fs`."""
    tol = int(round(tol_ms / 1000.0 * fs))
    m = match_peaks(np.sort(ref), np.sort(det), tol)
    tp = len(m["matched"]); fn = len(m["missed_ref"]); fp = len(m["extra_det"])
    se = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * se * ppv / (se + ppv)) if (se and ppv and not np.isnan(se) and not np.isnan(ppv)) else 0.0
    return {
        "tol_ms": tol_ms, "tol_samples": tol,
        "tp": tp, "fp": fp, "fn": fn,
        "n_ref": int(len(ref)), "n_det": int(len(det)),
        "sensitivity": round(se, 4) if not np.isnan(se) else None,
        "positive_predictivity": round(ppv, 4) if not np.isnan(ppv) else None,
        "f1": round(f1, 4),
    }


def transfer_labels(ref: np.ndarray, ref_labels: np.ndarray,
                    det: np.ndarray, tol: int) -> dict:
    """Carry reference AAMI labels onto detected peaks via nearest matching.

    Returns:
      det_labels : array over det; matched entries get the ref label, unmatched
                   (spurious) detections get '' (empty) so callers can drop them.
      matched_pairs, missed_ref, extra_det : as in match_peaks.
    """
    ref = np.asarray(ref); det = np.asarray(det)
    ref_labels = np.asarray(ref_labels)
    order_ref = np.argsort(ref)
    order_det = np.argsort(det)
    ref_s, det_s = ref[order_ref], det[order_det]
    lab_s = ref_labels[order_ref]
    m = match_peaks(ref_s, det_s, tol)
    det_labels_sorted = np.array([""] * len(det_s), dtype=object)
    for i, j in m["matched"]:
        det_labels_sorted[j] = lab_s[i]
    # unsort back to the original det ordering
    det_labels = np.empty(len(det), dtype=object)
    det_labels[order_det] = det_labels_sorted
    return {"det_labels": det_labels,
            "matched": m["matched"], "missed_ref": m["missed_ref"],
            "extra_det": m["extra_det"]}


def evaluate_record_with_detector(record_path: str, detector,
                                  lead_for_detection: str = "MLII",
                                  tol_ms: float = 150.0) -> dict:
    """End-to-end on ONE record: run `detector` on the chosen lead, score the
    detected peaks against the reference .atr annotations, and return the
    detection metrics plus the arrays needed to rebuild a detected-peak
    BeatSet. Requires wfdb and the real record on disk.

    detector : callable(signal_1d: np.ndarray, fs: int) -> np.ndarray of peak
               sample indices (in the record's ORIGINAL sampling rate).

    This is intentionally thin: it does not import the rest of the package, so
    it can be unit-tested and reused independently. To produce a classifiable
    BeatSet on detected peaks, feed `det_peaks` and `det_labels` (from
    transfer_labels) into the same segmentation code path used for reference
    peaks in data_io.load_record.
    """
    import wfdb
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    fs = int(rec.fs)
    names = [s.upper() for s in rec.sig_name]
    ch = names.index(lead_for_detection.upper()) if lead_for_detection.upper() in names else 0
    sig = np.asarray(rec.p_signal, dtype=np.float64)[:, ch]

    det = np.asarray(detector(sig, fs)).astype(int)
    ref = np.asarray(ann.sample).astype(int)
    det_score = score_detection(ref, det, fs, tol_ms=tol_ms)

    tol = det_score["tol_samples"]
    tr = transfer_labels(ref, np.asarray(ann.symbol), det, tol)
    return {
        "record": record_path, "fs": fs,
        "detection": det_score,
        "det_peaks": det, "det_labels": tr["det_labels"],
        "n_spurious_detections": int(np.sum(tr["det_labels"] == "")),
    }
