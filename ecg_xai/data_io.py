"""
Data I/O for MIT-BIH and INCART via the `wfdb` package, plus the
inter-patient / patient-grouped split machinery (fix #1).

Key correctness points
-----------------------
* R-peak locations and beat labels come from the *reference annotations*
  (`.atr`), not from a re-detector, for the classification benchmark. This is
  standard AAMI practice and removes a detector-induced confound.
* Splits are always patient- (record-) disjoint. There is NO beat-level
  shuffling anywhere. `group` ids (record names) are propagated so that
  GroupKFold can be used for any cross-validation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .config import (
    TARGET_FS, SYMBOL_TO_AAMI, AAMI_CLASSES,
    MITDB_DS1, MITDB_DS2, RANDOM_STATE,
)


@dataclass
class BeatSet:
    """Container for extracted beats and their provenance.

    Attributes
    ----------
    segments : (n_beats, n_leads, seg_len) float32
    labels   : (n_beats,) str  -- AAMI class
    groups   : (n_beats,) str  -- record id (used as the patient group)
    rr_pre   : (n_beats,) float -- preceding RR interval in seconds
    rr_post  : (n_beats,) float -- following RR interval in seconds
    fs       : int
    """
    segments: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    rr_pre: np.ndarray
    rr_post: np.ndarray
    fs: int = TARGET_FS

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, mask: np.ndarray) -> "BeatSet":
        return BeatSet(
            segments=self.segments[mask],
            labels=self.labels[mask],
            groups=self.groups[mask],
            rr_pre=self.rr_pre[mask],
            rr_post=self.rr_post[mask],
            fs=self.fs,
        )


def _resample_to(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Polyphase resample along axis 0 (time). sig: (n_samples, n_leads)."""
    if fs_in == fs_out:
        return sig
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up, down = fs_out // g, fs_in // g
    return resample_poly(sig, up, down, axis=0)


def _select_leads(sig: np.ndarray, sig_names: list[str], wanted: list[str]) -> np.ndarray:
    """Return columns for `wanted` lead names; fall back to first leads if absent."""
    cols = []
    lname = [s.upper() for s in sig_names]
    for w in wanted:
        w = w.upper()
        if w in lname:
            cols.append(lname.index(w))
    if len(cols) < len(wanted):
        # pad with leading channels not already chosen
        for i in range(sig.shape[1]):
            if i not in cols:
                cols.append(i)
            if len(cols) == len(wanted):
                break
    return sig[:, cols[:len(wanted)]]


def load_record(
    record_path: str,
    leads: tuple[str, str] = ("MLII", "V1"),
    pre_r_sec: float = 0.25,
    post_r_sec: float = 0.45,
) -> BeatSet:
    """Load one WFDB record, resample to TARGET_FS, segment annotated beats.

    Parameters
    ----------
    record_path : path WITHOUT extension, e.g. '/data/mitdb/100'
    leads       : preferred lead names (MIT-BIH: MLII, V1 ; INCART: II, V1)
    """
    import wfdb
    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    sig = np.asarray(rec.p_signal, dtype=np.float64)        # (n, n_leads)
    fs_in = int(rec.fs)
    sig = _select_leads(sig, list(rec.sig_name), list(leads))

    # Gain/amplitude normalisation per lead (handles INCART ADC-gain variance).
    # z-score per record per lead is the simplest fully record-local choice and
    # cannot leak across records.
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True) + 1e-8
    sig = (sig - mu) / sd

    # Resample the continuous signal and map annotation sample indices too.
    if fs_in != TARGET_FS:
        sig = _resample_to(sig, fs_in, TARGET_FS)
        scale = TARGET_FS / fs_in
        r_locs = np.round(np.asarray(ann.sample) * scale).astype(int)
    else:
        r_locs = np.asarray(ann.sample).astype(int)

    symbols = np.asarray(ann.symbol)

    # Keep only beats with a known AAMI mapping.
    keep = np.array([s in SYMBOL_TO_AAMI for s in symbols])
    r_locs = r_locs[keep]
    symbols = symbols[keep]
    labels = np.array([SYMBOL_TO_AAMI[s] for s in symbols])

    pre = int(round(pre_r_sec * TARGET_FS))
    post = int(round(post_r_sec * TARGET_FS))
    seg_len = pre + post
    n_samp, n_leads = sig.shape

    segs, labs, rr_pre, rr_post = [], [], [], []
    rec_name = os.path.basename(record_path)
    for i, r in enumerate(r_locs):
        a, b = r - pre, r + post
        if a < 0 or b > n_samp:
            continue
        seg = sig[a:b, :].T            # (n_leads, seg_len)
        if seg.shape[1] != seg_len:
            continue
        segs.append(seg.astype(np.float32))
        labs.append(labels[i])
        rr_pre.append((r_locs[i] - r_locs[i - 1]) / TARGET_FS if i > 0 else np.nan)
        rr_post.append((r_locs[i + 1] - r_locs[i]) / TARGET_FS if i < len(r_locs) - 1 else np.nan)

    if not segs:
        raise ValueError(f"No valid beats extracted from {record_path}")

    segments = np.stack(segs, axis=0)
    groups = np.array([rec_name] * len(labs))
    return BeatSet(
        segments=segments,
        labels=np.array(labs),
        groups=groups,
        rr_pre=np.array(rr_pre, dtype=np.float32),
        rr_post=np.array(rr_post, dtype=np.float32),
    )


def load_dataset(
    data_dir: str,
    records: Iterable[int | str],
    leads: tuple[str, str] = ("MLII", "V1"),
) -> BeatSet:
    """Concatenate several records into one BeatSet."""
    sets = []
    for r in records:
        path = os.path.join(data_dir, str(r))
        sets.append(load_record(path, leads=leads))
    return BeatSet(
        segments=np.concatenate([s.segments for s in sets], axis=0),
        labels=np.concatenate([s.labels for s in sets], axis=0),
        groups=np.concatenate([s.groups for s in sets], axis=0),
        rr_pre=np.concatenate([s.rr_pre for s in sets], axis=0),
        rr_post=np.concatenate([s.rr_post for s in sets], axis=0),
    )


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def mitdb_interpatient(data_dir: str) -> tuple[BeatSet, BeatSet]:
    """de Chazal inter-patient split: returns (train=DS1, test=DS2)."""
    train = load_dataset(data_dir, MITDB_DS1, leads=("MLII", "V1"))
    test = load_dataset(data_dir, MITDB_DS2, leads=("MLII", "V1"))
    return train, test


def patient_grouped_holdout(
    beats: BeatSet, test_frac: float = 0.3, seed: int = RANDOM_STATE
) -> tuple[BeatSet, BeatSet]:
    """Patient- (record-) disjoint hold-out for datasets without a standard
    split (e.g. INCART). Records are assigned wholesale to train or test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(beats.groups)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_recs = set(uniq[:n_test].tolist())
    test_mask = np.array([g in test_recs for g in beats.groups])
    return beats.subset(~test_mask), beats.subset(test_mask)


# --------------------------------------------------------------------------- #
# INCART patient-level grouping (reviewer comment 5).
#
# INCART has 75 records but only 32 patients; each record's .hea file carries a
# patient number (1-32), and all records with the same number come from the
# same Holter recording. Grouping by RECORD id therefore does NOT guarantee a
# patient-disjoint split. These helpers read the patient number from the header
# and group on it, then emit the exact allocation for reproducibility. Nothing
# here invents a mapping; if the patient number cannot be read, it refuses to
# proceed rather than silently falling back to record-level grouping.
# --------------------------------------------------------------------------- #
import re as _re


def read_incart_patient_number(record_path: str) -> str:
    """Parse the patient number from an INCART .hea file's comment lines.

    The INCART headers include a comment carrying the patient number (1-32).
    Header comment formats vary slightly across PhysioNet vintages, so this
    tries wfdb's parsed comments first, then a direct regex on the .hea text.
    Raises ValueError if no patient number can be found -- callers must handle
    that rather than assume record==patient.
    """
    # 1) via wfdb-parsed comments (optional; a .hea is plain text, so the
    #    raw-text fallback below works even if wfdb is unavailable).
    comments = []
    try:
        import wfdb
        rec = wfdb.rdheader(record_path)
        comments = list(getattr(rec, "comments", []) or [])
    except Exception:
        comments = []
    patterns = [
        _re.compile(r"patient\s*(?:number|no\.?|#)?\s*[:=]?\s*(\d{1,2})", _re.I),
        _re.compile(r"\bpatient\s+(\d{1,2})\b", _re.I),
    ]
    for c in comments:
        for pat in patterns:
            m = pat.search(c)
            if m:
                return f"P{int(m.group(1)):02d}"
    # 2) fall back to reading the raw .hea text
    hea = record_path + ".hea"
    if os.path.exists(hea):
        with open(hea, "r", errors="ignore") as f:
            text = f.read()
        for pat in patterns:
            m = pat.search(text)
            if m:
                return f"P{int(m.group(1)):02d}"
    raise ValueError(
        f"Could not read a patient number from {record_path}.hea. "
        "INCART patient-level grouping requires it; supply an explicit "
        "record->patient map instead of guessing."
    )


def attach_incart_patients(
    beats: "BeatSet", data_dir: str,
    record_to_patient: dict | None = None,
) -> "BeatSet":
    """Return a copy of `beats` whose `groups` are PATIENT ids instead of
    record ids. Uses `record_to_patient` if given, else parses each record's
    .hea via read_incart_patient_number. Every record must resolve to a patient
    or this raises, so a partial/guessed mapping can never slip through."""
    rec_ids = np.unique(beats.groups)
    mapping = {}
    for r in rec_ids:
        if record_to_patient and r in record_to_patient:
            mapping[r] = record_to_patient[r]
        else:
            mapping[r] = read_incart_patient_number(os.path.join(data_dir, str(r)))
    new_groups = np.array([mapping[g] for g in beats.groups])
    b = beats.subset(np.ones(len(beats), dtype=bool))
    b.groups = new_groups
    return b


def allocation_report(train: "BeatSet", test: "BeatSet") -> dict:
    """Reproducibility report the reviewer asks for: patient and record counts
    per partition, the patient/record ids in each, per-class beat counts, and a
    hard assertion that NO patient appears in both partitions."""
    tr_p = set(np.unique(train.groups).tolist())
    te_p = set(np.unique(test.groups).tolist())
    overlap = sorted(tr_p & te_p)
    def class_counts(bs):
        c, n = np.unique(bs.labels, return_counts=True)
        return {str(k): int(v) for k, v in zip(c, n)}
    return {
        "n_patients_train": len(tr_p), "n_patients_test": len(te_p),
        "patients_train": sorted(tr_p), "patients_test": sorted(te_p),
        "patient_overlap": overlap,
        "patient_disjoint": len(overlap) == 0,
        "n_beats_train": len(train), "n_beats_test": len(test),
        "class_counts_train": class_counts(train),
        "class_counts_test": class_counts(test),
    }


def incart_patient_grouped_holdout(
    data_dir: str, records, leads: tuple[str, str] = ("II", "V1"),
    test_frac: float = 0.3, seed: int = RANDOM_STATE,
    record_to_patient: dict | None = None,
) -> tuple["BeatSet", "BeatSet", dict]:
    """Load INCART, regroup beats by PATIENT (not record), then hold out whole
    PATIENTS. Returns (train, test, allocation_report). The report makes the
    patient-disjointness auditable and provides the exact ids for the paper."""
    beats = load_dataset(data_dir, records, leads=leads)
    beats = attach_incart_patients(beats, data_dir, record_to_patient)
    train, test = patient_grouped_holdout(beats, test_frac=test_frac, seed=seed)
    report = allocation_report(train, test)
    if not report["patient_disjoint"]:
        raise RuntimeError(
            f"Patient overlap across split: {report['patient_overlap']}. "
            "This must be empty; check the record->patient mapping."
        )
    return train, test, report
