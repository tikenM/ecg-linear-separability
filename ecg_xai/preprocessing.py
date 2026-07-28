"""Zero-phase Butterworth bandpass, realised as cascaded second-order
sections (SOS) for numerical stability, applied forward-backward (filtfilt)
to preserve QRS timing."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import BANDPASS, BP_ORDER, TARGET_FS


def bandpass(sig: np.ndarray, fs: int = TARGET_FS,
             band: tuple[float, float] = BANDPASS, order: int = BP_ORDER) -> np.ndarray:
    """Zero-phase bandpass along the LAST axis.

    sig may be (seg_len,), (n_leads, seg_len) or (n_beats, n_leads, seg_len).
    """
    lo, hi = band
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    # filtfilt needs length > 3 * (max section order); guard short beats.
    padlen = min(3 * (sos.shape[0]), sig.shape[-1] - 1)
    return sosfiltfilt(sos, sig, axis=-1, padlen=max(padlen, 0))
