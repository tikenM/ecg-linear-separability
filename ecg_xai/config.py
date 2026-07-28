# """
# Central configuration: AAMI labels, inter-patient splits, frequency bands,
# and the canonical list of feature families used for ablation.

# The four manuscript fixes this package implements:
#   (1) Inter-patient evaluation + no-leakage pipeline  -> data_io.py + pipeline.py
#   (2) Size/transparency story instead of latency       -> reporting in run_experiments.py
#   (3) Separability metric + feature-family ablation     -> separability.py + ablation.py
#   (4) "Recovers black-box explanations cheaply"         -> cnn_baseline.py + xai_alignment.py
# """
# from __future__ import annotations

# # --------------------------------------------------------------------------- #
# # Signal / processing constants
# # --------------------------------------------------------------------------- #
# TARGET_FS = 360            # Hz; INCART (257 Hz) is resampled to this
# BANDPASS = (0.5, 40.0)     # Hz, Butterworth passband
# BP_ORDER = 4               # filter order (realised as cascaded biquads)
# RANDOM_STATE = 13

# # Beat window relative to the R-peak, in SECONDS. The original paper used an
# # adaptive composite-loss window; for an AAMI *classification benchmark* the
# # accepted practice is a fixed physiological window anchored on the annotated
# # R-peak. This removes a segmentation-induced confound and is what reviewers
# # expect for the headline numbers. (The adaptive window remains available in
# # segmentation.py for the deployment/detection narrative.)
# PRE_R_SEC = 0.25
# POST_R_SEC = 0.45
# SEG_LEN = int(round((PRE_R_SEC + POST_R_SEC) * TARGET_FS))  # samples per beat

# # --------------------------------------------------------------------------- #
# # AAMI class scheme (5 classes). Standard MIT-BIH symbol -> AAMI mapping.
# # --------------------------------------------------------------------------- #
# AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
# SYMBOL_TO_AAMI = {
#     # Normal
#     "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
#     # Supraventricular ectopic
#     "A": "S", "a": "S", "J": "S", "S": "S",
#     # Ventricular ectopic
#     "V": "V", "E": "V",
#     # Fusion
#     "F": "F",
#     # Unknown / paced
#     "/": "Q", "f": "Q", "Q": "Q",
# }

# # --------------------------------------------------------------------------- #
# # Inter-patient split for MIT-BIH (de Chazal et al., 2004). Paced records
# # 102, 104, 107, 217 are excluded by convention. 22 train / 22 test, patient-
# # (record-) disjoint. This is the single most important correctness fix.
# # --------------------------------------------------------------------------- #
# MITDB_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
#              124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
# MITDB_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
#              212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
# MITDB_PACED_EXCLUDED = [102, 104, 107, 217]

# # --------------------------------------------------------------------------- #
# # Frequency bands (Hz) used both for band-power features and for the XAI
# # alignment common space. Chosen to be physiologically meaningful:
# #   low   ~ T-wave / baseline,  mid ~ QRS,  high ~ sharp transients/noise edge
# # --------------------------------------------------------------------------- #
# FREQ_BANDS = {
#     "low_0_5Hz":   (0.5, 5.0),
#     "mid_5_15Hz":  (5.0, 15.0),
#     "high_15_40Hz": (15.0, 40.0),
# }

# # --------------------------------------------------------------------------- #
# # Feature families. Each family is a logical group whose contribution can be
# # switched off for the ablation in ablation.py. The actual column->family map
# # is built at runtime by features.build_feature_matrix(), but the canonical
# # ordering and human-readable names live here.
# # --------------------------------------------------------------------------- #
# FEATURE_FAMILIES = ["time", "freq", "morph", "wavelet", "hrv", "graph"]

# # Which families are causal/streaming-safe at the single-beat level.
# # 'graph' and 'hrv' need a trailing window of previous beats; this is recorded
# # so the manuscript can state the true streaming latency honestly (fix #2).
# STREAMING_REQUIRES_WINDOW = {"hrv", "graph"}
# GRAPH_WINDOW_BEATS = 32   # trailing causal window for graph/HRV features
# GRAPH_TAU = 8.0           # temporal decay constant (in beats, not samples)


"""
Central configuration: AAMI labels, inter-patient splits, frequency bands,
and the canonical list of feature families used for ablation.

The four manuscript fixes this package implements:
  (1) Inter-patient evaluation + no-leakage pipeline  -> data_io.py + pipeline.py
  (2) Size/transparency story instead of latency       -> reporting in run_experiments.py
  (3) Separability metric + feature-family ablation     -> separability.py + ablation.py
  (4) "Recovers black-box explanations cheaply"         -> cnn_baseline.py + xai_alignment.py
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Signal / processing constants
# --------------------------------------------------------------------------- #
TARGET_FS = 360            # Hz; INCART (257 Hz) is resampled to this
BANDPASS = (0.5, 40.0)     # Hz, Butterworth passband
BP_ORDER = 4               # filter order (realised as cascaded biquads)
RANDOM_STATE = 13

# Beat window relative to the R-peak, in SECONDS. The original paper used an
# adaptive composite-loss window; for an AAMI *classification benchmark* the
# accepted practice is a fixed physiological window anchored on the annotated
# R-peak. This removes a segmentation-induced confound and is what reviewers
# expect for the headline numbers. (The adaptive window remains available in
# segmentation.py for the deployment/detection narrative.)
PRE_R_SEC = 0.25
POST_R_SEC = 0.45
SEG_LEN = int(round((PRE_R_SEC + POST_R_SEC) * TARGET_FS))  # samples per beat

# --------------------------------------------------------------------------- #
# AAMI class scheme (5 classes). Standard MIT-BIH symbol -> AAMI mapping.
# --------------------------------------------------------------------------- #
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
SYMBOL_TO_AAMI = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced
    "/": "Q", "f": "Q", "Q": "Q",
}

# --------------------------------------------------------------------------- #
# Inter-patient split for MIT-BIH (de Chazal et al., 2004). Paced records
# 102, 104, 107, 217 are excluded by convention. 22 train / 22 test, patient-
# (record-) disjoint. This is the single most important correctness fix.
# --------------------------------------------------------------------------- #
MITDB_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
             124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
MITDB_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
             212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
MITDB_PACED_EXCLUDED = [102, 104, 107, 217]

# --------------------------------------------------------------------------- #
# Frequency bands (Hz) used both for band-power features and for the XAI
# alignment common space. Chosen to be physiologically meaningful:
#   low   ~ T-wave / baseline,  mid ~ QRS,  high ~ sharp transients/noise edge
# --------------------------------------------------------------------------- #
FREQ_BANDS = {
    "low_0_5Hz":   (0.5, 5.0),
    "mid_5_15Hz":  (5.0, 15.0),
    "high_15_40Hz": (15.0, 40.0),
}

# --------------------------------------------------------------------------- #
# Feature families. Each family is a logical group whose contribution can be
# switched off for the ablation in ablation.py. The actual column->family map
# is built at runtime by features.build_feature_matrix(), but the canonical
# ordering and human-readable names live here.
# --------------------------------------------------------------------------- #
FEATURE_FAMILIES = ["time", "freq", "morph", "wavelet", "hrv", "graph"]

# Which families are causal/streaming-safe at the single-beat level.
# 'graph' and 'hrv' need a trailing window of previous beats; this is recorded
# so the manuscript can state the true streaming latency honestly (fix #2).
STREAMING_REQUIRES_WINDOW = {"hrv", "graph"}
GRAPH_WINDOW_BEATS = 16   # trailing causal window for graph/HRV features
                          # (operating point chosen from the window sweep:
                          #  best weighted-F1 at lowest latency; see Results)
GRAPH_TAU = 8.0           # temporal decay constant (in beats, not samples)