"""Synthetic smoke test - verifies every non-torch code path runs end to end."""
import sys
import numpy as np
sys.path.insert(0, "/home/claude/ecg_xai")

from ecg_xai.config import TARGET_FS, SEG_LEN, AAMI_CLASSES
from ecg_xai.data_io import BeatSet
from ecg_xai import features
from ecg_xai.pipeline_v1 import build_pipeline, evaluate_interpatient, cross_val_grouped
from ecg_xai.separability import separability_report
from ecg_xai.ablation import ablation_study
from ecg_xai.xai_alignment import linear_band_importance, cnn_band_importance, agreement, cost_ratio, extract_coef

import warnings
import numpy as np

# Suppress known non-critical numerical warnings from sklearn/numpy
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message=".*divide by zero encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message=".*overflow encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message=".*invalid value encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message=".*invalid value encountered in.*")

rng = np.random.default_rng(0)

def synth_beatset(n_records, beats_per_rec, prefix):
    segs, labs, grps, rrp, rrn = [], [], [], [], []
    L = SEG_LEN
    t = np.linspace(0, 1, L)
    for r in range(n_records):
        # class-dependent morphology so the problem is learnable but noisy
        for _ in range(beats_per_rec):
            cls = rng.choice(AAMI_CLASSES, p=[0.5, 0.15, 0.2, 0.1, 0.05])
            width = {"N": 1.0, "S": 1.1, "V": 2.2, "F": 1.6, "Q": 0.7}[cls]
            amp = {"N": 1.0, "S": 0.8, "V": 1.6, "F": 1.2, "Q": 0.5}[cls]
            qrs = amp * np.exp(-((t - 0.4) ** 2) / (2 * (0.02 * width) ** 2))
            twave = 0.3 * np.exp(-((t - 0.65) ** 2) / (2 * 0.05 ** 2))
            base = qrs + twave
            seg = np.stack([base + 0.08 * rng.standard_normal(L),
                            0.7 * base + 0.08 * rng.standard_normal(L)], axis=0)
            segs.append(seg.astype(np.float32)); labs.append(cls)
            grps.append(f"{prefix}{r:02d}")
            rrp.append(0.8 + 0.1 * rng.standard_normal())
            rrn.append(0.8 + 0.1 * rng.standard_normal())
    return BeatSet(np.stack(segs), np.array(labs), np.array(grps),
                   np.array(rrp, np.float32), np.array(rrn, np.float32))

print("building synthetic train/test (patient-disjoint) ...")
train = synth_beatset(8, 60, "TR")   # 8 records
test = synth_beatset(4, 60, "TE")    # 4 disjoint records
print("  train beats", len(train), "test beats", len(test),
      "| train groups", len(np.unique(train.groups)))

print("feature extraction ...")
Xtr, names, family, band_map = features.build_feature_matrix(train)
Xte, *_ = features.build_feature_matrix(test)
print("  X_train", Xtr.shape, "| families:",
      {f: sum(v == f for v in family.values()) for f in set(family.values())})
print("  band-mapped cols:", len(band_map))

print("leakage-safe inter-patient evaluation ...")
for name in ("svc", "lr", "dt"):
    res = evaluate_interpatient(build_pipeline(name), Xtr, train.labels,
                                Xte, test.labels, name=name)
    print(f"  {name}: test wF1={res.metrics['f1_weighted']:.3f} "
          f"acc={res.metrics['acc']:.3f} size={res.model_size_kb:.2f}KB")

print("grouped CV (LinearSVC) ...")
cv = cross_val_grouped(lambda: build_pipeline("svc"), Xtr, train.labels,
                       train.groups, n_splits=4)
print("  CV wF1:", round(cv["f1_weighted"]["mean"], 3), cv["f1_weighted"]["ci95"])

print("separability report ...")
sep = separability_report(Xtr, train.labels, train.groups)
print("  linear train acc:", round(sep["linear_train"]["train_accuracy"], 3),
      "| fisher trace ratio:", round(sep["fisher"]["trace_ratio"], 3),
      "| kernel gap:", round(sep["kernel_gap"]["gap_rbf_minus_linear"], 4),
      "| pairwise sep frac:", round(sep["pairwise"]["separable_fraction"], 2))

print("feature-family ablation ...")
abl = ablation_study(Xtr, train.labels, train.groups, names, family,
                     classifier="svc", n_splits=4)
print("  full wF1:", round(abl["full"]["f1_weighted"]["mean"], 3))
print("  delta (leave-one-out):", abl["delta_f1_weighted_leave_one_out"])

print("XAI alignment math (mock CNN attribution) ...")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
Xs = StandardScaler().fit_transform(Xtr)
lin = OneVsRestClassifier(
    LogisticRegression(solver="liblinear", class_weight="balanced",
                       max_iter=2000)).fit(Xs, train.labels)
lin_vec = linear_band_importance(extract_coef(lin), names, band_map)
# mock attribution: energy concentrated in QRS band -> shape (n,leads,L)
mock_attr = np.abs(test.segments[:64])
cnn_vec = cnn_band_importance(mock_attr)
ag = agreement(lin_vec, cnn_vec)
print("  linear band importance:", ag["linear_band_importance"])
print("  cnn band importance   :", ag["cnn_band_importance"])
print("  cosine:", ag["cosine_similarity"], "| spearman:", ag["spearman_rho"],
      "| top-band match:", ag["top_band_match"])
print("  cost ratio (mock):", cost_ratio(extract_coef(lin).size, 50000))

print("\nALL PATHS OK")
