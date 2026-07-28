"""
run_smote_report.py

Standalone: class-wise sample distributions before/after AdaptiveSMOTEENN,
independent of the full run_experiments.py driver. Loads the real MIT-BIH
inter-patient split, featurizes it once, and prints both the single-split
table and the per-fold (grouped-CV-matched) breakdown.

Usage:
    python -m ecg_xai.run_smote_report --mitdb data/mitdb --classifier svc
"""
from __future__ import annotations

import argparse
import json

from . import data_io, features
from .smote_report import (
    smote_enn_before_after, smote_enn_per_fold, format_before_after_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb", required=True, help="path to the MIT-BIH data dir")
    ap.add_argument("--classifier", default="svc", choices=("svc", "lr", "dt"))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="optional path to write the full report as JSON")
    args = ap.parse_args()

    print("[smote-report] loading inter-patient DS1/DS2 ...")
    train, test = data_io.mitdb_interpatient(args.mitdb)
    print(f"  train beats={len(train)}")

    print("[smote-report] featurizing ...")
    Xtr, names, family, band_map = features.build_feature_matrix(train)

    print("\n=== single-split report (full DS1 training set) ===")
    single = smote_enn_before_after(Xtr, train.labels, classifier=args.classifier)
    print(format_before_after_table(single))

    print(f"\n=== per-fold report ({args.n_splits}-fold, patient-grouped, "
         f"matches cross_val_grouped) ===")
    per_fold = smote_enn_per_fold(Xtr, train.labels, train.groups,
                                  classifier=args.classifier, n_splits=args.n_splits)
    print("branch_per_fold:", per_fold["branch_per_fold"])
    for f in per_fold["folds"]:
        print(f"  fold {f['fold']}: before={f['before']} -> final={f['final']}  ({f['branch']})")
    print("\naggregate (min/median/max) per class across folds:")
    for c, d in per_fold["aggregate_by_class"].items():
        print(f"  {c}: before={d['before_min_median_max']}  final={d['final_min_median_max']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"single_split": single, "per_fold": per_fold}, fh, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
