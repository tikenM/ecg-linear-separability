"""
End-to-end driver. Runs the four experiments and writes results.

Usage
-----
    python -m ecg_xai.run_experiments \
        --mitdb /path/to/mitdb --incartdb /path/to/incartdb \
        --out results/ --cnn            # --cnn requires torch

Each stage is independent; comment out what you don't need. Real data lives on
PhysioNet (`wfdb.dl_database('mitdb', 'mitdb')`), which must be downloaded on
your machine -- it is not bundled here.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import warnings

# Suppress known non-critical numerical warnings from sklearn/numpy
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*divide by zero encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*overflow encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered in matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered in.*")

from . import data_io, features
from .pipeline import (
    build_pipeline, evaluate_interpatient, cross_val_grouped, classifier_size_kb,
)
from .separability import separability_report
from .ablation import ablation_study
from .significance import (
    paired_cv_comparison, pairwise_mcnemar,
    patientwise_paired_test, patient_cluster_bootstrap,
)
from .cluster_separability import separability_report_extended, plot_embedding
from .threshold_analysis import minority_class_pr_report, plot_pr_curves, report_for_json


def _featurize(beats):
    X, names, family, band_map = features.build_feature_matrix(beats)
    return X, names, family, band_map


def run_mitdb(mitdb_dir: str, classifiers=("svc", "lr", "dt"),
             out_dir: str = "results") -> dict:
    print("[MIT-BIH] loading inter-patient DS1/DS2 ...")
    train, test = data_io.mitdb_interpatient(mitdb_dir)
    print(f"  train beats={len(train)}  test beats={len(test)}")

    Xtr, names, family, band_map = _featurize(train)
    Xte, *_ = _featurize(test)
    ytr, yte = train.labels, test.labels
    gtr = train.groups

    out = {"n_train": len(train), "n_test": len(test),
           "interpatient": {}, "grouped_cv": {}}

    test_preds = {}
    fitted_pipes = {}
    for name in classifiers:
        pipe = build_pipeline(name)
        res = evaluate_interpatient(pipe, Xtr, ytr, Xte, yte, name=name)
        test_preds[name] = pipe.predict(Xte)   # reuse the already-fitted pipe, no refit
        fitted_pipes[name] = pipe              # kept for the PR-curve stage below
        out["interpatient"][name] = {
            "metrics": res.metrics, "model_size_kb": res.model_size_kb}
        cv = cross_val_grouped(lambda n=name: build_pipeline(n),
                               Xtr, ytr, gtr, n_splits=5)
        out["grouped_cv"][name] = cv
        print(f"  {name}: test wF1={res.metrics['f1_weighted']:.4f} "
              f"acc={res.metrics['acc']:.4f}  CV wF1={cv['f1_weighted']['mean']:.4f}"
              f" {cv['f1_weighted']['ci95']}")

    # Separability + ablation on the engineered (training) space.
    print("[MIT-BIH] separability ...")
    out["separability"] = separability_report(Xtr, ytr, gtr)
    print("[MIT-BIH] feature-family ablation ...")
    out["ablation"] = ablation_study(Xtr, ytr, gtr, names, family, classifier="svc")

    # Statistical significance between classifiers (reviewer requirement):
    # paired CV (same folds across classifiers) + McNemar on the shared
    # DS1->DS2 test set, both Holm-Bonferroni corrected across the 3 pairs.
    print("[MIT-BIH] statistical significance (paired CV + McNemar) ...")
    make_pipes = {n: (lambda n=n: build_pipeline(n)) for n in classifiers}
    out["significance_cv"] = paired_cv_comparison(make_pipes, Xtr, ytr, gtr, n_splits=5)
    # Per-beat McNemar is retained ONLY as a descriptive discordance summary; it
    # treats correlated within-patient beats as independent and so overstates
    # evidence. It is NOT the headline inferential result.
    out["significance_mcnemar_descriptive"] = pairwise_mcnemar(yte, test_preds)

    # Patient-level inference (reviewer requirement): resampling / test unit is
    # the patient, not the beat. These supersede McNemar for any significance
    # claim on the inter-patient test set.
    print("[MIT-BIH] patient-level significance (cluster bootstrap + patient-wise) ...")
    out["significance_patient_cluster_bootstrap"] = patient_cluster_bootstrap(
        yte, test_preds, test.groups, metric="f1_weighted", n_boot=2000)
    out["significance_patientwise"] = patientwise_paired_test(
        yte, test_preds, test.groups, metric="acc")
    for k, d in out["significance_patient_cluster_bootstrap"]["pairwise"].items():
        print(f"  {k}: diff={d['observed_diff']:+.4f} CI={ [round(x,4) for x in d['ci']] } "
              f"p={d['p_bootstrap']:.4f}")

    # Classifier-independent separability geometry (reviewer requirement):
    # silhouette / Davies-Bouldin / Calinski-Harabasz + a t-SNE figure, to
    # complement (not replace) the linear-vs-RBF kernel-gap test above.
    print("[MIT-BIH] extended separability (silhouette/DB/CH + t-SNE) ...")
    ext = separability_report_extended(Xtr, ytr, gtr)
    plot_path = os.path.join(out_dir, "tsne_feature_space.png")
    plot_embedding(ext["embedding_2d"], plot_path, class_order=["N", "S", "V", "F", "Q"])
    # Raw arrays in embedding_2d aren't JSON-serializable; keep metadata +
    # the saved figure path only.
    ext["embedding_2d"] = {
        "method_used": ext["embedding_2d"]["method_used"],
        "n_points": ext["embedding_2d"]["n_points"],
        "plot_saved_to": plot_path,
    }
    out["separability_extended"] = ext

    # Minority-class (SVEB/VEB) precision-recall trade-off for the linear SVC
    # (reviewer requirement): the headline PPV figure corresponds to the
    # pipeline's default (threshold=0) one-vs-rest decision boundary; the
    # sweep table shows how PPV/sensitivity move as that boundary shifts.
    if "svc" in fitted_pipes:
        print("[MIT-BIH] minority-class PR curves (SVEB/VEB, linear SVC) ...")
        pr_report = minority_class_pr_report(fitted_pipes["svc"], Xte, yte,
                                             minority_classes=("S", "V"))
        pr_plot_path = os.path.join(out_dir, "pr_curve_minority_classes.png")
        plot_pr_curves(pr_report, pr_plot_path)
        pr_json = report_for_json(pr_report)
        pr_json["_plot_saved_to"] = pr_plot_path
        out["minority_class_pr"] = pr_json
        for c in ("S", "V"):
            op = pr_report[c]["default_threshold_operating_point"]
            print(f"  {c}: deployed PPV={op['ppv_positive_predictivity']:.4f} "
                  f"sens={op['sensitivity_recall']:.4f} "
                  f"FP:TP={op['false_alarms_per_true_positive']} "
                  f"AP={pr_report[c]['average_precision']:.4f}")
    else:
        print("[MIT-BIH] skipping minority-class PR curves: 'svc' not in "
              "classifiers (decision_function-based PR curve needs a "
              "margin-based classifier).")

    out["_artifacts"] = {"names": names, "family": family, "band_map": band_map}
    out["_train_cache"] = (Xtr, ytr, train)   # for optional CNN/XAI stage
    out["_test_cache"] = (Xte, yte, test)
    return out


def run_incart(incart_dir: str, classifiers=("svc", "lr", "dt"),
               record_to_patient=None) -> dict:
    # PATIENT-level (not record-level) hold-out: INCART's 75 records come from
    # only 32 patients, and each .hea carries a patient number (1-32). Grouping
    # by record would let two records of the same patient straddle train/test.
    # incart_patient_grouped_holdout reads the patient number from each header
    # (or uses record_to_patient if supplied), groups on it, and returns an
    # allocation report proving patient-disjointness (reviewer requirement).
    print("[INCART] loading + PATIENT-grouped hold-out ...")
    records = [f"I{n:02d}" for n in range(1, 76)]
    train, test, allocation = data_io.incart_patient_grouped_holdout(
        incart_dir, records, leads=("II", "V1"), test_frac=0.3,
        record_to_patient=record_to_patient)
    print(f"  patients: train={allocation['n_patients_train']} "
          f"test={allocation['n_patients_test']}  "
          f"disjoint={allocation['patient_disjoint']}")
    Xtr, names, family, band_map = _featurize(train)
    Xte, *_ = _featurize(test)
    out = {"n_train": len(train), "n_test": len(test),
           "allocation": allocation, "interpatient": {}}
    for name in classifiers:
        pipe = build_pipeline(name)
        res = evaluate_interpatient(pipe, Xtr, train.labels, Xte, test.labels, name=name)
        out["interpatient"][name] = {
            "metrics": res.metrics, "model_size_kb": res.model_size_kb}
    return out


def run_xai(mit_out: dict, epochs: int = 30) -> dict:
    """fix #4: needs torch. Trains CNN on raw train beats, aligns to linear."""
    from .cnn_baseline import ECGCNN
    from .xai_alignment import alignment_report

    Xtr, ytr, train = mit_out["_train_cache"]
    Xte, yte, test = mit_out["_test_cache"]
    names = mit_out["_artifacts"]["names"]
    band_map = mit_out["_artifacts"]["band_map"]

    # Interpretable linear model with explicit per-class coefficients on the
    # engineered (standardized) feature space.
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    Xs = StandardScaler().fit_transform(Xtr)
    lin_clf = OneVsRestClassifier(
        LogisticRegression(solver="liblinear", class_weight="balanced",
                           max_iter=2000)).fit(Xs, ytr)

    cnn = ECGCNN(n_leads=train.segments.shape[1],
                 seg_len=train.segments.shape[2],
                 n_classes=len(np.unique(ytr)), epochs=epochs)
    cnn.fit(train.segments, ytr)

    rep = alignment_report(lin_clf, names, band_map, cnn, test.segments[:512])

    # robustness of the alignment (Definition 3): bootstrap CI + permutation null
    from .xai_alignment import extract_coef, linear_band_importance
    from .analysis_extra import alignment_significance
    lin_vec = linear_band_importance(extract_coef(lin_clf), names, band_map)
    sal = cnn.saliency(test.segments[:512])
    rep["alignment_significance"] = alignment_significance(lin_vec, sal)
    return rep


def run_extra(mit_out: dict, classifier: str = "svc") -> dict:
    """Extra experiments that support the propositions and the complexity lemma.
    Uses the cached MIT-BIH train/test beats and features."""
    from .analysis_extra import (
        leakage_stress_test, nonlinear_headroom, causal_vs_acausal_graph,
        runtime_profile, window_size_sweep)

    Xtr, ytr, train = mit_out["_train_cache"]
    Xte, yte, test = mit_out["_test_cache"]
    X_all = np.vstack([Xtr, Xte])
    y_all = np.concatenate([ytr, yte])

    print("[extra] leakage stress test (Prop. 4) ...")
    leak = leakage_stress_test(Xtr, ytr, Xte, yte, X_all, y_all, classifier)
    print("        inflation (wF1):", leak["inflation"]["f1_weighted"])

    print("[extra] nonlinear headroom (Prop. 3) ...")
    head = nonlinear_headroom(Xtr, ytr, train.groups)
    print("        max gap over linear:", head["max_gap"])

    print("[extra] causal vs acausal graph (Prop. 1) ...")
    caus = causal_vs_acausal_graph(train, test, classifier, window=16)
    print("        leakage gap (wF1):", caus["leakage_gap_f1_weighted"])

    print("[extra] runtime profile (complexity lemma) ...")
    prof = runtime_profile(train)

    print("[extra] window-size sweep ...")

    sweep = window_size_sweep(train, test)

    return {"leakage_stress_test": leak, "nonlinear_headroom": head,
            "causal_vs_acausal": caus, "runtime_profile": prof,
            "window_size_sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitdb", required=True)
    ap.add_argument("--incartdb", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cnn", action="store_true", help="run XAI alignment (needs torch)")
    ap.add_argument("--extra", action="store_true",
                    help="run extra experiments (leakage, headroom, causality, runtime, window)")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bundle = {"mitdb": run_mitdb(args.mitdb, out_dir=args.out)}
    if args.incartdb:
        bundle["incart"] = run_incart(args.incartdb)
    if args.extra:
        bundle["extra"] = run_extra(bundle["mitdb"])
    if args.cnn:
        bundle["xai_alignment"] = run_xai(bundle["mitdb"], epochs=args.epochs)

    # strip non-serializable caches before dumping
    for k in ("_train_cache", "_test_cache", "_artifacts"):
        bundle["mitdb"].pop(k, None)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    print(f"\nWrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
