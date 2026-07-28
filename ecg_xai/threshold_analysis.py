"""
threshold_analysis.py

Addresses the reviewer's Major concern on the linear SVC's SVEB (S) operating
point: "generates five false alarms for every true positive in the SVEB
class, yielding a positive predictivity of 16.4%... a deployable system
requires configurable trade-offs. Include a Precision-Recall curve for the
minority classes (SVEB, VEB). Detail how modifying the linear decision
threshold shifts the operating point."

Mechanism
---------
The 'svc' pipeline's final estimator is a LinearSVC. With its default
multi_class='ovr', LinearSVC.decision_function returns shape
(n_samples, n_classes): column c is the one-vs-rest margin for class c, i.e.
exactly the score a clinical alarm would threshold on to say "flag this beat
as SVEB" independent of the full 5-way argmax decision. That is what makes a
per-class PR curve and a movable threshold meaningful here: the reported
16.4% PPV corresponds to the implicit threshold=0 decision boundary (the
LinearSVC prediction rule), and moving that threshold trades sensitivity for
PPV exactly as the reviewer is asking to see quantified.

Everything below operates on a FITTED pipeline (fit once on the inter-patient
training set, exactly as in run_mitdb) and the held-out DS2 test set, so the
PR curve reflects genuine inter-patient generalization, not training-fold
optimism.

Note on 'lr': OneVsRestClassifier(LogisticRegression) also exposes
decision_function (and predict_proba); the same functions work unchanged if
you want the PR curve for the 'lr' pipeline as a comparison.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def _final_estimator(pipe):
    """The fitted classifier at the end of the imblearn Pipeline."""
    return pipe.named_steps["clf"]


def get_class_scores(pipe, X) -> dict:
    """Fitted pipeline -> {class_label: decision_function column} for every
    class the final estimator was fit on. Works for LinearSVC (margin) and
    for OneVsRestClassifier-wrapped LogisticRegression (also has
    decision_function); raises clearly if the estimator has neither."""
    clf = _final_estimator(pipe)
    if not hasattr(pipe, "decision_function"):
        raise AttributeError(
            f"{type(clf).__name__} pipeline has no decision_function; "
            "use predict_proba-based scores instead for this classifier."
        )
    scores = pipe.decision_function(X)
    classes = list(clf.classes_)
    if scores.ndim == 1:
        # binary edge case: sklearn returns 1 column instead of 2
        if len(classes) != 2:
            raise ValueError("1-D decision_function but >2 classes; unexpected.")
        return {classes[1]: scores, classes[0]: -scores}
    return {c: scores[:, i] for i, c in enumerate(classes)}


def pr_curve_for_class(y_true, class_scores: dict, pos_label: str) -> dict:
    """Standard PR curve + average precision for one class, treating it as a
    one-vs-rest binary detection problem (as an alarm system would)."""
    y_true = np.asarray(y_true)
    y_bin = (y_true == pos_label).astype(int)
    scores = class_scores[pos_label]
    precision, recall, thresholds = precision_recall_curve(y_bin, scores)
    ap = average_precision_score(y_bin, scores)
    return {
        "class": pos_label,
        "n_positive": int(y_bin.sum()),
        "n_total": int(len(y_bin)),
        "precision": precision,   # len = len(thresholds)+1
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": float(ap),
    }


def operating_point_at_threshold(y_true, class_scores: dict, pos_label: str,
                                 threshold: float) -> dict:
    """PPV / sensitivity / raw counts at one specific decision threshold for
    one class, framed the way the reviewer framed the problem (FP-per-TP)."""
    y_true = np.asarray(y_true)
    y_bin = (y_true == pos_label).astype(int)
    pred = (class_scores[pos_label] >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_bin == 1)))
    fp = int(np.sum((pred == 1) & (y_bin == 0)))
    fn = int(np.sum((pred == 0) & (y_bin == 1)))
    tn = int(np.sum((pred == 0) & (y_bin == 0)))
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fp_per_tp = (fp / tp) if tp > 0 else float("inf")
    return {
        "threshold": float(threshold), "class": pos_label,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "ppv_positive_predictivity": round(ppv, 4),
        "sensitivity_recall": round(sens, 4),
        "false_alarms_per_true_positive": round(fp_per_tp, 2),
        "n_flagged": tp + fp,
    }


def threshold_sweep_table(y_true, class_scores: dict, pos_label: str,
                          n_points: int = 15) -> list[dict]:
    """A clinician-readable table: PPV/sensitivity/FP-per-TP across a grid of
    thresholds spanning the score range for `pos_label`, plus the exact
    threshold=0 row (the pipeline's actual default decision rule) so the
    reviewer's cited 16.4% PPV appears explicitly in the table."""
    scores = class_scores[pos_label]
    lo, hi = np.percentile(scores, [1, 99])
    grid = sorted(set(np.linspace(lo, hi, n_points).round(4).tolist() + [0.0]))
    return [operating_point_at_threshold(y_true, class_scores, pos_label, t)
           for t in grid]


def minority_class_pr_report(pipe, X_te, y_te, minority_classes=("S", "V")) -> dict:
    """Full report for the reviewer response: PR curve data + AP + threshold
    sweep table for each minority class, using the fitted `pipe` (must
    already be fit on the training set) evaluated on the held-out X_te/y_te."""
    class_scores = get_class_scores(pipe, X_te)
    out = {}
    for c in minority_classes:
        if c not in class_scores:
            raise ValueError(f"class {c!r} not among fitted classes "
                            f"{list(class_scores.keys())}")
        pr = pr_curve_for_class(y_te, class_scores, c)
        default_op = operating_point_at_threshold(y_te, class_scores, c, threshold=0.0)
        sweep = threshold_sweep_table(y_te, class_scores, c)
        out[c] = {
            "average_precision": pr["average_precision"],
            "n_positive_test_beats": pr["n_positive"],
            "default_threshold_operating_point": default_op,
            "threshold_sweep": sweep,
            "_pr_curve_arrays": {   # kept separate: not meant for JSON dump verbatim
                "precision": pr["precision"], "recall": pr["recall"],
                "thresholds": pr["thresholds"],
            },
        }
    return out


def plot_pr_curves(report: dict, out_path: str,
                   title: str = "Precision-Recall: minority classes (linear SVC)") -> str:
    """One PR-curve figure with both minority classes overlaid, the
    threshold=0 (deployed) operating point marked on each curve, and AP in
    the legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    colors = {"S": "tab:orange", "V": "tab:green"}
    for c, d in report.items():
        arrs = d["_pr_curve_arrays"]
        col = colors.get(c, None)
        ax.plot(arrs["recall"], arrs["precision"],
               label=f"{c}  (AP={d['average_precision']:.3f})", color=col)
        op = d["default_threshold_operating_point"]
        ax.scatter([op["sensitivity_recall"]], [op["ppv_positive_predictivity"]],
                  color=col, marker="o", s=50, zorder=5,
                  edgecolor="black", linewidth=0.8)
        ax.annotate(f"deployed\nPPV={op['ppv_positive_predictivity']:.1%}",
                   (op["sensitivity_recall"], op["ppv_positive_predictivity"]),
                   textcoords="offset points", xytext=(8, -8), fontsize=7)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (positive predictivity)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def report_for_json(report: dict) -> dict:
    """Strip the raw numpy PR-curve arrays (not cleanly JSON-serializable at
    full resolution / not needed in the results.json summary) and keep the
    threshold table + AP + default operating point, which is what the
    reviewer response actually quotes."""
    out = {}
    for c, d in report.items():
        out[c] = {k: v for k, v in d.items() if k != "_pr_curve_arrays"}
    return out
