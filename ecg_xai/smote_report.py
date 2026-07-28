"""
smote_report.py

Class-wise sample distributions before and after AdaptiveSMOTEENN, matching
EXACTLY what pipeline.AdaptiveSMOTEENN does at fit time -- not an
approximation. This matters because AdaptiveSMOTEENN:
  * picks SMOTE's k_neighbors adaptively from the rarest class present
    (k = min(max_k, rarest_count - 1)), which differs per CV fold,
  * falls back to plain RandomOverSampler (no ENN cleaning at all) if the
    rarest class has fewer than 2 samples.
A generic "call SMOTEENN and count" script would silently disagree with the
real pipeline whenever either branch fires. This module re-derives the same
branch decision from the same logic in pipeline.py, so the reported counts
are guaranteed consistent with what build_pipeline(..., use_smote=True)
actually produces.

Three views, from cheapest/most honest to most detailed:

1. class_distribution(y)                  -- a plain count table.
2. smote_enn_before_after(X, y, ...)       -- single split (e.g. the full
   DS1 training set): before -> after SMOTE oversampling -> after ENN
   cleaning, with the resampler branch actually taken made explicit.
3. smote_enn_per_fold(X, y, groups, ...)   -- the SAME patient-grouped
   GroupKFold folds used by cross_val_grouped, so the reported table matches
   what happened inside the actual reported CV run rather than a separate
   global fit. Reports per-fold AND aggregated (min/median/max across folds),
   since fold-to-fold variation is itself worth stating rather than hiding
   behind one global number.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours

from .config import RANDOM_STATE
from .pipeline import build_pipeline


def class_distribution(y) -> dict:
    c, n = np.unique(y, return_counts=True)
    return {str(k): int(v) for k, v in zip(c, n)}


def _pre_balance_transform(X, y, classifier: str = "svc", use_augment: bool = True):
    """Fit impute -> scale -> (augment) exactly as build_pipeline does, and
    return the transformed X the 'balance' step actually receives. Slicing
    on the real pipeline's steps (not reconstructing them) guarantees this
    matches production, including whichever augmenter hyperparameters
    build_pipeline uses."""
    full = build_pipeline(classifier, use_smote=True, use_augment=use_augment)
    names = [n for n, _ in full.steps]
    if "balance" not in names:
        raise ValueError("pipeline has no 'balance' step; call with use_smote=True")
    idx = names.index("balance")
    pre = SkPipeline(full.steps[:idx])
    Xt = pre.fit_transform(X, y)
    balancer_template = full.named_steps["balance"]  # for max_k / enn_neighbors params
    return Xt, balancer_template


def _adaptive_smoteenn_stages(Xt, y, balancer_template) -> dict:
    """Re-derive AdaptiveSMOTEENN's own branch decision (pipeline.py logic,
    copied exactly) and run each stage separately so before/after-SMOTE/
    after-ENN counts are all observable, not just the final result."""
    counts = Counter(y)
    min_count = min(counts.values())
    max_k = balancer_template.max_k
    enn_neighbors = balancer_template.enn_neighbors
    rs = balancer_template.random_state

    if min_count < 2:
        ros = RandomOverSampler(random_state=rs)
        X_final, y_final = ros.fit_resample(Xt, y)
        return {
            "branch": "RandomOverSampler_fallback",
            "reason": f"rarest class has {min_count} sample(s) (<2); SMOTE cannot fit",
            "k_neighbors_used": None,
            "after_smote": None,   # no SMOTE stage in this branch
            "after_enn": None,     # no ENN stage in this branch
            "final": class_distribution(y_final),
            "n_final": int(len(y_final)),
        }

    k = max(1, min(max_k, min_count - 1))
    smote = SMOTE(k_neighbors=k, random_state=rs)
    X_sm, y_sm = smote.fit_resample(Xt, y)
    enn = EditedNearestNeighbours(n_neighbors=enn_neighbors)
    X_final, y_final = enn.fit_resample(X_sm, y_sm)
    return {
        "branch": "SMOTE_ENN",
        "reason": f"rarest class has {min_count} sample(s) (>=2); k_neighbors={k}",
        "k_neighbors_used": k,
        "after_smote": class_distribution(y_sm),
        "n_after_smote": int(len(y_sm)),
        "after_enn": class_distribution(y_final),
        "final": class_distribution(y_final),
        "n_final": int(len(y_final)),
    }


def smote_enn_before_after(X, y, classifier: str = "svc",
                           use_augment: bool = True) -> dict:
    """Before / after-SMOTE / after-ENN class distributions for ONE split
    (e.g. the full DS1 training set), using the exact branch AdaptiveSMOTEENN
    would take on this data."""
    Xt, balancer_template = _pre_balance_transform(X, y, classifier, use_augment)
    before = class_distribution(y)
    stages = _adaptive_smoteenn_stages(Xt, np.asarray(y), balancer_template)
    return {"n_before": int(len(y)), "before": before, **stages}


def smote_enn_per_fold(X, y, groups, classifier: str = "svc",
                       use_augment: bool = True, n_splits: int = 5) -> dict:
    """Before/after distributions on the SAME patient-grouped GroupKFold
    training folds cross_val_grouped uses, so this matches the reported CV
    run rather than a separate global split. Reports every fold plus an
    aggregate (min/median/max final count per class across folds), since the
    adaptive k and the ENN fallback can differ fold to fold."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    folds = []
    for i, (tr, _te) in enumerate(gkf.split(X, y, groups)):
        rep = smote_enn_before_after(X[tr], y[tr], classifier, use_augment)
        rep["fold"] = i
        folds.append(rep)

    classes = sorted({c for f in folds for c in f["before"]})
    agg = {}
    for c in classes:
        before_vals = [f["before"].get(c, 0) for f in folds]
        final_vals = [f["final"].get(c, 0) for f in folds]
        agg[c] = {
            "before_min_median_max": [min(before_vals), int(np.median(before_vals)), max(before_vals)],
            "final_min_median_max": [min(final_vals), int(np.median(final_vals)), max(final_vals)],
        }
    branches = [f["branch"] for f in folds]
    return {
        "n_folds": n_splits, "folds": folds, "aggregate_by_class": agg,
        "branch_per_fold": branches,
        "note": ("If 'branch_per_fold' is not uniformly 'SMOTE_ENN', at least one "
                "fold fell back to plain oversampling because its rarest class had "
                "<2 samples; report that explicitly rather than averaging over it."),
    }


def format_before_after_table(report: dict, classes_order=("N", "S", "V", "F", "Q")) -> str:
    """Plain-text table for a single-split report (smote_enn_before_after
    output), in manuscript class order, ready to paste into a table draft."""
    lines = [f"{'Class':<8}{'Before':>10}{'After SMOTE':>14}{'After ENN (final)':>20}"]
    for c in classes_order:
        b = report["before"].get(c, 0)
        s = report.get("after_smote", {}).get(c, "-") if report.get("after_smote") else "-"
        f = report["final"].get(c, 0)
        lines.append(f"{c:<8}{b:>10}{str(s):>14}{f:>20}")
    lines.append(f"{'Total':<8}{report['n_before']:>10}"
                f"{report.get('n_after_smote','-'):>14}{report['n_final']:>20}")
    lines.append(f"\nBranch: {report['branch']} ({report['reason']})")
    return "\n".join(lines)
