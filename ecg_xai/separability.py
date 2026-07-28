"""
Direct separability evidence (fix #3a).

The manuscript currently *asserts* linear separability and infers it from a
Decision Tree failing (a non-sequitur). These functions measure it:

1. linear_train_separability : training accuracy + functional margin of a hard
   linear SVM on the engineered space. ~1.0 training accuracy is the actual
   operational meaning of "linearly separable".
2. fisher_ratio              : multiclass Fisher discriminant ratio
   (between/within scatter). Higher = classes more linearly separated.
3. linear_vs_kernel_gap      : grouped-CV accuracy of an RBF SVM minus a linear
   SVM. A gap near zero (or negative) means a nonlinear boundary buys nothing,
   i.e. the linear model is not leaving separability on the table. THIS is the
   honest test of the thesis.
4. pairwise_separable_fraction : fraction of one-vs-one class pairs a linear
   SVM separates with (near) zero training error.

All operate on the *standardized engineered features* the classifier sees.
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold


def _std(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def linear_train_separability(X, y, C: float = 1e4) -> dict:
    """High-C linear SVM ~ hard margin. Training accuracy near 1 == separable."""
    Xs = _std(X)
    svm = LinearSVC(C=C, dual="auto", max_iter=20000).fit(Xs, y)
    margins = svm.decision_function(Xs)
    if margins.ndim == 1:
        m = np.abs(margins)
    else:
        # multiclass OVR: margin of the true-class score over the runner-up
        srt = np.sort(margins, axis=1)
        m = srt[:, -1] - srt[:, -2]
    return {
        "train_accuracy": float(accuracy_score(y, svm.predict(Xs))),
        "mean_margin": float(np.mean(m)),
        "median_margin": float(np.median(m)),
        "frac_misclassified_train": float(1 - accuracy_score(y, svm.predict(Xs))),
    }


def fisher_ratio(X, y) -> dict:
    """Multiclass Fisher discriminant ratio on standardized features."""
    Xs = _std(X)
    classes = np.unique(y)
    m = Xs.mean(axis=0)
    Sb = np.zeros((Xs.shape[1], Xs.shape[1]))
    Sw = np.zeros_like(Sb)
    for c in classes:
        Xc = Xs[y == c]
        mc = Xc.mean(axis=0)
        Sb += len(Xc) * np.outer(mc - m, mc - m)
        Sw += (Xc - mc).T @ (Xc - mc)
    trace_ratio = float(np.trace(Sb) / (np.trace(Sw) + 1e-12))
    reg = np.trace(np.linalg.pinv(Sw + 1e-3 * np.eye(Sw.shape[0])) @ Sb)
    return {"trace_ratio": trace_ratio, "regularized_J": float(reg)}


def linear_vs_kernel_gap(X, y, groups, n_splits: int = 5,
                         rbf_C: float = 1.0, rbf_gamma="scale") -> dict:
    """Grouped-CV accuracy: RBF SVM minus linear SVM. Small gap => linear is
    enough => engineered space is (near-)linearly separable."""
    Xs = _std(X)
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    lin_acc, rbf_acc = [], []
    for tr, te in gkf.split(Xs, y, groups):
        lin = LinearSVC(C=1.0, dual="auto", max_iter=10000).fit(Xs[tr], y[tr])
        rbf = SVC(kernel="rbf", C=rbf_C, gamma=rbf_gamma).fit(Xs[tr], y[tr])
        lin_acc.append(accuracy_score(y[te], lin.predict(Xs[te])))
        rbf_acc.append(accuracy_score(y[te], rbf.predict(Xs[te])))
    lin_acc, rbf_acc = np.array(lin_acc), np.array(rbf_acc)
    per_fold_gap = rbf_acc - lin_acc

    # Uncertainty across patient-grouped folds + a PAIRED test of RBF vs linear
    # on the identical folds (reviewer comment 2: a small mean gap alone is not
    # a geometric certificate; report its uncertainty and test it).
    from scipy import stats as _stats
    n = len(per_fold_gap)
    mean_gap = float(per_fold_gap.mean())
    sd_gap = float(per_fold_gap.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(_stats.t.ppf(0.975, n - 1) * sd_gap / np.sqrt(n))
    else:
        half = 0.0
    # paired tests on the per-fold accuracies (same folds for lin and rbf)
    if np.allclose(lin_acc, rbf_acc):
        t_p, w_p = 1.0, 1.0
    else:
        t_p = float(_stats.ttest_rel(rbf_acc, lin_acc).pvalue)
        try:
            w_p = float(_stats.wilcoxon(rbf_acc, lin_acc).pvalue)
        except ValueError:
            w_p = 1.0

    return {
        "linear_acc_mean": float(lin_acc.mean()),
        "rbf_acc_mean": float(rbf_acc.mean()),
        "gap_rbf_minus_linear": mean_gap,
        "gap_std": sd_gap,
        "gap_ci95": [mean_gap - half, mean_gap + half],
        "n_folds": n,
        "per_fold_linear_acc": lin_acc.round(4).tolist(),
        "per_fold_rbf_acc": rbf_acc.round(4).tolist(),
        "per_fold_gap": per_fold_gap.round(4).tolist(),
        "paired_ttest_rbf_vs_linear_p": t_p,
        "paired_wilcoxon_rbf_vs_linear_p": w_p,
        "interpretation": ("A CI overlapping ~0 and a non-significant paired test "
                          "mean the RBF boundary does not measurably beat the linear "
                          "one on held-out patients. Report the CI and p, not the "
                          "point estimate alone."),
    }


def pairwise_separable_fraction(X, y, C: float = 1e4, tol: float = 0.01) -> dict:
    """Fraction of class pairs linearly separable to within `tol` train error."""
    Xs = _std(X)
    classes = np.unique(y)
    results = {}
    sep = 0
    pairs = list(combinations(classes, 2))
    for a, b in pairs:
        mask = (y == a) | (y == b)
        clf = LinearSVC(C=C, dual="auto", max_iter=20000).fit(Xs[mask], y[mask])
        err = 1 - accuracy_score(y[mask], clf.predict(Xs[mask]))
        results[f"{a}|{b}"] = float(err)
        sep += int(err <= tol)
    return {"separable_fraction": sep / len(pairs), "pair_train_error": results}


def pairwise_separable_fraction_sweep(
        X, y, C: float = 1e4,
        tolerances=(0.005, 0.01, 0.025, 0.05, 0.10)) -> dict:
    """Sensitivity of the 'separable fraction' to the tolerance epsilon
    (reviewer comment 2: epsilon was arbitrarily fixed). Computes the per-pair
    train errors ONCE, then reports the separable fraction at each tolerance so
    the reader sees how the conclusion moves with epsilon rather than trusting a
    single arbitrary cut."""
    base = pairwise_separable_fraction(X, y, C=C, tol=max(tolerances))
    errs = base["pair_train_error"]
    n_pairs = len(errs)
    sweep = {}
    for tol in tolerances:
        frac = sum(int(e <= tol) for e in errs.values()) / n_pairs
        sweep[f"eps_{tol}"] = {
            "tolerance": tol,
            "separable_fraction": frac,
            "n_separable": sum(int(e <= tol) for e in errs.values()),
            "n_pairs": n_pairs,
        }
    return {"pair_train_error": errs, "sweep": sweep}


def separability_report(X, y, groups) -> dict:
    return {
        "linear_train": linear_train_separability(X, y),
        "fisher": fisher_ratio(X, y),
        "kernel_gap": linear_vs_kernel_gap(X, y, groups),
        "pairwise": pairwise_separable_fraction(X, y),
        "pairwise_tolerance_sweep": pairwise_separable_fraction_sweep(X, y),
    }
