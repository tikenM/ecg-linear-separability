"""
significance.py

Paired statistical significance testing between classifiers, addressing:
"the code does not report statistical significance tests to determine
whether the observed performance differences are meaningful."

Two complementary paired tests, both standard for this comparison:

1. paired_cv_comparison
   Runs every classifier on the IDENTICAL patient-grouped CV folds (same
   GroupKFold split object, same fold indices) so per-fold scores are
   paired observations of the same train/test partition. Paired t-test
   (parametric) and Wilcoxon signed-rank (nonparametric, robust to the
   small n_splits typical of grouped CV) are both reported since n_splits
   is usually small (5) and normality of the fold-score differences is a
   real assumption to check, not something to assert.

2. mcnemar_test / pairwise_mcnemar
   For the single inter-patient DS1->DS2 test-set comparison: a per-BEAT
   paired test using the standard 2x2 discordant-pair contingency table
   (exact binomial for small discordant counts, chi-square with continuity
   correction otherwise). This is the correct test for "did classifier A
   get more test beats right than classifier B on the SAME test beats",
   which f1_weighted alone does not establish.

Both apply Holm-Bonferroni correction across all pairwise comparisons run
in a single call, since 3 classifiers -> 3 pairwise tests inflates the
family-wise false positive rate if left uncorrected.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

from .config import RANDOM_STATE


# --------------------------------------------------------------------------- #
# 1. Paired grouped-CV comparison
# --------------------------------------------------------------------------- #
def _score(y_true, y_pred, metric: str) -> float:
    if metric == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
    if metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if metric == "acc":
        return accuracy_score(y_true, y_pred)
    raise ValueError(metric)


def paired_fold_scores(make_pipes: dict, X, y, groups, n_splits: int = 5,
                       metric: str = "f1_weighted") -> dict[str, np.ndarray]:
    """Run every classifier in `make_pipes` (name -> zero-arg pipeline
    factory) on the SAME GroupKFold split. Returns {name: (n_splits,) array}
    of per-fold scores, aligned fold-for-fold across classifiers."""
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))  # fixed once, reused for every classifier

    scores = {name: np.zeros(n_splits) for name in make_pipes}
    for fold_i, (tr, te) in enumerate(splits):
        for name, factory in make_pipes.items():
            pipe = factory()
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            scores[name][fold_i] = _score(y[te], pred, metric)
    return scores


def holm_bonferroni(pvals: dict[tuple, float]) -> dict[tuple, dict]:
    """Holm-Bonferroni step-down correction over a dict of {pair: p_value}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    for rank, (pair, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        out[pair] = {"p_raw": p, "p_holm": running_max, "rank": rank + 1}
    return out


def paired_cv_comparison(make_pipes: dict, X, y, groups, n_splits: int = 5,
                         metric: str = "f1_weighted", alpha: float = 0.05) -> dict:
    """Paired t-test + Wilcoxon signed-rank for every classifier pair, on
    identical CV folds, with Holm-Bonferroni correction across pairs."""
    fold_scores = paired_fold_scores(make_pipes, X, y, groups, n_splits, metric)
    names = list(make_pipes.keys())
    pairs = list(combinations(names, 2))

    t_p, w_p = {}, {}
    detail = {}
    for a, b in pairs:
        sa, sb = fold_scores[a], fold_scores[b]
        diff = sa - sb
        # paired t-test
        if np.allclose(sa, sb):
            t_stat, t_pval = 0.0, 1.0
        else:
            t_stat, t_pval = stats.ttest_rel(sa, sb)
        # Wilcoxon signed-rank; undefined / degenerate if all diffs are 0 or n<1
        try:
            w_stat, w_pval = stats.wilcoxon(sa, sb)
        except ValueError:
            w_stat, w_pval = np.nan, 1.0
        t_p[(a, b)] = float(t_pval)
        w_p[(a, b)] = float(w_pval)
        detail[f"{a}_vs_{b}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "fold_scores": {a: sa.round(4).tolist(), b: sb.round(4).tolist()},
            "paired_ttest": {"stat": float(t_stat), "p": float(t_pval)},
            "wilcoxon": {"stat": float(w_stat) if not np.isnan(w_stat) else None,
                        "p": float(w_pval)},
        }

    t_holm = holm_bonferroni(t_p)
    w_holm = holm_bonferroni(w_p)
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        detail[key]["paired_ttest"]["p_holm"] = t_holm[(a, b)]["p_holm"]
        detail[key]["paired_ttest"]["significant_after_correction"] = (
            t_holm[(a, b)]["p_holm"] < alpha)
        detail[key]["wilcoxon"]["p_holm"] = w_holm[(a, b)]["p_holm"]
        detail[key]["wilcoxon"]["significant_after_correction"] = (
            w_holm[(a, b)]["p_holm"] < alpha)

    return {
        "metric": metric,
        "n_folds": len(next(iter(fold_scores.values()))),
        "alpha": alpha,
        "correction": "holm-bonferroni",
        "note": ("Paired t-test assumes approx-normal fold-score differences; "
                "with n_splits=5 this is a weak assumption. Wilcoxon is the "
                "nonparametric fallback and should be treated as primary "
                "when n_folds is small or differences look skewed."),
        "pairwise": detail,
    }


# --------------------------------------------------------------------------- #
# 2. McNemar's test on paired per-beat test-set predictions
# --------------------------------------------------------------------------- #
def mcnemar_table(y_true, pred_a, pred_b) -> dict:
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    n01 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n00 = int(np.sum(~a_correct & ~b_correct))
    n11 = int(np.sum(a_correct & b_correct))
    return {"both_correct": n11, "both_wrong": n00,
            "a_right_b_wrong": n01, "a_wrong_b_right": n10}


def mcnemar_test(y_true, pred_a, pred_b, exact_threshold: int = 25) -> dict:
    """McNemar's test on discordant pairs. Uses the exact binomial test when
    the number of discordant pairs is small (chi-square is unreliable there),
    otherwise chi-square with continuity correction."""
    tbl = mcnemar_table(y_true, pred_a, pred_b)
    n01, n10 = tbl["a_right_b_wrong"], tbl["a_wrong_b_right"]
    n_disc = n01 + n10
    if n_disc == 0:
        return {**tbl, "n_discordant": 0, "test": "none (no discordant pairs)",
                "statistic": 0.0, "p": 1.0}
    if n_disc < exact_threshold:
        res = stats.binomtest(min(n01, n10), n_disc, p=0.5, alternative="two-sided")
        return {**tbl, "n_discordant": n_disc, "test": "exact_binomial",
                "statistic": None, "p": float(res.pvalue)}
    stat = (abs(n01 - n10) - 1) ** 2 / n_disc
    p = float(1 - stats.chi2.cdf(stat, df=1))
    return {**tbl, "n_discordant": n_disc, "test": "chi2_continuity_corrected",
            "statistic": float(stat), "p": p}


def pairwise_mcnemar(y_true, preds: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """All pairwise McNemar tests among classifiers in `preds`
    (name -> prediction array on the SAME test set), Holm-corrected."""
    names = list(preds.keys())
    pairs = list(combinations(names, 2))
    raw = {}
    detail = {}
    for a, b in pairs:
        res = mcnemar_test(y_true, preds[a], preds[b])
        raw[(a, b)] = res["p"]
        detail[f"{a}_vs_{b}"] = res
    holm = holm_bonferroni(raw)
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        detail[key]["p_holm"] = holm[(a, b)]["p_holm"]
        detail[key]["significant_after_correction"] = holm[(a, b)]["p_holm"] < alpha
    return {"alpha": alpha, "correction": "holm-bonferroni", "pairwise": detail}


# =========================================================================== #
# Patient-level inference (reviewer comment 1: per-beat McNemar treats tens of
# thousands of correlated beats as independent, overstating evidence).
#
# Resampling / test UNIT here is the patient (record group), not the beat.
# These are the results the manuscript should quote as headline inference; the
# per-beat McNemar above is retained only as a descriptive discordance summary
# and is NOT a valid significance test under within-patient correlation.
# =========================================================================== #
def _score_subset(y_true, y_pred, metric):
    return _score(np.asarray(y_true), np.asarray(y_pred), metric)


def patientwise_scores(y_true, preds: dict, groups, metric: str = "acc") -> dict:
    """Per-patient score for each classifier on the SAME test beats.
    Returns {name: {patient: score}}. Accuracy is default because it is defined
    for every patient regardless of which classes that patient exhibits."""
    y_true = np.asarray(y_true); groups = np.asarray(groups)
    patients = np.unique(groups)
    out = {name: {} for name in preds}
    for p in patients:
        m = groups == p
        for name, pred in preds.items():
            out[name][str(p)] = float(_score_subset(y_true[m], np.asarray(pred)[m], metric))
    return out


def patientwise_paired_test(y_true, preds: dict, groups, metric: str = "acc",
                            alpha: float = 0.05) -> dict:
    """Paired Wilcoxon across PATIENTS: per-patient score per classifier, then
    signed-rank on paired per-patient differences for each pair, Holm-corrected.
    n is the number of PATIENTS, the honest sample size for inference."""
    pw = patientwise_scores(y_true, preds, groups, metric)
    patients = sorted(next(iter(pw.values())).keys())
    names = list(preds.keys())
    pairs = list(combinations(names, 2))
    raw, detail = {}, {}
    for a, b in pairs:
        sa = np.array([pw[a][p] for p in patients])
        sb = np.array([pw[b][p] for p in patients])
        diff = sa - sb
        if np.allclose(sa, sb):
            stat, pval = np.nan, 1.0
        else:
            try:
                stat, pval = stats.wilcoxon(sa, sb)
            except ValueError:
                stat, pval = np.nan, 1.0
        raw[(a, b)] = float(pval)
        detail[f"{a}_vs_{b}"] = {
            "n_patients": len(patients),
            "mean_per_patient_diff": float(diff.mean()),
            "median_per_patient_diff": float(np.median(diff)),
            "wins_a": int(np.sum(diff > 0)), "wins_b": int(np.sum(diff < 0)),
            "ties": int(np.sum(diff == 0)),
            "wilcoxon": {"stat": float(stat) if not np.isnan(stat) else None, "p": float(pval)},
        }
    holm = holm_bonferroni(raw)
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        detail[key]["wilcoxon"]["p_holm"] = holm[(a, b)]["p_holm"]
        detail[key]["wilcoxon"]["significant_after_correction"] = holm[(a, b)]["p_holm"] < alpha
    return {"metric": metric, "unit": "patient", "alpha": alpha,
            "correction": "holm-bonferroni", "pairwise": detail}


def patient_cluster_bootstrap(y_true, preds: dict, groups, metric: str = "f1_weighted",
                              n_boot: int = 2000, alpha: float = 0.05,
                              seed: int = RANDOM_STATE) -> dict:
    """Cluster (patient-level) bootstrap of the paired metric DIFFERENCE between
    classifiers. Each replicate resamples whole PATIENTS with replacement, pools
    their beats, and recomputes both classifiers' scores. Reports observed
    difference, percentile CI, and two-sided bootstrap p per pair, Holm-corrected."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); groups = np.asarray(groups)
    patients = np.unique(groups)
    beat_idx_by_patient = {p: np.where(groups == p)[0] for p in patients}
    names = list(preds.keys())
    pairs = list(combinations(names, 2))
    observed = {(a, b): (_score_subset(y_true, preds[a], metric)
                        - _score_subset(y_true, preds[b], metric)) for a, b in pairs}
    boot_diffs = {pair: np.empty(n_boot) for pair in pairs}
    for t in range(n_boot):
        drawn = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([beat_idx_by_patient[p] for p in drawn])
        yt = y_true[idx]
        for a, b in pairs:
            da = _score_subset(yt, np.asarray(preds[a])[idx], metric)
            db = _score_subset(yt, np.asarray(preds[b])[idx], metric)
            boot_diffs[(a, b)][t] = da - db
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    raw, detail = {}, {}
    for a, b in pairs:
        d = boot_diffs[(a, b)]
        pval = float(min(1.0, 2 * min(np.mean(d >= 0), np.mean(d <= 0))))
        raw[(a, b)] = pval
        ci = [float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))]
        detail[f"{a}_vs_{b}"] = {
            "observed_diff": float(observed[(a, b)]),
            "boot_mean_diff": float(d.mean()), "ci": ci,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "p_bootstrap": pval,
        }
    holm = holm_bonferroni(raw)
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        detail[key]["p_holm"] = holm[(a, b)]["p_holm"]
        detail[key]["significant_after_correction"] = holm[(a, b)]["p_holm"] < alpha
    return {"metric": metric, "unit": "patient", "n_boot": n_boot, "alpha": alpha,
            "correction": "holm-bonferroni", "n_patients": int(len(patients)),
            "pairwise": detail,
            "note": ("Resampling unit is the patient (record group); within-patient "
                    "beat correlation is preserved. Supersedes per-beat McNemar for "
                    "inferential claims.")}
