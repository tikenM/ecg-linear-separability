"""
cluster_separability.py

Addresses: "the kernel-gap analysis alone does not fully characterize the
geometry of the feature space. Additional analyses such as feature-space
visualization (t-SNE/UMAP), silhouette scores, or other cluster separability
measures would strengthen the central claim."

These are UNSUPERVISED, classifier-independent measures of how well the AAMI
classes are separated in the standardized engineered-feature space -- they do
not depend on which classifier is fit, unlike separability.py's
linear_train_separability and linear_vs_kernel_gap, which are properties of a
fitted decision boundary. That independence is exactly what makes them a
useful complement rather than a restatement of the kernel-gap result.

Metrics
-------
* silhouette_score      : [-1, 1], higher = tighter, better-separated clusters.
                          Computed on a stratified subsample when n is large,
                          since it's O(n^2).
* davies_bouldin_score  : >= 0, LOWER is better (avg similarity of each
                          cluster to its most-similar other cluster).
* calinski_harabasz_score : higher = better (between/within variance ratio;
                          related in spirit to fisher_ratio in separability.py
                          but class-count-normalized and a standard citable
                          metric in its own right).

Visualization
-------------
* embed_2d : t-SNE (always available via sklearn) with UMAP used instead if
  installed and requested. Returns the 2D embedding array so the caller can
  plot it with whatever plotting stack the manuscript pipeline uses, plus an
  optional direct-to-PNG convenience path.

Honesty notes baked in
-----------------------
* All metrics operate on the SAME standardized engineered space the
  classifiers see (via the shared `_std` helper below), not on raw beats.
* Class-imbalance caveat: AAMI classes are extremely imbalanced (N >> S,F).
  Silhouette/DB/CH are computed on the FULL multiclass label set by default,
  which means they will be dominated by how well N separates from everything
  else. `per_class_silhouette` is provided so rare-class separability isn't
  hidden inside a single global number, since that's exactly the kind of
  aggregation a careful reviewer will ask about next.
* t-SNE has no principled global-distance interpretation (only local
  neighborhood structure is meaningful) and is stochastic; a fixed
  random_state is used but the manuscript should describe the plot as
  illustrative, not as quantitative evidence on its own -- the silhouette/DB/
  CH numbers are the quantitative claim, the plot is the intuition pump.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
)

from .config import RANDOM_STATE


def _std(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def _stratified_subsample(X, y, max_n: int, seed: int = RANDOM_STATE):
    """Class-proportional subsample, needed because silhouette_score is
    O(n^2) and MIT-BIH DS1 has tens of thousands of beats."""
    n = len(y)
    if n <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    frac = max_n / n
    idx = []
    for c, cnt in zip(classes, counts):
        c_idx = np.where(y == c)[0]
        k = max(1, int(round(cnt * frac)))
        idx.append(rng.choice(c_idx, size=min(k, len(c_idx)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def cluster_metrics(X, y, max_n: int = 5000, seed: int = RANDOM_STATE) -> dict:
    """Global, classifier-independent separability metrics on the
    standardized feature space. Subsamples (class-stratified) for the O(n^2)
    silhouette computation only; DB and CH use the full standardized data
    since they are near-linear cost."""
    Xs_full = _std(X)
    Xs_sub, y_sub = _stratified_subsample(Xs_full, np.asarray(y), max_n, seed)

    classes_present = np.unique(y_sub)
    if len(classes_present) < 2:
        raise ValueError("Need >=2 classes present in the (sub)sample.")

    sil = float(silhouette_score(Xs_sub, y_sub))
    db = float(davies_bouldin_score(Xs_full, y))
    ch = float(calinski_harabasz_score(Xs_full, y))

    return {
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "calinski_harabasz_score": round(ch, 2),
        "n_used_for_silhouette": int(len(y_sub)),
        "n_total": int(len(y)),
        "interpretation": {
            "silhouette": "higher (max 1.0) = tighter, better-separated classes",
            "davies_bouldin": "lower (min 0.0) = better-separated classes",
            "calinski_harabasz": "higher = better between/within variance ratio",
        },
    }


def per_class_silhouette(X, y, max_n: int = 5000, seed: int = RANDOM_STATE) -> dict:
    """Per-class mean silhouette, so a rare class (S, F) being poorly
    separated isn't hidden inside one global average dominated by N."""
    Xs_full = _std(X)
    Xs_sub, y_sub = _stratified_subsample(Xs_full, np.asarray(y), max_n, seed)
    sample_sil = silhouette_samples(Xs_sub, y_sub)
    out = {}
    for c in np.unique(y_sub):
        mask = y_sub == c
        out[str(c)] = {"mean_silhouette": round(float(sample_sil[mask].mean()), 4),
                       "n": int(mask.sum())}
    return out


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def embed_2d(X, y, method: str = "tsne", max_n: int = 3000,
            seed: int = RANDOM_STATE, **kwargs) -> dict:
    """2D embedding of the standardized feature space for a class-colored
    scatter plot. method='tsne' always works (sklearn). method='umap' is used
    only if the `umap-learn` package is installed; falls back to t-SNE with a
    warning otherwise so this never hard-fails a pipeline run."""
    Xs_full = _std(X)
    Xs_sub, y_sub = _stratified_subsample(Xs_full, np.asarray(y), max_n, seed)

    used = method
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(random_state=seed, **kwargs)
            emb = reducer.fit_transform(Xs_sub)
        except ImportError:
            used = "tsne (umap-learn not installed, fell back)"
            from sklearn.manifold import TSNE
            emb = TSNE(n_components=2, random_state=seed,
                      init="pca", learning_rate="auto", **kwargs).fit_transform(Xs_sub)
    else:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=seed,
                  init="pca", learning_rate="auto", **kwargs).fit_transform(Xs_sub)

    return {"embedding": emb, "labels": y_sub, "method_used": used,
            "n_points": int(len(y_sub))}


def plot_embedding(embedding_result: dict, out_path: str, class_order=None,
                   title: str = "Feature-space embedding") -> str:
    """Save a class-colored scatter of embed_2d's output to `out_path` (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb, y = embedding_result["embedding"], embedding_result["labels"]
    classes = class_order if class_order is not None else sorted(np.unique(y).tolist())

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6,
                  label=f"{c} (n={int(m.sum())})", color=cmap(i % 10))
    ax.set_title(f"{title}  [{embedding_result['method_used']}]")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def separability_report_extended(X, y, groups, max_n: int = 5000,
                                 embed_max_n: int = 3000,
                                 seed: int = RANDOM_STATE) -> dict:
    """Convenience wrapper: cluster metrics + per-class silhouette + a t-SNE
    embedding, meant to be merged alongside separability.separability_report
    (which stays as-is; this does not replace the kernel-gap analysis, it
    adds independent evidence next to it)."""
    return {
        "cluster_metrics": cluster_metrics(X, y, max_n=max_n, seed=seed),
        "per_class_silhouette": per_class_silhouette(X, y, max_n=max_n, seed=seed),
        "embedding_2d": embed_2d(X, y, method="tsne", max_n=embed_max_n, seed=seed),
    }
