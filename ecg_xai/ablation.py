"""
Feature-family ablation (fix #3b).

Answers the reviewer question "which families actually do the work?" and, in
particular, whether the expensive graph stream (which breaks single-beat
streaming) earns its place. Two complementary views:

* leave-one-family-OUT : how much does removing a family hurt?
* only-one-family-IN   : how far does each family get on its own?

Everything runs through the leakage-safe grouped-CV pipeline.
"""
from __future__ import annotations

import numpy as np

from .config import FEATURE_FAMILIES
from .pipeline import build_pipeline, cross_val_grouped


def _family_mask(names, family, keep_families) -> np.ndarray:
    keep = set(keep_families)
    return np.array([family[n] in keep for n in names])


def ablation_study(X, y, groups, names, family,
                   classifier="svc", n_splits: int = 5,
                   families=FEATURE_FAMILIES) -> dict:
    families = [f for f in families if any(family[n] == f for n in names)]

    def run(mask):
        Xs = X[:, mask]
        return cross_val_grouped(
            lambda: build_pipeline(classifier, use_smote=True, use_augment=True),
            Xs, y, groups, n_splits=n_splits,
        )

    full_mask = np.ones(X.shape[1], dtype=bool)
    out = {"full": run(full_mask)}

    out["leave_one_out"] = {}
    for f in families:
        keep = [g for g in families if g != f]
        mask = _family_mask(names, family, keep)
        if mask.any():
            out["leave_one_out"][f"drop_{f}"] = run(mask)

    out["only_one_in"] = {}
    for f in families:
        mask = _family_mask(names, family, [f])
        if mask.any():
            out["only_one_in"][f"only_{f}"] = run(mask)

    # Convenience deltas vs full (weighted-F1 mean)
    base = out["full"]["f1_weighted"]["mean"]
    out["delta_f1_weighted_leave_one_out"] = {
        k: round(v["f1_weighted"]["mean"] - base, 4)
        for k, v in out["leave_one_out"].items()
    }
    return out


# --------------------------------------------------------------------------- #
# Graph cost-benefit vs a cheaper rhythm feature (reviewer comment 4).
#
# The leave-one-family-out delta answers "does graph help at all". It does NOT
# answer the reviewer's actual question: does graph add anything BEYOND the
# cheaper HRV/RR family that already carries rhythm context? These functions
# isolate graph's MARGINAL contribution given HRV is present, and pair every
# family's ablation delta with its measured per-beat compute cost.
# --------------------------------------------------------------------------- #
def graph_marginal_over_hrv(X, y, groups, names, family,
                            classifier="svc", n_splits: int = 5) -> dict:
    """Compare three configs to isolate what the graph family adds once the
    cheap rhythm family (HRV, which already contains RR statistics) is present:

      cfg_hrv_no_graph   : everything EXCEPT graph  (HRV present, graph absent)
      cfg_full           : everything              (HRV present, graph present)
      cfg_no_hrv_no_graph: everything EXCEPT hrv AND graph (rhythm context off)

    marginal_graph_given_hrv = full - hrv_no_graph
        -> the honest 'what does the expensive graph stream buy on top of the
           cheap RR/HRV features' number.
    rhythm_from_hrv_alone   = hrv_no_graph - no_hrv_no_graph
        -> how much the cheap HRV family alone recovers, for comparison.
    """
    def mask_for(keep_families):
        keep = set(keep_families)
        return np.array([family[n] in keep for n in names])

    all_families = sorted({family[n] for n in names})
    def run(keep_families):
        m = mask_for(keep_families)
        if not m.any():
            return None
        return cross_val_grouped(
            lambda: build_pipeline(classifier, use_smote=True, use_augment=True),
            X[:, m], y, groups, n_splits=n_splits)

    fams_no_graph = [f for f in all_families if f != "graph"]
    fams_no_hrv_no_graph = [f for f in all_families if f not in ("graph", "hrv")]

    full = run(all_families)
    hrv_no_graph = run(fams_no_graph)
    no_hrv_no_graph = run(fams_no_hrv_no_graph)

    def wf1(r):
        return r["f1_weighted"]["mean"] if r else float("nan")

    out = {
        "cfg_full_wf1": wf1(full),
        "cfg_hrv_no_graph_wf1": wf1(hrv_no_graph),
        "cfg_no_hrv_no_graph_wf1": wf1(no_hrv_no_graph),
        "marginal_graph_given_hrv": round(wf1(full) - wf1(hrv_no_graph), 4),
        "rhythm_from_hrv_alone": round(wf1(hrv_no_graph) - wf1(no_hrv_no_graph), 4),
        "full": full, "hrv_no_graph": hrv_no_graph,
        "no_hrv_no_graph": no_hrv_no_graph,
    }
    return out


def cost_benefit_table(leave_one_out_deltas: dict, per_family_ms: dict) -> dict:
    """Join each family's leave-one-out weighted-F1 delta to its measured
    per-beat cost (ms), giving a benefit-per-millisecond column. `per_family_ms`
    comes from the runtime profile (e.g. Figure 7 values); pass it explicitly
    rather than hardcoding, so the table always reflects the measured numbers.

    leave_one_out_deltas : {'drop_<family>': delta_wf1}  (as produced by
                           ablation_study()['delta_f1_weighted_leave_one_out'])
    per_family_ms        : {'<family>': ms_per_beat}
    """
    rows = {}
    total_ms = sum(per_family_ms.values()) if per_family_ms else float("nan")
    for key, delta in leave_one_out_deltas.items():
        fam = key.replace("drop_", "")
        ms = per_family_ms.get(fam)
        # delta is (dropped - full); benefit of KEEPING the family is -delta
        benefit = -delta
        rows[fam] = {
            "benefit_wf1_from_including": round(benefit, 4),
            "ms_per_beat": ms,
            "pct_of_total_time": (round(100 * ms / total_ms, 1)
                                  if ms is not None and total_ms else None),
            "benefit_per_ms": (round(benefit / ms, 5)
                               if ms not in (None, 0) else None),
        }
    return {"total_ms_per_beat": total_ms, "families": rows}
