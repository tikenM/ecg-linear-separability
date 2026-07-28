# ECG-XAI: interpretable, resource-efficient arrhythmia framework

A clean reimplementation of the hybrid feature-engineering pipeline with the
four revisions agreed for the Q1 resubmission. The headline thesis is
re-centred from *accuracy/latency* to **separability + transparency + cheap
recovery of black-box explanations** — the claims that survive scrutiny.

## What changed, and where it lives

| # | Fix | Module(s) |
|---|-----|-----------|
| 1 | **Inter-patient evaluation + no leakage.** de Chazal DS1/DS2 for MIT-BIH; patient-grouped hold-out + GroupKFold for INCART. Every fit-transform step (impute → scale → MI/RFE select → PCA-augment → SMOTE-ENN) is inside one `imblearn` Pipeline, so it only ever sees the training fold. No beat-level shuffling anywhere. | `data_io.py`, `pipeline.py` |
| 2 | **Size/transparency story, not latency.** `classifier_size_kb` reports the on-device *decision* footprint; HRV/graph features are computed from a causal trailing window and flagged as needing a buffer, so the paper states the true per-decision latency honestly instead of quoting the classifier-only microsecond figure. | `pipeline.py`, `features.py`, `config.STREAMING_REQUIRES_WINDOW` |
| 3 | **Separability measured, plus a family ablation.** Linear-SVM training separability + margin, multiclass Fisher ratio, the **linear-vs-RBF kernel gap** (the honest test), and pairwise separable fraction; leave-one-family-out and only-one-in ablation under grouped CV. | `separability.py`, `ablation.py` |
| 4 | **"Recovers black-box explanations cheaply."** Train a 1D-CNN, take saliency / Integrated-Gradients attributions, project both the CNN and the linear coefficients into a shared 3-band frequency space, and report Spearman/cosine/top-band agreement plus the parameter cost ratio. | `cnn_baseline.py`, `xai_alignment.py` |

## Install

```bash
pip install -r requirements.txt          # torch only needed for fix #4
# Download the data on your machine (not bundled):
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incartdb')"
```

## Run

```bash
python -m ecg_xai.run_experiments \
    --mitdb data/mitdb --incartdb data/incartdb \
    --out results/ --cnn --epochs 30
```

Writes `results/results.json` with: inter-patient test metrics + per-class
recall + confusion matrices, grouped-CV mean/CI, the separability report, the
ablation deltas, and (with `--cnn`) the XAI-alignment report.

## Reviewer-facing claims this now supports

- **No leakage / inter-patient**: numbers come from disjoint patients; expect
  them to be *lower and more honest* than the original intra-patient 98.44%.
  Report the grouped-CV mean with 95% CI as the headline.
- **Separability is shown, not asserted**: if `kernel_gap ≈ 0`, an RBF boundary
  buys nothing over a linear one — that is the actual evidence the engineered
  space is (near-)linearly separable. The Decision Tree is no longer used as
  "proof".
- **Ablation answers "do the graph features earn their place?"**: read
  `delta_f1_weighted_leave_one_out['drop_graph']`. If ≈ 0, drop the graph
  stream — it also removes the streaming-latency objection (fix #2).
- **Cheap explanation recovery**: `saliency_alignment` / `integrated_gradients_alignment`
  give cosine + Spearman between the linear model's band importance and the
  CNN's, with `cost_ratio.cnn_over_linear` quantifying the parameter saving.

## Notes / honest caveats baked into the code

- The classification benchmark uses **annotated R-peaks** (standard AAMI
  practice). The ensemble detector belongs to the deployment narrative, not the
  accuracy table.
- INCART grouping is by **record id**; a few patients contribute multiple
  records, so the split is slightly conservative — documented in `data_io.py`.
- Per-record z-score normalisation is fully record-local and cannot leak.
- The XAI common space is **frequency bands**. A second axis (P/QRS/T temporal
  segments) is a natural extension if a reviewer wants morphology-level
  agreement too.

A `smoke_test.py` at the repo root exercises every non-torch path on synthetic
multi-record data and prints shapes/metrics; run it to confirm the environment.
