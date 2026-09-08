# ECG-XAI: interpretable, resource-efficient arrhythmia framework

A hybrid feature-engineering pipeline for AAMI-class arrhythmia classification on MIT-BIH and INCART. The headline thesis is **separability + transparency + cheap recovery of black-box explanations**, evaluated under a leakage-controlled inter-patient protocol with the statistical rigor a deployable clinical claim requires.

**Paper (accepted).** Moirangthem Tiken Singh, Manibhushan Yaikhom, Rabinder Kumar Prasad. *Hybrid Feature Engineering for Resource-Efficient Arrhythmia Detection in Electrocardiogram Signals: An Interpretable, Separability-Driven Framework.* Computers in Biology and Medicine. Manuscript CIBM-D-26-04155R2.

On MIT-BIH DS2 a linear SVC reaches a weighted F1 of 0.780 and a four-class macro F1 of 0.493. On INCART, with disjoint patient groups, logistic regression reaches a weighted F1 of 0.847. A same-budget 1D-CNN under the identical protocol is less accurate overall and collapses the supraventricular class. The linear model recovers the frequency evidence the network attends to (band-importance cosine 0.95–0.97) at about 43× fewer parameters and a 2.37 KB decision footprint.

## Repository layout

```
ecg_xai/
config.py                 AAMI classes, MITDB splits, feature families
data_io.py                WFDB loading, patient-disjoint splits, INCART patient grouping
preprocessing.py          zero-phase bandpass
features.py               feature extraction by family (time/freq/morph/wavelet/hrv/graph)
pipeline.py               leakage-safe imblearn pipeline, AdaptiveSMOTEENN, metrics
pipeline_v1.py            (not included here — see "External dependencies" below)
separability.py           linear vs RBF kernel-gap with CI, paired test, and ε-tolerance sweep
ablation.py               feature-family ablation and graph cost–benefit
significance.py           paired CV tests, McNemar (descriptive), patient-level bootstrap / Wilcoxon
cluster_separability.py   silhouette / Davies–Bouldin / Calinski–Harabasz + t-SNE
threshold_analysis.py     minority-class PR curves for the linear SVC
smote_report.py           class-wise distributions before / after AdaptiveSMOTEENN
detection_eval.py         R-peak detector scoring + reference-to-detected label transfer
cnn_baseline.py           1D-CNN black-box baseline (torch, optional)
xai_alignment.py          linear-vs-CNN frequency-band explanation agreement
analysis_extra.py         (not included here — see "External dependencies" below)
run_experiments.py        main end-to-end driver
run_smote_report.py       standalone SMOTE-ENN distribution report
smoke_test.py             synthetic-data sanity check, no real data required
extract_reported_numbers.py
                          maps results.json keys to manuscript-reported quantities
requirements.txt
```

## Install

```bash
pip install -r requirements.txt   # torch only needed for the CNN / XAI stage
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incartdb')"
```

## Run

```bash
python -m ecg_xai.run_experiments \
  --mitdb data/mitdb --incartdb data/incartdb \
  --out results/ --cnn --epochs 30
```

Flags are independent; drop `--incartdb` or `--cnn` if those stages are not needed. `--extra` runs the leakage, nonlinear-headroom, causality, runtime, and window-sweep experiments (requires `analysis_extra.py`; see below).

Writes `results/results.json` plus two figures, `results/tsne_feature_space.png` and `results/pr_curve_minority_classes.png`.

### Standalone: SMOTE-ENN class distribution report

Independent of the main driver:

```bash
python -m ecg_xai.run_smote_report \
  --mitdb data/mitdb --classifier svc --out results/smote_report.json
```

Prints before / after-SMOTE / after-ENN class counts, both as a single DS1-training-set table and per patient-grouped CV fold (matching the exact folds `cross_val_grouped` uses, since `AdaptiveSMOTEENN`'s chosen `k` and its fallback-to-plain-oversampling branch can differ fold to fold).

## What's in `results.json`

| Key | Source | Contents |
| --- | --- | --- |
| `mitdb.interpatient`, `mitdb.grouped_cv` | `pipeline.py` | Inter-patient evaluation and patient-grouped CV; no train–test leakage |
| `mitdb.separability` | `separability.py` | Linear / RBF kernel gap with per-fold CI, paired test, and ε-tolerance sweep |
| `mitdb.ablation` | `ablation.py` | Leave-one-family-out and only-one-in scores, including graph cost–benefit given HRV |
| `mitdb.significance_cv` | `significance.py` | Paired classifier comparison on identical CV folds |
| `mitdb.significance_mcnemar_descriptive` | `significance.py` | Per-beat McNemar — descriptive only; overstates evidence under within-patient correlation |
| `mitdb.significance_patient_cluster_bootstrap`, `mitdb.significance_patientwise` | `significance.py` | Patient-level inference (cluster bootstrap and paired Wilcoxon, Holm–Bonferroni). These are the primary significance results |
| `mitdb.separability_extended` | `cluster_separability.py` | Silhouette / Davies–Bouldin / Calinski–Harabasz + t-SNE, independent of any fitted classifier |
| `mitdb.minority_class_pr` | `threshold_analysis.py` | SVEB / VEB precision–recall curves and threshold sweep for the linear SVC |
| `incart.allocation` | `data_io.py` | Patient-level (not record-level) hold-out, with exact patient / record IDs per partition and a hard disjointness check |
| `xai_alignment` | `xai_alignment.py`, `cnn_baseline.py` | Linear-vs-CNN frequency-band agreement (needs `--cnn`) |
| `extra.*` | `analysis_extra.py` | Leakage stress test, nonlinear headroom, causal vs acausal context, runtime profile, window sweep (needs `--extra`) |

## Method notes (code ↔ paper)

These modules implement the analyses reported in the paper. They are part of the public experiment suite, not optional add-ons.

1. **Patient-level significance** (`significance.py`). Per-beat McNemar treats correlated within-patient beats as independent and overstates evidence. Primary inference uses a patient-cluster bootstrap and a patient-wise paired Wilcoxon test, both Holm–Bonferroni corrected.

2. **Kernel-gap uncertainty** (`separability.py`). The linear-vs-RBF gap reports a per-fold 95% CI, a paired significance test, and a sweep over five separability tolerances ε, so sensitivity to the operating tolerance is shown rather than asserted.

3. **Risk estimation.** Proposition 4 in the paper states that the empirical risk on a patient-disjoint test partition is a valid estimator of risk on the observed test-patient distribution. It does not claim an unbiased estimate of risk on a broader clinical population.

4. **Graph feature cost–benefit** (`ablation.py`). Isolates the graph family's marginal benefit given that HRV is already present (not only leave-one-out), and joins every family's F1 delta to its measured per-beat cost.

5. **INCART patient-level grouping** (`data_io.py`). INCART's 75 records come from 32 patients. Splits are grouped on the patient number parsed from each `.hea` file, with a hard runtime assertion that no patient appears in both partitions.

6. **Detected-peak evaluation** (`detection_eval.py`). AAMI-style R-peak matching, detection scoring, and reference-to-detected label transfer, for running the classification benchmark on an ensemble detector's output instead of reference annotations. Not wired into the CLI: supply the detector as a callable.

## External dependencies not included in this repository snapshot

Two modules are imported by `run_experiments.py` but are not part of this snapshot:

- **`pipeline_v1.py`** — imported for `build_pipeline`, `evaluate_interpatient`, `cross_val_grouped`, `classifier_size_kb`. Testing used `pipeline.py` as a stand-in for these four names; confirm `pipeline_v1.py`'s signatures match before trusting the wiring on a full run.
- **`analysis_extra.py`** — required for `--extra` and part of `--cnn` (`alignment_significance`).

## Testing status

Functions that implement the analyses above were run against `data_io.py` / `features.py` / `pipeline.py` on synthetic patient-grouped beat data, with hand-verified correctness checks where a wrong answer would otherwise be silent: e.g. `decision_function` column order was checked against `pipe.predict()` (100% agreement required), INCART patient grouping was checked to prevent a record of a held-out patient from appearing in the training set, and R-peak matching in `detection_eval.py` was checked against a hand-constructed case with a known TP / FP / FN count.

None of this substitutes for running on real MIT-BIH / INCART data. After a full run, `extract_reported_numbers.py` maps `results.json` keys to the quantities reported in the paper (see that script's docstring).

## Citation

```bibtex
@article{singh2026hybrid,
  title   = {Hybrid Feature Engineering for Resource-Efficient Arrhythmia Detection in Electrocardiogram Signals: An Interpretable, Separability-Driven Framework},
  author  = {Singh, Moirangthem Tiken and Yaikhom, Manibhushan and Prasad, Rabinder Kumar},
  journal = {Computers in Biology and Medicine},
  year    = {2026},
  note    = {Accepted. Manuscript CIBM-D-26-04155R2}
}
```
