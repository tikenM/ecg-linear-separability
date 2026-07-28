# ECG-XAI: interpretable, resource-efficient arrhythmia framework

A hybrid feature-engineering pipeline for AAMI-class arrhythmia
classification on MIT-BIH and INCART, built around four core correctness
fixes and extended with a second round of additions written in response to
peer review. The headline thesis: **separability + transparency + cheap
recovery of black-box explanations**, evaluated with the statistical rigor
a deployable clinical claim requires.

## Repository layout

```
ecg_xai/
  config.py                  constants: AAMI classes, MITDB splits, feature families
  data_io.py                 WFDB loading, patient-disjoint splits, INCART patient grouping
  preprocessing.py           zero-phase bandpass
  features.py                feature extraction by family (time/freq/morph/wavelet/hrv/graph)
  pipeline.py                leakage-safe imblearn pipeline, AdaptiveSMOTEENN, metrics
  pipeline_v1.py              (not included here -- see "External dependencies" below)
  separability.py            linear separability evidence + kernel-gap CI + tolerance sweep
  ablation.py                 feature-family ablation + graph cost-benefit
  significance.py             paired CV tests, McNemar, patient-level bootstrap/Wilcoxon
  cluster_separability.py     silhouette / Davies-Bouldin / Calinski-Harabasz + t-SNE
  threshold_analysis.py       minority-class PR curves for the linear SVC
  smote_report.py             class-wise distributions before/after AdaptiveSMOTEENN
  detection_eval.py           R-peak detector scoring + reference-to-detected label transfer
  cnn_baseline.py             1D-CNN black-box baseline (torch, optional)
  xai_alignment.py            linear-vs-CNN frequency-band explanation agreement
  analysis_extra.py           (not included here -- see "External dependencies" below)
  run_experiments.py          main end-to-end driver
  run_smote_report.py         standalone SMOTE-ENN distribution report
smoke_test.py                 synthetic-data sanity check, no real data required
extract_reviewer_numbers.py   pulls response-to-reviewers numbers out of results.json
response_to_reviewers.md      point-by-point reviewer response, code-linked
requirements.txt
```

## Install

```bash
pip install -r requirements.txt          # torch only needed for the CNN/XAI stage
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incartdb')"
```

## Run

```bash
python -m ecg_xai.run_experiments \
    --mitdb data/mitdb --incartdb data/incartdb \
    --out results/ --cnn --epochs 30
```

Flags are independent; drop `--incartdb` or `--cnn` if you don't need those
stages. `--extra` runs the leakage/headroom/causality/runtime/window-sweep
experiments (needs `analysis_extra.py`, see below).

Writes `results/results.json` plus two figures, `results/tsne_feature_space.png`
and `results/pr_curve_minority_classes.png`.

### Standalone: SMOTE-ENN class distribution report

Independent of the main driver:

```bash
python -m ecg_xai.run_smote_report --mitdb data/mitdb --classifier svc --out results/smote_report.json
```

Prints before / after-SMOTE / after-ENN class counts, both as a single
DS1-training-set table and per patient-grouped CV fold (matching the exact
folds `cross_val_grouped` uses, since `AdaptiveSMOTEENN`'s chosen `k` and its
fallback-to-plain-oversampling branch can differ fold to fold).

## What's in `results.json`

| Key | From | Fix / comment addressed |
|---|---|---|
| `mitdb.interpatient`, `mitdb.grouped_cv` | original | inter-patient eval, no leakage |
| `mitdb.separability` | `separability.py` | linear/RBF kernel gap, now with per-fold CI, paired test, and an epsilon-tolerance sweep |
| `mitdb.ablation` | `ablation.py` | leave-one-family-out / only-one-in |
| `mitdb.significance_cv` | `significance.py` | paired classifier comparison on identical CV folds |
| `mitdb.significance_mcnemar_descriptive` | `significance.py` | per-beat McNemar -- **descriptive only**, overstates evidence under within-patient correlation |
| `mitdb.significance_patient_cluster_bootstrap`, `mitdb.significance_patientwise` | `significance.py` | patient-level inference; these are the headline significance results |
| `mitdb.separability_extended` | `cluster_separability.py` | silhouette / Davies-Bouldin / Calinski-Harabasz + t-SNE, independent of any fitted classifier |
| `mitdb.minority_class_pr` | `threshold_analysis.py` | SVEB/VEB precision-recall curves and threshold sweep for the linear SVC |
| `incart.allocation` | `data_io.py` | patient-level (not record-level) hold-out, with the exact patient/record IDs per partition and a hard disjointness check |
| `xai_alignment` | `xai_alignment.py`, `cnn_baseline.py` | linear-vs-CNN frequency-band agreement (needs `--cnn`) |
| `extra.*` | `analysis_extra.py` | leakage stress test, nonlinear headroom, causal-vs-acausal graph, runtime profile, window sweep (needs `--extra`) |

## Reviewer-response additions

This revision responds to six review comments. The mapping from comment to
code, and what still needs a real-data run to complete, is documented in
full in [`response_to_reviewers.md`](./response_to_reviewers.md). Summary:

1. **Patient-level significance** (`significance.py`). Per-beat McNemar
   treats correlated within-patient beats as independent and overstates
   evidence; superseded by a patient cluster bootstrap and a patient-wise
   paired Wilcoxon test, both Holm-Bonferroni corrected.
2. **Kernel-gap uncertainty** (`separability.py`). The linear-vs-RBF gap now
   reports a per-fold 95% CI, a paired significance test, and a sweep over
   five separability tolerances (epsilon), so the conclusion's sensitivity to
   an arbitrarily chosen epsilon is shown rather than asserted. The term
   "separability certificate" has been retired in favor of language the
   evidence actually supports.
3. **"Unbiased estimate of population risk" claim.** Manuscript-text only;
   no code asserted this. See `response_to_reviewers.md` for the revised
   proposition.
4. **Graph feature cost-benefit** (`ablation.py`). Isolates graph's marginal
   benefit *given HRV is already present* (not just leave-one-out), and joins
   every family's F1 delta to its measured per-beat cost.
5. **INCART patient-level grouping** (`data_io.py`). The original hold-out
   grouped by *record*; INCART's 75 records come from only 32 patients, so
   this could split one patient's records across train and test. Fixed by
   parsing the patient number from each `.hea` file and grouping on it, with
   a hard runtime assertion that no patient appears in both partitions.
6. **Detected-peak evaluation** (`detection_eval.py`). A harness --
   AAMI-style R-peak matching, detection scoring, and reference-to-detected
   label transfer -- for running the classification benchmark on your
   ensemble detector's output instead of reference annotations. Not wired
   into the CLI: supply your detector as a callable.

## External dependencies not included in this repository snapshot

Two modules are imported by `run_experiments.py` but were not part of this
revision pass:

- **`pipeline_v1.py`** -- imported for `build_pipeline`, `evaluate_interpatient`,
  `cross_val_grouped`, `classifier_size_kb`. All testing in this revision
  used `pipeline.py`'s implementation as a stand-in for these four names;
  confirm `pipeline_v1.py`'s signatures match before trusting the wiring on
  a real run.
- **`analysis_extra.py`** -- required for `--extra` and part of `--cnn`
  (`alignment_significance`). Not exercised by any test in this revision.

## Testing status

Every function in the reviewer-response additions was run against the real
`data_io.py` / `features.py` / `pipeline.py` on synthetic patient-grouped
beat data, with hand-verified correctness checks where a wrong answer would
otherwise be silent: e.g. `decision_function` column order was checked
against `pipe.predict()` (100% agreement required), INCART patient
grouping was checked to prevent a record of a held-out patient from
appearing in the training set, and R-peak matching in `detection_eval.py`
was checked against a hand-constructed case with a known TP/FP/FN count.

None of this substitutes for running on real MIT-BIH/INCART data. Numbers
in `response_to_reviewers.md` are marked `[TO FILL]` rather than estimated,
and should be populated by `extract_reviewer_numbers.py` after a real run
-- see that script's docstring for which `results.json` key feeds which
placeholder.

