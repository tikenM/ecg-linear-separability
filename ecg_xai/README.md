# ECG-XAI: interpretable, resource-efficient arrhythmia framework

A clean reimplementation of the hybrid feature-engineering pipeline.

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


A `smoke_test.py` at the repo root exercises every non-torch path on synthetic
multi-record data and prints shapes/metrics; run it to confirm the environment.
