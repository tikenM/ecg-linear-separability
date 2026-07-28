# ECG-XAI: Interpretable, Resource-Efficient Arrhythmia Detection

Hybrid feature-engineering pipeline for inter-patient ECG arrhythmia classification.  
The framework combines time-domain, frequency-domain, morphological, wavelet, HRV, and causal graph-theoretic features with a linear classifier under a strict leakage-controlled evaluation protocol.

## Installation

```bash
pip install -r requirements.txt


python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incartdb')"


python -m ecg_xai.run_experiments \
    --mitdb data/mitdb \
    --incartdb data/incartdb \
    --out results/ \
    --cnn \
    --epochs 30
