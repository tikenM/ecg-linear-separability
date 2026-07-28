import numpy as np
from ecg_xai import data_io, features
from ecg_xai.analysis_extra import leakage_stress_test, causal_vs_acausal_graph

train, test = data_io.mitdb_interpatient("data/mitdb")
Xtr, names, family, band_map = features.build_feature_matrix(train)
Xte, *_ = features.build_feature_matrix(test)
ytr, yte = train.labels, test.labels
X_all = np.vstack([Xtr, Xte]); y_all = np.concatenate([ytr, yte])

leak = leakage_stress_test(Xtr, ytr, Xte, yte, X_all, y_all, "svc")
caus = causal_vs_acausal_graph(train, test, "svc", window=16)

# sanity: these three must reproduce the existing JSON before you trust the new field
print("leaky 5-class :", leak["leaky"])      # expect acc~0.8847 wF1~0.8844 macro~0.8836
print("causal        :", caus["causal"])     # expect acc~0.6962 wF1~0.7796 macro~0.3942
print("acausal       :", caus["acausal"])    # expect acc~0.7141 wF1~0.7937 macro~0.3936

# the values you need
print("safe   4-class:", round(leak["safe"]["f1_macro_4"], 3))   # expect 0.493 (matches Table 2 SVC)
print("LEAKY4        :", round(leak["leaky"]["f1_macro_4"], 3))
print("INFL4         :", leak["inflation"]["f1_macro_4"])
print("causal 4-class:", round(caus["causal"]["f1_macro_4"], 3)) # expect 0.493
print("acausal 4-cls :", round(caus["acausal"]["f1_macro_4"], 3))# the real value for the table