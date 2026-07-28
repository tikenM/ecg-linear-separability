from ecg_xai.pipeline import build_pipeline
from ecg_xai.ablation import ablation_study, graph_marginal_over_hrv, cost_benefit_table
from ecg_xai import features
from ecg_xai.data_io import mitdb_interpatient

train, test = mitdb_interpatient("data/mitdb")
Xtr, names, family, band_map = features.build_feature_matrix(train)

gm = graph_marginal_over_hrv(Xtr, train.labels, train.groups, names, family,
                              classifier="svc", n_splits=5)
print("marginal_graph_given_hrv:", gm["marginal_graph_given_hrv"])
print("rhythm_from_hrv_alone:", gm["rhythm_from_hrv_alone"])

abl = ablation_study(Xtr, train.labels, train.groups, names, family, classifier="svc")
per_family_ms = {"time": ..., "freq": ..., "morph": ..., "wavelet": ...,
                  "hrv": ..., "graph": 0.512}   # your real Figure 7 measurements
cbt = cost_benefit_table(abl["delta_f1_weighted_leave_one_out"], per_family_ms)
print(cbt)