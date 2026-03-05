import numpy as np

from sklearn.datasets import fetch_openml

categ_values = {
    "FACTOR1": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18"],
    "FACTOR2": ["V1", "V2", "V3", "V4", "V8", "V9", "V10", "V11", "V13", "V14", "V15", "V16"],
    "FACTOR3": ["V2"],
    "FACTOR4": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"],
    "FACTOR5": ["V1", "V2", "V3", "V4"],
    "FACTOR6": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"],
    "FACTOR7": ["V2", "V3", "V4"],
    "FACTOR8": ["V1", "V2"],
    "FACTOR9": ["V1", "V2", "V3", "V4"],
    "FACTOR10": ["V1", "V2", "V3", "V4"],
    "FACTOR11": ["V2", "V4", "V5"],
    "FACTOR12": ["V1", "V2", "V3"],
    "FACTOR13": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12"],
    "FACTOR14": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"],
    "FACTOR15": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"],
    "FACTOR16": ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17"],
    "FACTOR17": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "FACTOR18": ["V1", "V2", "V4", "V5", "V6", "V7", "V8"],
}

def fetch_TelecomChurn(return_X_y=False, **args):
    D = fetch_openml("churn-uplift-mlg", parser="auto", **args)
    D.treatment = np.asarray(D.data.t, dtype=np.int32)
    D.data = D.data.drop(columns="t")
    D.feature_names.remove("t")
    D.treatment_values = ["0", "1"]
    D.n_trt = 1
    D.target = D.target.values.astype(np.int32)
    D.target_names = ["target"]
    D.categ_values = categ_values
    if return_X_y:
        return D.data, D.target, D.trt
    return D
