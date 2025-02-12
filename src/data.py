import pandas as pd
import numpy as np
import wfdb
import ast
import torch
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '../Dataset/physionet.org/files/ptb-xl/1.0.3/'
sampling_rate=500

Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

def load_traindata(num_classes):
    Y_norm = Y[Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x)]
    X_norm = load_raw_data(Y_norm, sampling_rate, path)
    #X_norm = X[np.where(Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x))]
    Y_norm_ids = torch.tensor(Y_norm['patient_id'].values[:len(X_norm)], dtype=torch.int32)
    return X_norm[:num_classes], Y_norm_ids[:num_classes]

