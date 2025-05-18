import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from collections import Counter
def load_raw_data(df, path):
    data_100 = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    #data_500 = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data_100 = np.array([signal for signal, meta in data_100])
    #print(len(data_100[0]),len(data_100[0][0]))
    #data_500 = np.array([signal for signal, meta in data_500])
    #print(len(data_500[0]),len(data_500[0][0]))
    #data = np.concatenate((data_100, data_500), axis=1)
    return data_100

#path = '../Dataset/physionet.org/files/ptb-xl/1.0.3/'
path = '../Dataset/'
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
    Y_norm = Y_norm.drop_duplicates(subset='patient_id', keep='first') #remove duplicates in patient_id
    X = load_raw_data(Y_norm[:num_classes], path)
    #X_norm = X[np.where(Y['diagnostic_superclass'].apply(lambda x: 'NORM' in x))]
    X_norm = X
    Y_norm_ids = torch.tensor(Y_norm['patient_id'].values[:len(X_norm)], dtype=torch.int32)
    return X_norm[:num_classes], Y_norm_ids[:num_classes]

