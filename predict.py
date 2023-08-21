import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import random


def get_col1(fn):
    finame = r'E:/QD065LPSc/Ksuc/transfer/10features_for_experiment/' + str.upper(fn) + r'/E9PTN6_213.' + str.lower(fn)
    with open(finame) as file:
        values = file.read().rstrip().split('\t')
        i = 0
        columns = []
        while i < len(values):
            columns.append(str(i))
            i += 1
    return columns, len(values)


def prep_dataset1(feature_name):
    global dataset
    col, size = get_col1(feature_name)
    x = dataset[col]
    ID = dataset['pepname']
    return x, ID, size


def auc_11(fileplace):
    features = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
    scorelist = []
    for feature in features:
        df = pd.read_csv(fileplace + r'/' + feature + r'_scores.csv')
        scorelist.append(df['score'].values)
    return np.array(scorelist).T


def get_name_and_label():
    df = pd.read_csv('E:/QD065LPSc/Ksuc/transfer/DNN_result_for_experiment/ASA_scores.csv')
    name = df['pepname']
    name = name.values
    return name


feature_list = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
for feature in feature_list:
    model = load_model(f'E:/QD065LPSc/Ksuc/transfer/models/DNN/%s/%s_DNN_%d.h5' % (feature, feature, random.randint(1, 10)))
    dataset = pd.read_csv(f'E:/QD065LPSc/Ksuc/transfer/1D_dataset_for_experiment/%s_dataset.csv' % feature)
    data, peplist, data_size = prep_dataset1(feature)
    print(feature)
    y_score = model.predict(data)
    df_y = pd.concat([pd.DataFrame(list(peplist), columns=['pepname']),
                      pd.DataFrame(list(y_score), columns=['score'])], axis=1)
    path = f'E:/QD065LPSc/Ksuc/transfer/DNN_result_for_experiment/'
    if not os.path.exists(path):
        os.makedirs(path)
    df_y.to_csv(path + f'%s_scores.csv' % feature, index=False)

for count in range(1, 11):
    data = auc_11('E:/QD065LPSc/Ksuc/transfer/DNN_result_for_experiment')
    iDNN = load_model(f'E:/QD065LPSc/Ksuc/transfer/models/integrated_DNN/iDNN_%d.h5' % count)
    #iDNN = load_model(f'E:/QD065LPSc/Ksuc/models/integrated_DNN/iDNN_%d.h5' % count)
    score = iDNN.predict(data)
    peplist = get_name_and_label()
    df_y = pd.concat([pd.DataFrame(list(peplist), columns=['pepname']),
                  pd.DataFrame(list(score), columns=['score'])], axis=1)
    df_y.to_csv(f'E:/QD065LPSc/Ksuc/transfer/iDNN_%d_result_for_experiment.csv' % count, index=False)
