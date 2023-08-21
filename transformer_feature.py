import os.path

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Model
from keras.layers import *
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.pylab as pylab
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from tensorflow.keras.metrics import AUC
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import re

import keras_ttrans


# Reset Keras Session
def reset_keras():
    from keras.backend import set_session
    from keras.backend import clear_session
    from keras.backend import get_session
    import gc
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    print(gc.collect())  # if it does something you should see a number as output

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))


def prepare_data():
    df = pd.read_csv('./models/PLR/ASA/ASA_y_label&score.csv')
    pepname = df['pepname']
    label = df['label']

    def seq_dic(fileplace):
        with open(fileplace, mode='r') as file:
            peptides = file.readlines()
            pepdict = {}
            for peptide in peptides:
                peptide = peptide.rstrip().split('\t')
                pepdict[peptide[0]] = peptide[1]
        return pepdict

    pos_dict = seq_dic('pos_functional_Ksuc.txt')
    neg_dict = seq_dic('neg_functional_Ksuc.txt')

    pep_seq = []
    for i, pepID in enumerate(pepname):
        if label[i] == 0:
            pep_seq.append(neg_dict[pepID])
        else:
            pep_seq.append(pos_dict[pepID])

    return np.array(pepname), np.array(pep_seq), label


def prepare_data1(transfer_or_not):
    if transfer_or_not:
        path = './transfer'
    else:
        path = '.'
    df = pd.read_csv(path + '/1D_dataset/transformer_dataset.csv')
    pepname = df['pepname']
    label = df['label']

    def seq_dic(fileplace):
        with open(fileplace, mode='r') as file:
            peptides = file.readlines()
            pepdict = {}
            for peptide in peptides:
                peptide = peptide.rstrip().split('\t')
                pepdict[peptide[0]] = peptide[1]
        return pepdict

    pos_dict = seq_dic(path + '/pos_functional_Ksuc.txt')
    neg_dict = seq_dic(path + '/neg_functional_Ksuc.txt')

    pep_seq = []
    for i, pepID in enumerate(pepname):
        if label[i] == 0:
            pep_seq.append(neg_dict[pepID])
        else:
            pep_seq.append(pos_dict[pepID])

    return np.array(pepname), np.array(pep_seq), label


def prepare_data2():
    df = pd.read_csv('./transfer/1D_dataset_for_experiment/ASA_dataset.csv')
    pepname = df['pepname']

    def seq_dic(fileplace):
        with open(fileplace, mode='r') as file:
            peptides = file.readlines()
            pepdict = {}
            for peptide in peptides:
                peptide = peptide.rstrip().split('\t')
                pepdict[peptide[0]] = peptide[1]
        return pepdict

    pep_dict = seq_dic('./transfer/experiment_sites.txt')

    pep_seq = []
    for i, pepID in enumerate(pepname):
        pep_seq.append(pep_dict[pepID])

    return np.array(pepname), np.array(pep_seq)


def store_code1(peplist, codes, labels, transfer_or_not):
    for i, la in enumerate(labels):
        if transfer_or_not:
            storehouse = './transfer'
        else:
            storehouse = '.'
        if la == 0:
            storehouse = storehouse + '/10features_for_negative_data/transformer/'
        else:
            storehouse = storehouse + '/10features/transformer/'
        if not os.path.exists(storehouse):
            os.makedirs(storehouse)
        with open(storehouse + str(peplist[i]) + r'.transformer', mode='w') as file:
            for co in codes[i]:
                file.write(str(co) + '\t')
            file.write('\n')


def store_code2(peplist, codes):
    for i, pepname in enumerate(peplist):
        storehouse = './transfer/10features_for_experiment/transformer/'
        if not os.path.exists(storehouse):
            os.makedirs(storehouse)
        with open(storehouse + pepname + r'.transformer', mode='w') as file:
            for co in codes[i]:
                file.write(str(co) + '\t')
            file.write('\n')


def turn_to_float64(feature):
    x = np.array(list(feature), dtype=np.float64)
    y = x.tolist()
    return y


def getfeatures1(namelist, labels):
    feature = []
    for i, name in enumerate(namelist):
        if labels[i] == 0:
            fileplace = 'E:/QD065LPSc/Ksuc/transfer/10features_for_negative_data'
        else:
            fileplace = 'E:/QD065LPSc/Ksuc/transfer/10features'
        with open(fileplace + '/transformer/' + name + '.transformer', mode='r') as file:
            fea = file.read().rstrip().split('\t')
            fea = turn_to_float64(fea)
            feature.append(fea)
    return np.array(feature)


def getfeatures2(namelist):
    feature = []
    for i, name in enumerate(namelist):
        fileplace = 'E:/QD065LPSc/Ksuc/transfer/10features_for_experiment'
        with open(fileplace + '/transformer/' + name + '.transformer', mode='r') as file:
            fea = file.read().rstrip().split('\t')
            fea = turn_to_float64(fea)
            feature.append(fea)
    return np.array(feature)


def get_col(feature):
    col = []
    for i in range(0, len(feature)):
        col.append(i)
    return col


# encoding transfer-learning data or pre-training data
#'''
trans_or_not = True
namelist, data, label = prepare_data1(trans_or_not)
data = keras_ttrans.seq2num(data)
count = 5 #np.random.randint(1, 10)
model = load_model(f'./models/transformer/transformer_%s.model' % count)
model.summary()
dropout_2_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
dropout_2_output = dropout_2_model.predict(x=data, batch_size=128)

store_code1(namelist, dropout_2_output, label, trans_or_not)
# '''
# turn transformer feature in to csv for transfer-learning or pre-training data
#'''
trans_or_not = True
namelist, data, label = prepare_data1(trans_or_not)
transformer_feature = getfeatures1(namelist, label)
df_trans = pd.DataFrame(transformer_feature)
df_peps = pd.DataFrame(namelist, columns=['pepname'])
df_labels = df_label = pd.DataFrame(label, columns=['label'])
df_transformer = pd.concat([df_peps, df_trans, df_label], axis=1)
df_transformer.to_csv('./transfer/1D_dataset/transformer_dataset1.csv', index=False)
# '''

# encoding experiment data for predicting
'''
namelist, data = prepare_data2()
data = keras_ttrans.seq2num(data)
count = 1
model = load_model(f'./models/transformer/transformer_%s.model' % count)
model.summary()
dropout_2_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
dropout_2_output = dropout_2_model.predict(x=data, batch_size=128)
store_code2(namelist, dropout_2_output)
# '''

# turn transformer feature in to csv for predicting
'''
namelist, data = prepare_data2()
transformer_feature = getfeatures2(namelist)
df_trans = pd.DataFrame(transformer_feature)
df_peps = pd.DataFrame(namelist, columns=['pepname'])
df_transformer = pd.concat([df_peps, df_trans], axis=1)
df_transformer.to_csv('./transfer/1D_dataset_for_experiment/transformer_dataset.csv', index=False)
# '''
