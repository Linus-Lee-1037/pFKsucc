import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import random
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from imblearn.over_sampling import SMOTE


def auc_11(fileplace):
    models = ['DNN']
    features = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
    scorelist = []
    for mod in models:
        for feature in features:
            df = pd.read_csv(fileplace + r'/' + mod + r'/' + feature + r'/' + feature + r'_y_label&score.csv')
            scorelist.append(df['score'].values)
    return np.array(scorelist).T


def get_name_and_label():
    df = pd.read_csv('./models/DNN/ASA/ASA_y_label&score.csv')
    name = df['pepname']
    name = name.values
    label = df['label']
    label = label.values
    return name, label


def get_model(size):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Sequential
    dnn = Sequential()
    dnn.add(Dense(11, input_shape=(size,), bias_initializer='ones', name='Input'))
    dnn.add(Dense(1024, activation='relu', name='Hidden1'))
    dnn.add(layers.Dropout(0.5, name='Dropout1'))
    dnn.add(Dense(512, activation='relu', name='Hidden2'))
    dnn.add(layers.Dropout(0.5, name='Dropout2'))
    dnn.add(Dense(1, activation='sigmoid', name='Output'))
    dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return dnn


def training_DNN(pepID, x, y):
    print('\n')
    print('———————————— Training integrated DNN model ————————————')
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from keras.backend import set_session
    from keras.backend import clear_session
    from keras.backend import get_session
    import gc

    # Reset Keras Session
    def reset_keras():
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()

        try:
            del dnn  # this is from global space - change this as you need
        except:
            pass

        print(gc.collect())  # if it does something you should see a number as output

        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.compat.v1.Session(config=config))

    feature_size = x[0].shape[0]

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    count = 1
    y_label = []
    y_score = []
    # models = []
    peplist = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dnn = get_model(feature_size)
        dnn.fit(x_train, y_train, epochs=1, batch_size=4096)

        y_label.append(list(y_test))
        y_test_score = dnn.predict(x_test)
        y_score.append(list(y_test_score))

        dnn.save(r'./models/integrated_DNN/' + r'iDNN_' + str(count) + r'.h5')
        peplist.append(list(pepID[test_index]))

        auc_score = roc_auc_score(y_test, y_test_score)
        print('第', count, '个模型的AUC分数为：', auc_score)
        count += 1
        reset_keras()

    from itertools import chain
    df_y = pd.concat([pd.DataFrame(list(chain.from_iterable(peplist)), columns=['pepname']),
                      pd.DataFrame(list(chain.from_iterable(y_label)), columns=['label']),
                      pd.DataFrame(list(chain.from_iterable(y_score)), columns=['score'])], axis=1)
    df_y.to_csv(r'./models/integrated_DNN/' + r'iDNN_y_label&score.csv', index=False)

    AUC_score = roc_auc_score(list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score)))
    print('———————————— 模型的最终AUC分数为：', AUC_score, ' ————————————')
    return AUC_score


def trans_training_DNN(pepID, x, y):
    print('\n')
    print('———————————— Training small sample integrated DNN model ————————————')
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from keras.backend import set_session
    from keras.backend import clear_session
    from keras.backend import get_session
    import gc

    # Reset Keras Session
    def reset_keras():
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()

        try:
            del dnn  # this is from global space - change this as you need
        except:
            pass

        print(gc.collect())  # if it does something you should see a number as output

        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.compat.v1.Session(config=config))

    feature_size = x[0].shape[0]

    skf = StratifiedKFold(n_splits=10, shuffle=True)
    y_label = []
    y_score = []
    # models = []
    peplist = []
    model_count = 1
    AUC_scores = [0]

    while model_count < 11:
        print(f'——————————训练第%s个integrated_DNN模型——————————' % model_count)
        dnn = load_model(f'./models/integrated_DNN/iDNN_%d.h5' % model_count)
        count = 1
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #'''
            x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
            xy = list(zip(x_resampled, y_resampled))
            random.shuffle(xy)
            x_resampled[:], y_resampled[:] = zip(*xy)
            x_train = x_resampled
            y_train = y_resampled
            #'''
            dnn.fit(x_train, y_train, epochs=5, batch_size=2048)

            y_label.append(list(y_test))
            y_test_score = dnn.predict(x_test)
            y_score.append(list(y_test_score))
            peplist.append(list(pepID[test_index]))

            auc_score = roc_auc_score(y_test, y_test_score)
            print('第', count, '个模型的AUC分数为：', auc_score)

            if auc_score >= max(AUC_scores):
                dnn.save(r'./transfer/models/integrated_DNN/' + r'iDNN_' + str(count) + r'.h5')
                from itertools import chain
                df_y = pd.concat([pd.DataFrame(list(chain.from_iterable(peplist)), columns=['pepname']),
                                  pd.DataFrame(list(chain.from_iterable(y_label)), columns=['label']),
                                  pd.DataFrame(list(chain.from_iterable(y_score)), columns=['score'])], axis=1)
                df_y.to_csv(r'./transfer/models/integrated_DNN/' + r'iDNN_y_label&score.csv', index=False)
                AUC_scores.append(roc_auc_score(list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score))))
            reset_keras()
            count += 1
        model_count += 1
        print('———————————— 模型的最终AUC分数为：', AUC_scores[-1], ' ————————————')
    return max(AUC_scores)


# pre-training
#'''
dataset = auc_11('./models')
peplist, labels = get_name_and_label()
DNN_score = training_DNN(peplist, dataset, labels)
#'''

# trans-training
'''
dataset = auc_11('./transfer/models')
peplist, labels = get_name_and_label()
DNN_score = trans_training_DNN(peplist, dataset, labels)
print('最高分数为：' + str(DNN_score))
# '''

# integrate the scores from transfer learned DNNs and then predict to give the final scores
'''
highest_AUC_scores = {}

count = 0
auc_score = 0
AUCs = []
model_s = []
while count < 10:
    count += 1
    model = load_model(f'./transfer/models/integrated_DNN/iDNN_%d.h5' % count)
    #model = load_model(f'./models/integrated_DNN/iDNN_%d.h5' % count)
    model_s.append(model)
    dataset = auc_11('./transfer/models')
    peplist, labels = get_name_and_label()
    y_score = model.predict(dataset)
    auc_score = roc_auc_score(labels, y_score)
    AUCs.append(auc_score)
    df_y = pd.concat([pd.DataFrame(list(peplist), columns=['pepname']),
                        pd.DataFrame(list(labels), columns=['label']),
                        pd.DataFrame(list(y_score), columns=['score'])], axis=1)
    if not os.path.exists(r'./transfer/models/integrated_DNN'):
        os.makedirs(r'./transfer/models/integrated_DNN')

    # df_y.to_csv(f'./transfer/models/integrated_DNN/iDNN_%d_y_label&score.csv' % count, index=False)
    if auc_score >= max(AUCs):
        df_y.to_csv(f'./transfer/models/integrated_DNN/iDNN_y_label&score.csv', index=False)

print(f'%s模型的auc分数为：' % 'iDNN', max(AUCs))
print(f'分最高的模型是第%d个' % AUCs.index(max(AUCs)))
highest_AUC_scores['function_Ksuc'] = max(AUCs)
model_s[AUCs.index(max(AUCs))].save(f'./transfer/models/integrated_DNN/iDNN_%d.h5' % AUCs.index(max(AUCs)))


# collect the auc scores of 11 DNNs and integrated_DNN's results
# '''
df_iDNN = pd.read_csv('./models/integrated_DNN/iDNN_y_label&score.csv')
label = df_iDNN['label']
score = df_iDNN['score']
df_scores = []
feature_list = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
modelname = 'DNN'
for featurename in feature_list:
    df = pd.read_csv(f'./models/%s/%s/%s_y_label&score.csv' % (modelname, featurename, featurename))
    label_array = df['label']
    score_array = df['score']
    auc_score_array = roc_auc_score(list(label_array), list(score_array))
    #precision, recall, _ = precision_recall_curve(label_array, score_array)
    #auc_score_array = auc(recall, precision)
    df_scores.append(auc_score_array)
    print(f'%s特征的%s模型的rocauc分数为：%s' % (featurename, modelname, str(auc_score_array)))
df_scores.append(roc_auc_score(list(label), list(score)))
pd.DataFrame(df_scores).to_csv('./scores.csv', index=False)
# '''

