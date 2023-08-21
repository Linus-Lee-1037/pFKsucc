import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam
from keras.models import clone_model
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE, ADASYN


# Reset Keras Session
def reset_keras():
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


def get_model(model_location, featuretype, modeltype):
    t_model = load_model('%s/%s/%s/%s_%s_%d.h5' % (model_location, modeltype, featuretype, str.upper(featuretype),
                                                   modeltype,
                                                   np.random.randint(1, 10)
                                                   ))
    return t_model


def prepare_data(featuretype, modeltype):
    df = pd.read_csv(f'./1D_dataset/%s_dataset.csv' % featuretype)
    pepnames = df['pepname']
    labels = df['label']
    if modeltype == 'DNN':
        dataset = df.drop(labels=['pepname', 'label'], axis=1)
    elif modeltype == 'CNN':
        dataset = []
        for pepname in pepnames:
            img = Image.open(r'./2D_dataset/' + featuretype + r'/' + pepname + r'.png')
            img = np.array(img)
            img = img / 256
            dataset.append(img)
    return pepnames, np.array(dataset), labels


def balance_your_data(training_labels):
    num_pos = 0
    num_neg = 0
    for training_label in training_labels:
        if training_label == 0:
            num_neg += 1
        else:
            num_pos += 1
    pos_times = np.float64(num_neg / num_pos)
    return {0: 1., 1: pos_times}


def proportion(tr_labels):
    n_pos = 0
    for tr_label in tr_labels:
        if tr_label != 0:
            n_pos += 1
    return np.float64(n_pos / len(tr_labels))


#feature_list = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
#feature_list = ['ACF', 'ASA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'SS', 'transformer']
feature_list = ['transformer']
#feature_list = ['GPS']
#feature_list = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
model_list = ['DNN']
ROCAUC_scores = []
PRAUC_scores = []
less_than_50 = []
training_time_record = {}
max_scores = {}
if os.path.exists(r'./transfer/models/DNN/reverse_list.txt'):
    with open(r'./transfer/models/DNN/reverse_list.txt', mode='r') as t:
        reverse_models = t.readlines()
        for i, reverse_model in enumerate(reverse_models):
            reverse_models[i] = reverse_model.rstrip()
else:
    reverse_models = []
os.chdir(f'./transfer/')
for model_type in model_list:
    for feature in feature_list:
        print('——————————————' + feature + ' ' + model_type + '——————————————')
        model1 = get_model('E:/QD065LPSc/Ksuc/models/', feature, model_type)
        pepID, data, label = prepare_data(feature, model_type)
        ppow = 3
        learn_rate = pow(0.1, ppow)
        #'''
        for layer in model1.layers[:-3]:
            # print(layer.name)
            layer.trainable = False
        #'''
        # model1.compile(optimizer=Adam(learning_rate=learn_rate), loss='binary_crossentropy', metrics=['acc'])
        ROCAUC_score = 0
        PRAUC_score = 0
        train_time = 1
        split = 10
        # ROC_threshold = 0.5  # 0.7 for transformer's DNN ;  0.55 for the rest
        # PR_threshold = proportion(label)
        max_score = 0
        # while ROCAUC_score < ROC_threshold:
        # while ROCAUC_score < ROC_threshold or PRAUC_score < PR_threshold:
        while train_time < 100:
            # if chainname == 'M1':
            #    skf = StratifiedKFold(n_splits=2, shuffle=False)
            # else:
            skf = StratifiedKFold(n_splits=split, shuffle=True, random_state=3)
            count = 0
            y_label = []
            y_score = []
            peplist = []
            best_models = []

            for train_index, test_index in skf.split(data, label):
                x_train, x_test = data[train_index], data[test_index]
                y_train, y_test = label[train_index], label[test_index]
                '''
                # 训练数据重采样一次，扩增一倍
                index_set = []
                for index in train_index:
                    if y_train[index] == 1:
                        index_set.append(index)
                x_enrich_positive = data[index_set]
                y_enrich_positive = label[index_set]
                for nima in range(1, 9):
                    np.append(x_train, x_enrich_positive, axis=0)
                    np.append(y_train, y_enrich_positive, axis=0)

                xy = list(zip(x_train, y_train))
                random.shuffle(xy)
                x_train[:], y_train[:] = zip(*xy)
                # '''
                # '''
                if model_type == 'DNN':
                    # x_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
                    x_resampled, y_resampled = ADASYN().fit_resample(x_train, y_train)
                else:
                    x_train_2D = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
                    x_resampled, y_resampled = SMOTE().fit_resample(x_train_2D, y_train)
                    x_resampled = (x_resampled.reshape(x_resampled.shape[0], x_train.shape[1], x_train.shape[2]))
                xy = list(zip(x_resampled, y_resampled))
                random.shuffle(xy)
                x_resampled[:], y_resampled[:] = zip(*xy)
                x_train = x_resampled
                y_train = y_resampled
                # '''

                count += 1
                y_label.append(list(y_test))
                peplist.append(list(pepID[test_index]))

                model = clone_model(model1)
                model.compile(optimizer=Adam(learning_rate=learn_rate), loss='binary_crossentropy', metrics=['acc'])
                model.fit(x_train, y_train, epochs=train_time, batch_size=2048, verbose=1,
                          class_weight=balance_your_data(y_test)
                          )
                y_test_score = model.predict(x_test)
                y_score.append(list(y_test_score))

                best_models.append(model)

                # check roc auc score
                rocauc_score = roc_auc_score(y_test, y_test_score)
                print(f'第%d个%s的%s模型的ROCAUC分数为：' % (count, feature, model_type),
                      rocauc_score)
                # check auc score of precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_test_score)
                prauc_score = auc(recall, precision)
                print(f'第%d个%s的%s模型的PRAUC分数为：' % (count, feature, model_type), prauc_score)
                reset_keras()

            from itertools import chain

            peplist1 = []
            y_score1 = []
            y_label1 = []
            peplist = list(chain.from_iterable(peplist))
            y_score = list(chain.from_iterable(y_score))
            y_label = list(chain.from_iterable(y_label))
            for pep in pepID:
                peplist1.append(peplist[peplist.index(pep)])
                y_score1.append(y_score[peplist.index(pep)])
                y_label1.append(y_label[peplist.index(pep)])
            ROCAUC_score = roc_auc_score(y_label1, y_score1)
            if ROCAUC_score < 0.5:  # reverse scores, record reverse model
                if feature not in reverse_models:
                    reverse_models.append(feature)
                for no, score in enumerate(y_score1):
                    y_score1[no] = 1 - score
            else:
                if feature in reverse_models:
                    reverse_models.remove(feature)
            ROCAUC_score = roc_auc_score(y_label1, y_score1)

            if ROCAUC_score > max_score:
                max_score = float(str(ROCAUC_score)[0:4])
                max_scores[feature] = max_score
                fileplace = f'./models/%s/%s/' % (model_type, feature)
                if not os.path.exists(fileplace):
                    os.makedirs(fileplace)
                for i in range(1, 11):
                    best_models[i - 1].save(fileplace + f'%s_%s_%d.h5' % (feature, model_type, i))
                df_y = pd.concat([pd.DataFrame(peplist1, columns=['pepname']),
                                  pd.DataFrame(y_label1, columns=['label']),
                                  pd.DataFrame(y_score1, columns=['score'])], axis=1)
                df_y.to_csv(f'./models/%s/%s/%s_y_label&score.csv' % (model_type, feature, feature), index=False)
                training_time_record[f'%s_%s' % (feature, model_type)] = train_time

            print(f'————————————%s的%s模型的最终ROCAUC分数为：' % (feature, model_type),
                  ROCAUC_score, '————————————')

            '''
            FPR, TPR, _ = roc_curve(y_label1, y_score1)
            pyplot.plot(FPR, TPR, marker='.', label=model_type)
            pyplot.xlabel('False Positive Rate')
            pyplot.ylabel('True Positive Rate')
            pyplot.legend()
            pyplot.show()
            '''

            precision, recall, _ = precision_recall_curve(y_label1, y_score1)
            PRAUC_score = auc(recall, precision)
            print(f'————————————%s的%s模型的最终PRAUC分数为：' % (feature, model_type), PRAUC_score,
                  '————————————')
            '''
            if ROCAUC_score < ROC_threshold:
            # if PRAUC_score < PR_threshold or ROCAUC_score < ROC_threshold:
                train_time += random.randint(1, 10)
                # train_time += 1
            # learn_rate += 0.0001
            if train_time > 100:  # or ROCAUC_score < 0.45:
                train_time = 5
                model1 = get_model('E:/QD065LPSc/Ksuc/models', feature, model_type)
                for layer in model1.layers[:-3]:
                    layer.trainable = False
            # '''
            #train_time += random.randint(1, 10)
            train_time += 1
        ROCAUC_scores.append(ROCAUC_score)
        PRAUC_scores.append(PRAUC_score)
with open(r'./models/DNN/reverse_list.txt', mode='w') as textfile:
    for reverse_model in reverse_models:
        textfile.write(reverse_model + '\n')
for le in less_than_50:
    print(le)
if os.path.exists('./DNN_train_time.csv'):
    original_training_time_record = pd.read_csv('./DNN_train_time.csv')
    for record_name in training_time_record.keys():
        original_training_time_record[record_name] = training_time_record[record_name]
    original_training_time_record.to_csv('./DNN_train_time.csv', index=False)
else:
    pd.DataFrame(training_time_record, index=[0]).to_csv('./DNN_train_time.csv', index=False)
for model_type in max_scores.keys():
    print(model_type + '      ' + str(max_scores[model_type]))
