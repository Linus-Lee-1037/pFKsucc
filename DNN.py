import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam


# ACF+ASA+AAINDEX+BTA+CKSAAP+GPS+OBC+PSEAAC+PSSM+SS
# 610  61   610   244   411  300 1342   20  1220 183


def get_col(fn):
    finame = r'./10features/' + str.upper(fn) + r'/P53534_3.' + str.lower(fn)
    with open(finame) as file:
        values = file.read().rstrip().split('\t')
        i = 0
        columns = []
        while i < len(values):
            columns.append(str(i))
            i += 1
    return columns, len(values)


def prep_dataset(feature_name):
    global dataset
    col, size = get_col(feature_name)
    x = dataset[col]
    y = dataset['label']
    ID = dataset['pepname']
    return x, y, ID, size


Hidden1_size = {'PSEAAC': 2048, 'CKSAAP': 2048, 'OBC': 64, 'AAINDEX': 1024, 'ACF': 2048, 'GPS': 512, 'PSSM': 1024,
                'ASA': 2048, 'SS': 2048, 'BTA': 2048, 'transformer': 2048}
Hidden2_size = {'PSEAAC': 1024, 'CKSAAP': 1024, 'OBC': 32, 'AAINDEX': 512, 'ACF': 1024, 'GPS': 256, 'PSSM': 512,
                'ASA': 1024, 'SS': 1024, 'BTA': 1024, 'transformer': 1024}


def get_model(size):
    global dropout_rate, feature
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Sequential
    dnn = Sequential()
    dnn.add(Dense(size, input_shape=(size,), bias_initializer='ones', name='Input'))
    dnn.add(Dense(Hidden1_size[feature], activation='relu', name='Hidden1'))
    dnn.add(layers.Dropout(dropout_rate[feature], name='Dropout1'))
    dnn.add(Dense(Hidden2_size[feature], activation='relu', name='Hidden2'))
    dnn.add(layers.Dropout(dropout_rate[feature], name='Dropout2'))
    dnn.add(Dense(1, activation='sigmoid', name='Output'))
    dnn.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['acc'])
    return dnn


def training_DNN(feature_name):
    print('\n')
    print('————————————' + feature_name + '————————————')
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

    x, y, pepID, feature_size = prep_dataset(feature_name)
    x = x.values
    y = y.values
    pepID = pepID.values

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    count = 1
    y_label = []
    y_score = []
    # models = []
    peplist = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dnn = get_model(feature_size)
        dnn.fit(x_train, y_train, epochs=15, batch_size=4096)

        y_label.append(list(y_test))
        y_test_score = dnn.predict(x_test)
        y_score.append(list(y_test_score))
        # models.append(dnn)
        dnn.save(r'./models/DNN/' + feature_name + r'/' + feature_name + r'_DNN_' + str(count) +
                 r'.h5')
        peplist.append(list(pepID[test_index]))

        auc_score = roc_auc_score(y_test, y_test_score)
        print('第', count, '个', feature_name, '模型的AUC分数为：', auc_score)
        count += 1
        reset_keras()

    # for j, model in enumerate(models):  # 保存十个模型
    # model.save(r'./models/DNN/' + str.upper(feature_name) + r'/' + str.upper(feature_name) + r'_DNN_' + str(j) +
    # r'.h5')
    from itertools import chain
    df_y = pd.concat([pd.DataFrame(list(chain.from_iterable(peplist)), columns=['pepname']),
                      pd.DataFrame(list(chain.from_iterable(y_label)), columns=['label']),
                      pd.DataFrame(list(chain.from_iterable(y_score)), columns=['score'])], axis=1)
    df_y.to_csv(r'./models/DNN/' + feature_name + r'/' + feature_name + r'_y_label&score.csv', index=False)

    AUC_score = roc_auc_score(list(chain.from_iterable(y_label)), list(chain.from_iterable(y_score)))
    print('————————————', feature_name, '模型的最终AUC分数为：', AUC_score, '————————————')
    return AUC_score


#features = ['ACF', 'ASA', 'AAINDEX', 'BTA', 'CKSAAP', 'GPS', 'OBC', 'PSEAAC', 'PSSM', 'SS', 'transformer']
features = ['transformer']
feature_scores = []
dropout_rate = {'ACF': 0.1, 'ASA': 0.2, 'AAINDEX': 0.1, 'BTA': 0.2, 'CKSAAP': 0.5, 'GPS': 0.2, 'OBC': 0.3,
                'PSEAAC': 0.5, 'PSSM': 0.2, 'SS': 0.2, 'transformer': 0.5}
for feature in features:
    dataset = pd.read_csv(r'./1D_dataset/' + feature + r'_dataset1.csv')
    feature_scores.append(training_DNN(feature))
