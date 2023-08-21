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


def roc(y_tests, y_test_scores):
    font = {'family': 'arial',
            'weight': 'bold',
            'size': 20}
    params = {'axes.labelsize': '20',
              'xtick.labelsize': '20',
              'ytick.labelsize': '20',
              'lines.linewidth': '4'}
    pylab.rcParams.update(params)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Arial']
    pylab.rcParams['font.weight'] = 'bold'
    plt.figure(figsize=(7, 7), dpi=300)
    AUC = roc_auc_score(y_tests, y_test_scores)
    fpr1, tpr1, thresholds1 = roc_curve(y_tests, y_test_scores)
    plt.plot(fpr1, tpr1, linewidth='3', color='tomato', label='AUC = {:.3f}'.format(AUC))
    plt.plot([0, 1], [0, 1], linewidth='1', color='grey', linestyle="--")
    plt.yticks(np.linspace(0, 1, 6))
    plt.xticks(np.linspace(0, 1, 6))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(prop={'size': 20}, loc=4, frameon=False)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    plt.xlabel('1–Specificity', font)
    plt.ylabel('Sensitivity', font)
    # plt.savefig('F:\SUMO\data_process\models\site_models\site_4fold_trans.jpg')
    #plt.show()


'''多头Attention'''


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


'''Transformer的Encoder部分'''


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.005):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


'''Transformer输入的编码层'''


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def seq2num(seqlist):
    out = []
    transdic = {'A': 8, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 0, 'L': 9, 'M': 10,
                'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '*': 20}
    for seq in seqlist:
        seq = seq.replace('U', '*').replace('X', '*')
        vec = [transdic[i] for i in seq]
        out.append(vec)
    out = np.array(out)
    # out=tf.convert_to_tensor(out)
    return out


def prepare_data():
    df = pd.read_csv('./1D_dataset/ASA_dataset.csv')
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

    pos_dict = seq_dic('./pos_Ksuc.txt')
    neg_dict = seq_dic('./neg_Ksuc.txt')

    pep_seq = []
    for i, pepID in enumerate(pepname):
        if label[i] == 0:
            pep_seq.append(neg_dict[pepID])
        else:
            pep_seq.append(pos_dict[pepID])

    return np.array(pep_seq), label


if __name__ == '__main__':
    '''读取数据'''
    vocab_size = 600  # Only consider the top 20k words

    maxlen = 21
    '''搭建模型'''
    embed_dim = 128  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = transformer_block(x)

    x = layers.GlobalAveragePooling1D()(x)
    O_seq = Dropout(0.05)(x)
    print(O_seq.shape)
    O_seq = Dense(64, activation='selu')(O_seq)

    O_seq = Dense(16, activation='selu')(O_seq)
    O_seq = Dropout(0.1)(O_seq)

    outputs = Dense(2, activation='softmax')(O_seq)

    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    data, labels = prepare_data()
    data = seq2num(data)

    count = 0

    sfolder = StratifiedKFold(n_splits=10, shuffle=False)
    all_loc_pred = []
    all_loc_label = []
    for train, test in sfolder.split(data, labels):
        xtrain_pos = np.zeros((1, maxlen))
        ytrain_pos = []
        count += 1
        x_train, x_test = data[train], data[test]
        y_train, y_test = labels[train], labels[test]

        y_train = np.array(to_categorical(y_train))
        y_test = np.array(to_categorical(y_test))

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy'])

        print('----------------training-----------------------')

        model.fit(x_train, y_train,
                  batch_size=2048,
                  epochs=50,
                  validation_data=(x_test, y_test)
                  )

        print('----------------testing------------------------')
        loss, accuracy = model.evaluate(x_test, y_test)
        print('\n test loss:', loss)
        print('\n test accuracy', accuracy)

        model.save(f'./models/transformer/transformer_%s.model' % count)
        reset_keras()
        model = load_model(f'./models/transformer/transformer_%s.model' % count)

        predicts = list(model.predict(x_test)[:, 1])
        roc(list([list(i)[1] for i in y_test]), predicts)
        all_loc_pred += predicts
        all_loc_label += list([list(i)[1] for i in y_test])
        reset_keras()

    roc(all_loc_label, all_loc_pred)
