import re
import os
import sys
import time
import json
import pickle
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")

import re
import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd


import matplotlib as mpl
if platform.platform().lower().startswith('linux'):
    mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from .selfattention import SelfAttention




class WDnet(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  filter_sizes = [3, 4, 5, 6] #卷积核大小
    #  num_filters = 200  # Number of filters per filter size 卷价个数
    #  num_rnn_layers = 2 #lstm层数
    #  rnn_units = 64 # lstm维度
    #  att_units=50 # selfattention维度
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        filter_sizes=[2, 3, 4, 5],
        num_filters=200,
        num_rnn_layers=1,
        rnn_units=64,
        att_units=50,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.filters_sizes=filter_sizes
        self.num_filters=num_filters
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.att_units = att_units
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def build(self):
        inputx = tf.keras.layers.Input(
            shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim,
                                          weights=[self.embedding_matrix
                                                   ])(inputx)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim)(inputx)
        ###  卷积池化层
        convs = []
        for sizes in self.filter_sizes:
            conv = tf.keras.layers.Conv1D(self.num_filters,
                                          sizes,
                                          padding='same',
                                          strides=1,
                                          activation='relu')(x)
            conv=tf.keras.layers.MaxPooling1D(
                padding='same',
                pool_size=32,)(conv)
            convs.append(conv)
        ## 拼接上述卷积池化层的输出
        convs = tf.keras.layers.Concatenate(axis=1)(convs)

        ###  BILSTM层由num_rnn_layers取定是单层还是多层叠加
        for nrl in range(self.num_rnn_layers):
            att = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=self.rnn_units,
                                     return_sequences=True,
                                     activation='relu'))(x)
            att = tf.keras.layers.Dropout(self.dropout_keep_prob)(att)
        att = SelfAttention(units=self.att_units,
                          attention_activation='tanh',
                          attention_type=SelfAttention.ATTENTION_TYPE_MUL,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          bias_regularizer=tf.keras.regularizers.l1(1e-4),
                          attention_regularizer_weight=1e-4,
                          name='SelfAttention')(att)
        att = tf.keras.layers.Dropout(self.dropout_keep_prob)(att)
        ## 拉直维度降低
        att = tf.keras.layers.Flatten()(att)

        concat=tf.keras.layers.concatenate([convs,att])

        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(concat)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model



class TextBILSTMattention(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  num_rnn_layers = 2 #lstm层数
    #  rnn_units = 64 # lstm维度
    #  att_units=50 # selfattention维度
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        num_rnn_layers=1,
        rnn_units=64,
        att_units=50,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.att_units = att_units
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def build(self):
        inputx = tf.keras.layers.Input(
            shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim,
                                          weights=[self.embedding_matrix
                                                   ])(inputx)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim)(inputx)

        ###  BILSTM层由num_rnn_layers取定是单层还是多层叠加
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=self.rnn_units,
                                     return_sequences=True,
                                     activation='relu'))(x)
            x = tf.keras.layers.Dropout(self.dropout_keep_prob)(x)
        x = SelfAttention(units=self.att_units,
                          attention_activation='tanh',
                          attention_type=SelfAttention.ATTENTION_TYPE_MUL,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                          bias_regularizer=tf.keras.regularizers.l1(1e-4),
                          attention_regularizer_weight=1e-4,
                          name='SelfAttention')(x)
        x = tf.keras.layers.Dropout(self.dropout_keep_prob)(x)
        ## 拉直维度降低
        x = tf.keras.layers.Flatten()(x)
        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model


class TextCRNN(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  num_rnn_layers = 2 #lstm层数
    #  rnn_units = 64 # lstm维度
    #  filter_sizes = [3, 4, 5, 6] #卷积核大小
    #  num_filters = 200  # Number of filters per filter size 卷价个数
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        num_rnn_layers=1,
        rnn_units=64,
        filter_sizes=[2, 3, 4, 5],
        num_filters=200,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def create_model(self):
        inputx = tf.keras.layers.Input(
            shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim,
                                          weights=[self.embedding_matrix
                                                   ])(inputx)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim)(inputx)
        ###  卷积池化层
        convs = []
        for sizes in self.filter_sizes:
            conv = tf.keras.layers.Conv1D(self.num_filters,
                                          sizes,
                                          padding='same',
                                          strides=1,
                                          activation='relu')(embedx)
            convs.append(conv)

        ## 拼接上述卷积池化层的输出
        x = tf.keras.layers.Concatenate(axis=1)(convs)
        ###  BILSTM层由num_rnn_layers取定是单层还是多层叠加
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=self.rnn_units,
                                     return_sequences=True,
                                     activation='relu'))(x)
            x = tf.keras.layers.Dropout(self.dropout_keep_prob)(x)
        ## 拉直维度降低
        x = tf.keras.layers.Flatten()(x)
        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model


class TextRCNN(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  num_rnn_layers = 2 #lstm层数
    #  rnn_units = 64 # lstm维度
    #  filter_sizes = [3, 4, 5, 6] #卷积核大小
    #  num_filters = 200  # Number of filters per filter size 卷价个数
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        num_rnn_layers=1,
        rnn_units=64,
        filter_sizes=[2, 3, 4, 5],
        num_filters=200,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def create_model(self):
        inputx = tf.keras.layers.Input(
            shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim,
                                          weights=[self.embedding_matrix
                                                   ])(inputx)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                          self.embedding_dim)(inputx)
        ## 反向lstm
        x_backwords=tf.keras.layers.LSTM(
            units=self.rnn_units,
            return_sequences=True,
            go_backwards=True,
        )(x)
        x_backwords=tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.reverse(x,axes=1)
        )(x_backwords)
        ## 前向lstm
        x_fordwords=tf.keras.layers.LSTM(
            units=self.rnn_units,
            return_sequences=True,
            go_backwards=False,
        )
        x_feb=tf.keras.layers.Concatenate(axis=2)(
            [x_forwords,x,x_backwords]
        )
        dim_2 = tf.keras.backend.int_shape(x_feb)[2]
        x=tf.keras.layers.Reshape((dim_2,self.max_sentence_length))(x_feb)

        ## 卷积池化层
        convs=[]
        for sizes in self.filter_sizes:
            conv=tf.keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=sizes,
                padding='same',
                activation='relu'
            )(x)
            conv=tf.keras.layers.MaxPooling1D(
                padding='same',
                pool_size=32,
            )(conv)
            convs.append(conv)
        x=tf.keras.layers.Concatenate(axis=1)(convs)
        x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dropout(self.dropout_keep_prob)(x)
        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model


class TextBILSTM(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  num_rnn_layers = 2 #lstm层数
    #  rnn_units = 64 # lstm维度
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        num_rnn_layers=1,
        rnn_units=64,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def build(self):
        inputx = tf.keras.layers.Input(shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            x = tf.keras.layers.Embedding(self.vocab_size,
                                               self.embedding_dim,
                                               weights=[self.embedding_matrix
                                                        ])(inputx)
        else:
            x = tf.keras.layers.Embedding(self.vocab_size,
                                               self.embedding_dim)(inputx)

        ###  BILSTM层由num_rnn_layers取定是单层还是多层叠加
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=self.rnn_units,
                    return_sequences=True,
                    activation='relu'))(x)
            x = tf.keras.layers.Dropout(self.dropout_keep_prob)(x)
        ## 拉直维度降低
        x = tf.keras.layers.Flatten()(x)
        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model


class TextCNN(object):
    '''
    #  max_sentence_length = 300 # 最大句子长度，也就是说文本样本中字词的最大长度，不足补零，多余的截断
    #  embedding_matrix=np.array() #预训练词向量
    #  embedding_dim = 128 #词向量长度，即每个字词的维度
    #  filter_sizes = [3, 4, 5, 6] #卷积核大小
    #  num_filters = 200  # Number of filters per filter size 卷价个数
    #  dropout_keep_prob = 0.5
    #  lr=0.001 # 学习率
    '''
    def __init__(
        self,
        max_sentence_length=100,
        vocab_size=10000,
        embedding_dim=300,
        embedding_matrix=None,
        filter_sizes=[2, 3, 4, 5],
        num_filters=200,
        dropout_keep_prob=0.5,
        num_classes=2,
    ):
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes

    def build(self):
        inputx = tf.keras.layers.Input(
            shape=(self.max_sentence_length, ))  ##设定输入层

        ## 本身每一个embedding都熟产出一个词向量，所以实际不需要加载
        ## 如果非要加载，这里提供两种方法，一个加载，一个不加载
        if self.embedding_matrix.any():
            embedx = tf.keras.layers.Embedding(self.vocab_size,
                                               self.embedding_dim,
                                               weights=[self.embedding_matrix
                                                        ])(inputx)
        else:
            embedx = tf.keras.layers.Embedding(self.vocab_size,
                                               self.embedding_dim)(inputx)

        ###  卷积池化层
        convs = []
        for sizes in self.filter_sizes:
            conv = tf.keras.layers.Conv1D(self.num_filters,
                                          sizes,
                                          padding='same',
                                          strides=1,
                                          activation='relu')(embedx)
            conv = tf.keras.layers.MaxPooling1D(pool_size=3,
                                                padding='same')(conv)
            convs.append(conv)

        ## 拼接上述卷积池化层的输出
        convs = tf.keras.layers.Concatenate(axis=1)(convs)
        ## 拉直维度降低
        flat = tf.keras.layers.Flatten()(convs)
        ##失活层
        drop = tf.keras.layers.Dropout(self.dropout_keep_prob)(flat)
        ##分类层
        output = tf.keras.layers.Dense(self.num_classes,
                                       activation='softmax')(drop)
        model = tf.keras.models.Model(inputs=inputx, outputs=output)
        return model


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)


def remove_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def save_json(file, json_dict):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_dict, indent=4, ensure_ascii=False))


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    return json_dict


def save_model(model, modelfile):
    with open(modelfile, 'wb') as f:
        pickle.dump(model, f)


def read_model(modelfile):
    with open(modelfile, 'rb') as f:
        model = pickle.load(f)
    return model


def load_stopwords(file='./哈工大停用词表.txt'):
    with open(file, 'r', encoding='utf-8') as f:
        sp = [x.strip() for x in f if x.strip()]
    return sp


def split_data(datas_k, labels_k, test_size=0.1):
    train_kx, test_kx, train_ky, test_ky = train_test_split(
        datas_k, labels_k, test_size=test_size)
    return train_kx, test_kx, train_ky, test_ky


def create_tokenizer(alldata=None, tokenizer_config=None):
    if os.path.isfile(tokenizer_config):
        #由配置文件创建一个Tokenizer对象
        tokenconfig = read_json(tokenizer_config)
        tokenizer = tokenizer_from_json(tokenconfig)
    else:
        #创建一个Tokenizer对象
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(alldata)
        tokenconfig = tokenizer.to_json()
        save_json(tokenizer_config, tokenconfig)
    vocab = tokenizer.word_index  #得到每个词的编号
    return vocab, tokenizer


def data_to_seq(tokenizer, data, max_sequence_length):
    #序列模式
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    # 将超过固定值的部分截掉，不足的用0填充
    data_ids = tokenizer.texts_to_sequences(data)
    data_seq = pad_sequences(data_ids,
                             maxlen=max_sequence_length,
                             padding='post')
    return data_seq


def label_to_onehot(label):
    # [0,1] --> [[0,1],[1,0]]
    label = to_categorical(label)
    return label


stopwords = load_stopwords('./哈工大停用词表.txt')


def segment(x):
    ## 正则去除非中文
    x = re.sub('[^\u4e00-\u9fa5]+', '', x)
    ##jiab分词，去除停用词，且保留长度大于2的词
    x = [
        x.replace(' ', '') for x in jieba.cut(x)
        if x not in stopwords and len(x) >= 2
    ]
    return x

def psegment(x,tags=[]):
    ## 正则去除非中文
    x = re.sub('[^\u4e00-\u9fa5]+', '', x)
    ##jiab分词，去除停用词，且保留长度大于2的词
    x = [
        [word.replace(' ', ''),flag] for word,flag in pseg.cut(x)
        if word not in stopwords and len(word) >= 2 and flag in tags
    ]
    return x


def load_data(file='./数据/附件2.xlsx', eda=False):
    df = pd.read_excel(file)[['留言详情', '一级标签']]  ##读取数据
    df.columns = ['text', 'label']  ##设定使用列名
    df['text'] = df['text'].apply(lambda x: x.strip())  ##去除文本头尾字符
    df['text'] = df['text'].apply(lambda x: ' '.join(segment(x)))  ###正则分词去停用词
    label = {y: x
             for x, y in enumerate(set(df['label'].values.tolist()))}  ###获取标签名
    save_json('./output/label.json', label)  ##保存标签
    df['label'] = df['label'].apply(lambda x: label.get(x))  ###转化标签本文为id
    df.to_excel('./output/train.xlsx', index=False)  ##保存分完词的数据

    ### 作图：文本长度柱状图，这里可以设定max_sequences_length值得依据
    ### 原则就是这个长度覆盖所有文本长度的80%
    fig = plt.figure()
    df['text'].apply(lambda x: len(x.split())).hist()
    plt.show()
    plt.savefig('./output/data_hist.png')
    return df, label


def train():
    #### 初始化网络参数
    epochs = 2  ###训练轮数
    batch_size = 64  ###batch尺寸
    max_sentence_length = 500  ##输入文本最大长度
    embedding_dim = 300  ### 词嵌入层大小
    filter_sizes = [2, 3, 4, 5]  ###卷积层卷积核大小
    num_filters = 200  ###卷积层过滤器输出维度
    dropout_keep_prob = 0.5  ###失活率
    lr = 0.001  ###学习率

    ### 读取数据
    df, label = load_data(file='./数据/附件2.xlsx')
    print(df)
    num_classes = len(label)  ###分类类别数目
    label_name = list(label.keys())  ##标签名称

    ### 建立token 词语id特征转化
    tokenizer_config = './output/tokenizer_config.json'
    vocab, tokenizer = create_tokenizer(df['text'], tokenizer_config)
    vocab_size = len(vocab) + 1

    ### 加载word2vec词向量
    vec = load_word2vec(df, './output/word2vec.txt', embedding_dim=300)
    embedding_matrix = []
    # 首先加id为0的初始词向量,一次加载每个词的词向量
    embedding_matrix.append(np.zeros(embedding_dim))
    for word in vocab:
        try:
            embedding_matrix.append(vec.wv[word])
        except:
            embedding_matrix.append(np.zeros(embedding_dim))
    embedding_matrix = np.array(embedding_matrix)

    ### 按照类别分组，从每组抽取9:1训练测试集，再整合
    train, test = pd.DataFrame(), pd.DataFrame()
    for name, group in df.groupby(['label']):
        trainx, testx, trainy, testy = split_data(group['text'],
                                                  group['label'],
                                                  test_size=0.1)
        traind = pd.concat([trainx, trainy], axis=1)
        testd = pd.concat([testx, testy], axis=1)
        train = pd.concat([train, traind])
        test = pd.concat([test, testd])
    train.columns = ['text', 'label']
    test.columns = ['text', 'label']
    train.to_excel('./output/traindata.xlsx', index=False)
    test.to_excel('./output/testdata.xlsx', index=False)
    ###转化词语特征
    trainx = data_to_seq(tokenizer,
                         train['text'],
                         max_sequence_length=max_sentence_length)
    testx = data_to_seq(tokenizer,
                        test['text'],
                        max_sequence_length=max_sentence_length)
    ###转化标签为onehot
    trainy = label_to_onehot(train['label'])
    testy = label_to_onehot(test['label'])

    ###初始化模型
    model = TextCNN(max_sentence_length, vocab_size, embedding_dim,
                    embedding_matrix, filter_sizes, num_filters,
                    dropout_keep_prob, num_classes)
    model = model.build()
    ## 模型作图
    model.summary()
    ##定义模型训练优化器
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ###训练模型
    model.fit(trainx,
              trainy,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(testx, testy),
              shuffle=True)
    ###保存模型
    model.save('./output/textcnn_model.h5')


def evaluate():
    from sklearn.metrics import classification_report
    max_sentence_length = 500
    test = pd.read_excel('./output/testdata.xlsx')
    modelfile = './output/textcnn_model.h5'
    model = tf.keras.models.load_model(modelfile)
    tokenizer_config = './output/tokenizer_config.json'
    vocab, tokenizer = create_tokenizer(tokenizer_config=tokenizer_config)
    testx = data_to_seq(tokenizer,
                        test['text'],
                        max_sequence_length=max_sentence_length)
    result = [np.argmax(x) for x in model.predict(testx)]
    testy = test['label'].values.tolist()
    label_name = read_json('./output/label.json').keys()
    output_dict = False
    score = classification_report(y_true=testy,
                                  y_pred=result,
                                  target_names=label_name,
                                  output_dict=output_dict)
    print(score)


def predict(model,text):
    tokenizer_config = './output/tokenizer_config.json'
    vocab, tokenizer = create_tokenizer(tokenizer_config=tokenizer_config)
    text = [x for x in jieba.cut(text) if len(x) >= 2]
    text = data_to_seq(tokenizer, text, 500)
    result = model.predict(text)
    print(result)
    print(np.argmax(result[0]))


if __name__ == '__main__':
    model = TextCNN(100, 1000, 100, [2, 3, 4], 200, 0.5, 3)
    model = model.build()
    model.summary()
    train()
    evaluate()
