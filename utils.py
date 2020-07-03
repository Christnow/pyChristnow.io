import os
import sys
import json
import shutil
import pickle
import warnings

from .tokenizer import Tokenizer
from .tokenzier import pad_sequences
from .tokenizer import text_to_word_sequence

import numpy as np


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


def split_data(df, test_size=0.1):
    train, test = pd.DataFrame(), pd.DataFrame()
    for label, group in df.groupby(['label']):
        trainx, testx, trainy, testy = train_test_split(group['text'],
                                                        group['label'],
                                                        test_size=test_size)
        traind = pd.concat([trainx, trainy], axis=1)
        train = pd.concat([train, traind])
        testd = pd.concat([testx, testy], axis=1)
        test = pd.concat([test, testd])
    train = shuffle(train)
    test = shuffle(test)
    return train, test


def create_tokenizer(alldata=None, tokenizer_config=None):
    if os.path.isfile(tokenizer_config):
        #由配置文件创建一个Tokenizer对象
        tokenconfig = read_json(tokenizer_config)
        tokenizer = Tokenizer.tokenizer_from_json(tokenconfig)
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
    # 将超过固定值的部分截掉，不足的在最前面用0填充
    data_ids = tokenizer.texts_to_sequences(data)
    data_seq = pad_sequences(data_ids,
                             maxlen=max_sequence_length,
                             padding='post')
    return data_seq


def label_to_onehot(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical