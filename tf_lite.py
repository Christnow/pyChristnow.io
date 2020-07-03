
import os
import sys
import time

import numpy as np
from sklearn import svm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D


class TextCNNa(object):
    def __init__(self, hparam):
        self.hparam = hparam

    def build_layer(self, inputx, param, input_length):

        filters = param.get('filters')
        filters_num = param.get('filters_num')
        label_number = param.get('label_number')
        dropout_spatial = param.get('dropout')
        vocab_size = param.get('vocab')
        embedding_size = param.get('embedding_size')
        dense_name = param.get('dense_name')
        embeddding = Embedding(vocab_size,
                               embedding_size,
                               input_length=input_length,
                               trainable=True)(inputx)
        # CNN
        convs = []
        for kernel_size in filters:
            conv = Conv1D(filters_num,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='SAME',
                          activation='relu')(embeddding)
            pooled = MaxPooling1D(
                padding='valid',
                pool_size=32,
            )(conv)
            convs.append(pooled)
        x = Concatenate(axis=1)(convs)
        x = Flatten()(x)
        x = Dropout(dropout_spatial)(x)
        dense = Dense(label_number, activation='softmax', name=dense_name)(x)
        return dense

    def build_model(self):

        input_length = self.hparam.get('max_len')
        filters = self.hparam.get('filters')
        filters_num = self.hparam.get('filters_num')
        label_number = self.hparam.get('label_number')
        dropout_spatial = self.hparam.get('dropout')
        vocab_size = self.hparam.get('vocab')
        embedding_size = self.hparam.get('embedding_size')
        dense_name = self.hparam.get('dense_name')
        inputx = Input(shape=(input_length, ), dtype='int32', name='Input')
        embeddding = Embedding(vocab_size,
                               embedding_size,
                               input_length=input_length,
                               trainable=True)(inputx)
        # CNN
        convs = []
        for kernel_size in filters:
            conv = Conv1D(filters_num,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='SAME',
                          activation='relu')(embeddding)
            pooled = MaxPooling1D(
                padding='valid',
                pool_size=32,
            )(conv)
            convs.append(pooled)
        x = Concatenate(axis=1)(convs)
        x = Flatten()(x)
        x = Dropout(dropout_spatial)(x)
        dense = Dense(label_number, activation='softmax', name=dense_name)(x)
        output = [dense]
        model = Model(inputx, output)
        return model

### 初始化textcnn各种参数
from script import create_tokenizer
hparam_cnn = {
    'batch_size': 64,  ##训练batch
    'epoches': 2,  ##训练轮数，超过1轮效果没什么提高
    'max_len': 50,  ##输入序列长度
    'embedding_size': 300,  ##词向量层大小
    'filters': [2, 3, 4, 5],  ##cnn卷积核大小
    'filters_num': 100,  ##cnn过滤器输出
    'label_number': 2,  ##分类类别
    'dropout': 0.1,  ##失活率
    'dense_name': 'dense1'
}
tokenconfig = './model/alldata_tokenconfig.json'
tokenizer = create_tokenizer(tokenconfig)
hparam_cnn['vocab'] = len(tokenizer.word_index) + 1

model = TextCNNa(hparam_cnn)
model = model.build_model()
model.load_weights('./model/weixiu_model_weights.h5')
# from tensorflow.lite.python.lite import TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# print(converter._post_training_quantize)
converter._post_training_quantize = True
tflite_model = converter.convert()
open('weixiu_model.tflite', 'wb').write(tflite_model)


import tensorflow as tf
interpreter = tf.lite.Interpreter(
    model_path=
    "./model_tflite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
import numpy as np
xx=np.array([[1]*50]*10)
print(xx)
print(xx.shape)
interpreter.resize_tensor_input(input_details[0]['index'], (10, 50))
interpreter.resize_tensor_input(output_details[0]['index'], (10, 2))
interpreter.reset_all_variables()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
interpreter.set_tensor(input_details[0]['index'], xx)
interpreter.invoke()
print(1)
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)