<center><h1>Python优化</h1></center>

## 一、Python代码部分
### 1.Cython加速
项目文件树如下：
```python
|--logs
|--script
|    |
|    |--model.pyx
|
|--setup.py
|--setup.sh

1.将原Python文件model.py改写成model.pyx

2.setup.py代码：
  from distutils.core import setup
  from Cython.Build import cythonize
  import os
  setup(name='byv4_script',
      ext_modules=cythonize(os.path.join(os.getcwd(), "script/byv4.py")),
      version='1.0.0')

3.setup.sh代码：
  echo `python setup.py build_ext --inplace`

4.windows生成直接调用编译过的pyd文件
  linux生成系统可调用so文件
  python除了上述两个还会产生一个编译成c语言的代码

5.使用setup注意：
  os.path.dirname(os.path.abspath(__file__))
  此处代码不能使用，因为编译过程会默认执行不存在自身文件名__file__
  增加默认搜索路径改为：
  os.path.join(os.getcwd(), 'script')
  插入路径：
  sys.path.insert(0, childir)
  或者可增加__init__.py文件，import和from .. import
  改成from .model import

此种修改多循环代码可增加10倍左右速度
```
### 2.Python生成器

> 无论是numpy还是其他数据格式均改成生成器进行迭代，
> 无垃圾丢弃，用一次即可释放所有内存。

## 二、神经网络模型部分
### 1.predict函数的速率
```python
以keras或者tensorflow2.0为例：
几个预测函数的速度优先：

1.预测一次方法：
  tf.lite.invoke >> model.predict_generator >> model.predict
  后两者差距不算大，tf.lite速度可比单纯的predict快10-100倍

2.一次需要预测多次的方法：
  model.predict_generator >> model.predict
  tf.lite.invoke需要编写for训练。
  单纯的假设有数据：a=[[1,2],[3,4]]
  需要一次预测两个的情况：

  def tflite_predict(x):
      for x in a:
          result=tf.lite.invoke()

  def model_predict(x):
      result=model.predict(x)

  > 此种情况model.predict最快，没有之一
```
### 2.tf.lite的使用
> tensorflow1.x系列支持：keras保存的model文件加载压缩，
> 压缩率3/4
> 
> tensorflow2.x支持对于Python代码的keras模块直接转化，
> 单次预测速度极快
#### tf.lite原理
1. Post Training Quantization
   > Post Training Quantization合理说,计算过程皆为Float,而非Int,
   > 所以只能在减少模型的大小,速度方面并不能得到提升.
   - Weight only quantization
     > 这种模式,是将模型的weight进行quantized压缩成uint8,
     > 但在计算过程中,会将weight进行dequantized回Float.
   - Quantizing weights and activations
     > 这种模式,在weight quantization的基础上,对某些支持quantized的Kernel,先进行quantization,
     > 再进行activation计算,再de-quant回float32,不支持的话会直接使用Float32进行计算,
     > 这相对与直接使用Float32进行计算会快一些.
2. Quantization Aware Training
   > 这种模式,除了会对weight进行quantization,也会在训练过程中,
   > 进行模拟量化,求出各个op的max跟min输出,实现不仅仅在训练过程,
   > 在测试过程,全程计算过程皆为uint8.不仅仅实现模型的压缩,计算速度也得到提高.

#### 使用方法：
> python3.7.4
> 
> keras2.3
> 
> tensorflow2.2
#### model.py
```python
@add_start_docstrings('保养分类和维修分类模型：四维并列卷积')
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

>model.fit
>model.save_weights
```
#### model_to_tflite.py
```python
import tensorflow as tf
model = TextCNNa(hparam_cnn)
model = model.build_model()
model.load_weights('./model/weixiu_model_weights.h5')
# from tensorflow.lite.python.lite import TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# print(converter._post_training_quantize)
# converter._post_training_quantize = True
tflite_model = converter.convert()
open('weixiu_model.tflite', 'wb').write(tflite_model)
```
#### model_predict.py
```python
import tensorflow as tf
import numpy as np
xx=np.array([[1]*50])
interpreter = tf.lite.Interpreter(
    model_path=
    "./model_tflite.tflite")
interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# interpreter.resize_tensor_input(input_details[0]['index'], (10, 50))
# interpreter.resize_tensor_input(output_details[0]['index'], (10, 2))
# interpreter.reset_all_variables()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], xx)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```