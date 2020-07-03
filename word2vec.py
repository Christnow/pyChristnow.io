import pandas as pd
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

sentences = {'text':['我 爱 北京 天安门','我 要 吃饭']} 
sentences=pd.DataFrame(sentences)

sentences = [x.split() for x in sentences['text']]
print(sentences[:2])
model = Word2Vec(sentences, min_count=1, window=3, size=100)
model.wv.save_word2vec_format('./word2vec.txt', binary=False)
model = KeyedVectors.load_word2vec_format('./word2vec.txt',binary=False)
print(model.wv['我'])