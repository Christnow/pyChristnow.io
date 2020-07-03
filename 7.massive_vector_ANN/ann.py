'''
海量向量最邻近搜索：
1.sklearn.neighbors.Nearestneighbors
2.annoy.AnnoyIndex

两者皆支持numpy数组
训练环节：sklearn速度比annoy快很多
搜索环节：annoy速度比sklearn快很多个数量级
模型存储和加载：同数量annoy模型大小比sklearn小一倍左右，加载时间更快
               且自带加载函数，sklearn需要结合pickle，
               annoy:目前版本：1.16.0不支持pickle保存模型，保存必须自带的函数

推荐annoy：c/c++实现速度极快

example1:
        a=np.random.rand(1,500)[0]
        x=[]
        for i in range(10000):
            x.append(a+np.array([i]*500))
        x=np.array(x)
        y=np.random.rand(1,500)
        t1=time.time()
        nbrs = NearestNeighbors(n_neighbors=2,metric='cosine').fit(x)
        print('nn train time: {}'.format(time.time()-t1))
        t2=time.time()
        neighbors_exact = nbrs.kneighbors(y, return_distance=False)
        print('nn search time: {}'.format(time.time()-t2))
        with open('nbrs.model','wb') as f:
            pickle.dump(nbrs,f)
        with open('nbrs.model','rb') as f:
            nbrsmodel=pickle.load(f)
        neighbors_exact=nbrsmodel.kneighbors(y, return_distance=False)

example2:
        from annoy import AnnoyIndex
        import random
        import numpy as np
        f = 500
        y=[random.gauss(0, 1) for z in range(f)]
        y=np.array(y)
        t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
        for i in range(10000):
            v = [random.gauss(0, 1) for z in range(f)]
            v=np.array(v)
            t.add_item(i, v)
        t1=time.time()
        t.build(10) # 10 trees
        print('annoy train time: {}'.format(time.time()-t1))
        t2=time.time()
        t.save('annoy4.model')
        u=AnnoyIndex(f, 'angular')
        u.load('annoy1.model')
        print(u.get_nns_by_vector(y, 2)) # will find the 1000 nearest neighbors
        print('annoy search time: {}'.format(time.time()-t2))

'''

import pickle
import numpy as np

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors


class aoyAnn(object):
    def __init__(self,
                 vector=[[0, 0], [1, 1]],
                 treesnum: int = 10,
                 metric: str = 'angular'):
        '''
        Parameters description:
            k: kneighbors number(type: int)
            >>> k = 1 or 2......
            treesnum: 树的深度
            metric: 距离度量方法："angular"：余弦距离，
                                "euclidean"：欧式距离，
                                "manhattan"：曼哈顿距离，
                                "hamming"：汉明距离，
                                "dot"：内积距离
        '''
        self.vector = vector
        self.treesnum = treesnum
        self.metric = metric
        self.model = None

    def _call(self, name, *props):
        callback = getattr(self, name)
        if callback:
            callback(*props)

    def save_annoy_model(self, model, file):
        model.save(file)

    def load_annoy_model(self, modelfile):
        self.model = AnnoyIndex(self.vector.shape[1], self.metric)
        self.model.load(modelfile)

    def ayAnn(self, vector, treesnum=10, metric='angular'):
        '''
        vector: 训练搜索的向量海
        treesnum: 树的深度
        metric: 距离度量方法："angular"：余弦距离，
                             "euclidean"：欧式距离，
                             "manhattan"：曼哈顿距离，
                             "hamming"：汉明距离，
                             "dot"：内积距离
        '''
        vector = np.array(vector)
        annmodel = AnnoyIndex(vector.shape[1], metric)
        for ids, arr in enumerate(vector):
            annmodel.add_item(ids, arr)
        annmodel.build(treesnum)
        return annmodel

    @property
    def train(self):
        self.model = self.ayAnn(self.vector, self.treesnum, self.metric)

    def save(self, file):
        self._call('save_annoy_model', self.model, file)

    def load(self, modelfile):
        self._call('load_annoy_model', modelfile)

    def get_nns(self, v, k, include_distance=True):
        result = []
        nbrs = self.model.get_nns_by_vector(v,
                                            k,
                                            include_distances=include_distance)
        if include_distance:
            for ids, i in enumerate(nbrs[0]):
                nn = self.model.get_item_vector(i)
                info = {'id': i, 'vector': nn, 'distance': nbrs[1][ids]}
                result.append(info)
        else:
            for i in nbrs:
                nn = self.model.get_item_vector(i)
                info = {'id': i, 'vector': nn}
                result.append(info)
        return result


class skAnn(object):
    def __init__(self,
                 k: int = 1,
                 vector=[],
                 algorithm: str = 'auto',
                 metric: str = 'cosine',
                 n_jobs: int or None = None):
        '''
        Parameters description:
            k: kneighbors number(type: int)
            >>> k = 1 or 2......
            Algorithm used to compute the nearest neighbors:
            ‘ball_tree’ will use BallTree
            ‘kd_tree’ will use KDTree
            ‘brute’ will use a brute-force search.
            ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
            metric: from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
            n_jobs: The number of parallel jobs to run for neighbors search. 
                    None means 1 unless in a joblib.parallel_backend context. 
                    -1 means using all processors. 
                    See Glossary for more details.
        '''
        self.k = k
        self.vector = vector
        self.algo = algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        self.model = None

    def _call(self, name, *props):
        callback = getattr(self, name)
        if callback:
            callback(*props)

    def save_sklearn_model(self, model, file):
        with open(file, 'wb') as f:
            pickle.dump(model, f)

    def load_sklearn_model(self, modelfile):
        with open(modelfile, 'rb') as f:
            model = pickle.load(f)
        return model

    def skAnn(self, k, vector, algo='auto', metric='cosine', n_jobs=None):
        '''
        vector: 训练搜索的向量海
        algorithm: 邻近算法:{
            ‘ball_tree’ will use BallTree
            ‘kd_tree’ will use KDTree
            ‘brute’ will use a brute-force search.
            ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        }
        metric: 距离度量方法：[‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
        '''
        skannmodel = NearestNeighbors(n_neighbors=k,
                                      algorithm=algo,
                                      metric=metric,
                                      n_jobs=n_jobs).fit(vector)
        return skannmodel

    @property
    def train(self):
        self.model = self.skAnn(self.k, self.vector, self.algo, self.metric,
                                self.n_jobs)

    def save(self, file):
        self._call('save_sklearn_model', self.model, file)

    def load(self, modelfile):
        self.model = self.load_sklearn_model(modelfile)

    def get_nns(self, vector, k, return_distance=True):
        result = []
        nbrs = self.model.kneighbors(vector,
                                     k,
                                     return_distance=return_distance)
        if return_distance:
            for ids, i in enumerate(nbrs[1][0]):
                nn = self.vector[i]
                info = {'id': i, 'vector': nn, 'distance': nbrs[0][0][ids]}
                result.append(info)
        else:
            for i in nbrs[0]:
                nn = self.vector[i]
                info = {'id': i, 'vector': nn}
                result.append(info)
        return result


'''
class docAnn():
    from gensim.test.utils import common_texts
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
            min_count：忽略所有单词中单词频率小于这个值的单词。
            window：窗口的尺寸。（句子中当前和预测单词之间的最大距离）
            size:特征向量的维度
            sample：高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
            negative: 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）。默认值是5。
            workers：用于控制训练的并行数。

            model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
            total_examples：统计句子数
            epochs：在语料库上的迭代次数(epochs)。

    model.train(documents, total_examples=model.corpus_count, epochs=70)
    print(common_texts)
    # model.train()
    dm = model.infer_vector(['human', 'time'])
    print(dm)
    rr=[]
    result = model.docvecs.most_similar([dm], topn=1)
    for ids,sim in result:
        rr.append({'ids': ids, 'vector': common_texts[ids], 'similar': sim})
    print(rr)
    model.save('doc.model')

    model=Doc2Vec.load('doc.model')
    dm = model.infer_vector(['human', 'time'])
    print(dm)
    rr = []
    result = model.docvecs.most_similar([dm], topn=1)
    for ids, sim in result:
        rr.append({'ids': ids, 'vector': common_texts[ids], 'similar': sim})
    print(rr)

'''

if __name__ == '__main__':
    import time
    import random
    import numpy as np
    x = []
    a = np.random.rand(1, 2)[0]
    for i in range(10):
        x.append(a + [i] * 2)
    x = np.array(x)
    y = np.random.rand(1, 2)[0]
    model = aoyAnn(x, 10)
    t = time.time()
    model.train
    nbr = model.get_nns(y, 2, True)
    nbr1 = model.get_nns(y, 2, False)
    print('aotann train time: {}'.format(time.time() - t))
    print(nbr)
    print(nbr1)
    model.save('testann1.model')

    # model = aoyAnn(x, 10)
    # model.load('testann1.model')
    # print(model.get_nns(y, 1))
    print('#######################################################')
    print(x)
    print(y)
    model = skAnn(2, x)
    model.train
    print(model.get_nns([y], 2, False))
    print(model.get_nns([y], 2, True))
    model.save('testskann.model')
    model = skAnn(2, x)
    model.load('testskann.model')
    print(model.get_nns([y], 2, False))
    print(model.get_nns([y], 2, True))
