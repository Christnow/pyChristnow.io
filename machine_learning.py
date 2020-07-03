import os
import re
import sys
import time
import json
import platform
import pickle
import traceback
import collections
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
if platform.platform().lower().startswith('linux'):
    mpl.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import jieba
import jieba.posseg as pseg
from PIL import Image
from wordcloud import WordCloud

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.svm import SVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report


def save_model(model, file):
    ## 保存模型
    with open(file, 'wb') as f:
        pickle.dump(model, f)


def read_model(file):
    ## 读取模型
    with open(file, 'rb') as f:
        model = pickle.load(f)
    return model


def load_stopwords(file):
    with open(file, 'r', encoding='utf-8') as f:
        sp = [x.strip() for x in f if x.strip()]
    return sp


stopwords = load_stopwords('./data/哈工大停用词表.txt')


def segment(text):
    cnp = re.compile('[^\u4e00-\u9fa5]+')
    text = re.sub(cnp, '', text)
    text = [x for x in jieba.cut(text) if len(x) >= 2 and x not in stopwords]
    return text


def plot_wordcloud(df):
    object_list = []
    for x in df['text'].values.tolist():
        object_list.extend(x.split())

    word_counts = collections.Counter(object_list)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common(10)  # 获取前10最高频的词
    print(f'word_counts_top10: {word_counts_top10}')  # 输出检查

    font = './data/STKAITI.TTF'
    wc = WordCloud(font_path=font,
                   mode='RGBA',
                   background_color='white',
                   max_words=200).generate_from_frequencies(word_counts)

    # 显示词云
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # 保存到文件
    wc.to_file('./result/wordcloud_alice.png')


def train_tfidf(alldata):
    ###训练tfidf模型
    tf = TfidfVectorizer()
    tf.fit(alldata['text'])
    save_model(tf, './result/tfidf.model')
    print(tf.transform([alldata['text'].values.tolist()[0]]))
    return tf


def train_lda(df):
    data = df['text'].values.tolist()
    data = [i for i in data if i is not np.nan]
    cvectorizer = CountVectorizer(min_df=1)
    cvz = cvectorizer.fit_transform(data)
    save_model(cvectorizer, f'./result/cvect.model')
    lda = LatentDirichletAllocation(n_components=8,
                                    learning_method='batch',
                                    max_iter=25,
                                    random_state=0)
    lda.fit_transform(cvz)
    save_model(lda, f'./result/lda/lda.model')
    return cvz, cvectorizer, lda


def print_top_words(model, feature_names, n_top_words):
    #打印每个主题下权重较高的term
    topic_set = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_set[topic_idx] = []
        word = []
        weight = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            word.append(feature_names[i])
            weight.append(topic[i])
        topic_set[topic_idx] = {
            'word': word,
            'weight': weight,
        }
    return topic_set


def train_svm(trainx, trainy, testx, testy, label_name):
    ###用一个分类器对应一个类别， 每个分类器都把其他全部的类别作为相反类别看待。
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    clf.fit(trainx, trainy)
    save_model(clf, './result/svm/svm_tfidf_model.pkl')
    ### 保存评估结果
    scores = clf.score(testx, testy)
    print('knn score: {}'.format(scores))
    with open('./result/svm/svm_scores.txt', 'w', encoding='utf-8') as f:
        f.write('{}\n'.format(scores))
    ## 画roc
    testy1 = testy.reshape(1, -1)[0]
    testp = clf.predict_proba(testx)[:, 1]
    fpr, tpr, _ = roc_curve(testy1, testp, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM TFIDF MODEL')
    plt.legend(loc="lower right")
    plt.savefig('./result/svm/svm_tfidf_roc.png')
    plt.show()
    y_pred = clf.predict(testx)
    score = classification_report(y_true=testy,
                                  y_pred=y_pred,
                                  target_names=label_name)
    with open('./result/svm/label_score.txt', 'w', encoding='utf-8') as f:
        f.write(score)


def train_knn(trainx, trainy, testx, testy, label_name):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(trainx, trainy)
    save_model(clf, './result/knn/knn_tfidf_model.pkl')
    ### 保存评估结果
    scores = clf.score(testx, testy)
    print('knn score: {}'.format(scores))
    with open('./result/knn/knn_scores.txt', 'w', encoding='utf-8') as f:
        f.write('{}\n'.format(scores))
    ## 画roc
    testy1 = testy.reshape(1, -1)[0]
    testp = clf.predict_proba(testx)[:, 1]
    fpr, tpr, _ = roc_curve(testy1, testp, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN TFIDF MODEL')
    plt.legend(loc="lower right")
    plt.savefig('./result/knn/knn_tfidf_roc.png')
    plt.show()
    y_pred = clf.predict(testx)
    score = classification_report(y_true=testy,
                                  y_pred=y_pred,
                                  target_names=label_name)
    with open('./result/knn/label_score.txt', 'w', encoding='utf-8') as f:
        f.write(score)


def train_dtree(trainx, trainy, testx, testy, label_name):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(trainx, trainy)
    save_model(clf, './result/dtree/dtree_tfidf_model.pkl')
    ### 保存评估结果
    scores = clf.score(testx, testy)
    print('dtree score: {}'.format(scores))
    with open('./result/dtree/dtree_scores.txt', 'w', encoding='utf-8') as f:
        f.write('{}\n'.format(scores))
    ## 画roc
    testy1 = testy.reshape(1, -1)[0]
    testp = clf.predict_proba(testx)[:, 1]
    fpr, tpr, _ = roc_curve(testy1, testp, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DTREE TFIDF MODEL')
    plt.legend(loc="lower right")
    plt.savefig('./result/dtree/dtree_tfidf_roc.png')
    plt.show()
    y_pred = clf.predict(testx)
    score = classification_report(y_true=testy,
                                  y_pred=y_pred,
                                  target_names=label_name)
    with open('./result/dtree/label_score.txt', 'w', encoding='utf-8') as f:
        f.write(score)


def main():
    ### 读取原数据，预处理：分词，zhengze
    datafile = './data/results.xlsx'
    savefile = './result/data.xlsx'
    if os.path.isfile(savefile):
        df = pd.read_excel(savefile)
    else:
        df = pd.read_excel(datafile)[['微博内容']]
        df.columns = ['text']
        df['text'] = df['text'].apply(lambda x: x.strip())
        df = df.fillna(-1)
        df = df[df['text'] != -1]
        df['text'] = df['text'].apply(lambda x: ' '.join(segment(x)))
        df.to_excel('./result/data.xlsx', index=False)
    print(df.head())

    ### 画词云图
    plot_wordcloud(df)

    ### 训练tfidf模型
    if os.path.isfile('./result/tfidf.model'):
        tfidf = read_model('./result/tfidf.model')
    else:
        tfidf = train_tfidf(df)

    ### 训练lda模型
    if os.path.isfile('./result/lda/lda.model'):
        cvect = read_model('./result/cvect.model')
        lda = read_model('./result/lda/lda.model')
        cvz = cvect.transform(df['text'].values.tolist())
    else:
        cvz, cvect, lda = train_lda(df)

    ### 打印lda模型主题结果，n_top_words：表示每个主题打印20个主题词
    feature_names = cvect.get_feature_names()
    topic_set = print_top_words(lda, feature_names, n_top_words=20)
    df_lda = pd.DataFrame()
    for j in topic_set.keys():
        df_lda['topic_{}_word'.format(j)] = topic_set[j]['word']
        df_lda['topic_{}_weight'.format(j)] = topic_set[j]['weight']
    df_lda.to_excel(f'./result/lda/lda_result.xlsx', index=False)

    ### 运用lda模型为原数据打上八个类别的标签
    df_cvect = cvect.transform(df['text'].values.tolist())
    lda_label = lda.transform(df_cvect)
    for number in range(8):
        topic_name = f'topic_{number}'
        df[topic_name] = lda_label[:, number].tolist()
    df['label'] = np.argmax(df[[f'topic_{number}'
                                for number in range(8)]].values,
                            axis=1).tolist()
    df.to_excel('./result/data_lda_label.xlsx', index=False)

    df = df[['text', 'label']]
    ### 画图：每个主题数目
    pf = {'topic': [], 'number': []}
    for label, group in df.groupby(['label']):
        pf['topic'].append(label)
        pf['number'].append(group.shape[0])
    pf = pd.DataFrame(pf)
    pf.plot(kind='barh', title='topic number')
    plt.savefig('./result/topic_number.png')

    ### 总计八个类别，按照类别分割训练测试集
    train, test = pd.DataFrame(), pd.DataFrame()
    for label, group in df.groupby(['label']):
        trainx, testx, trainy, testy = train_test_split(group['text'],
                                                        group['label'],
                                                        test_size=0.1)
        traind = pd.concat([trainx, trainy], axis=1)
        train = pd.concat([train, traind])
        testd = pd.concat([testx, testy], axis=1)
        test = pd.concat([test, testd])
    train.to_excel('./result/train.xlsx', index=False)
    test.to_excel('./result/test.xlsx', index=False)
    train = shuffle(train)
    test = shuffle(test)

    ### 转化word特征为tfidf特征
    label_name = [f'topic_{number}' for number in range(8)]
    trainx = tfidf.transform(train['text'])
    trainy = train['label'].values
    testx = tfidf.transform(test['text'])
    testy = test['label'].values

    ### 训练 knn dtree svm，并计算评估分数，roc曲线图，类别报告
    train_knn(trainx, trainy, testx, testy, label_name)
    train_dtree(trainx, trainy, testx, testy, label_name)
    # train_svm(trainx, trainy, testx, testy, label_name)


def predict(text='我爱北京天安门'):
    ### 单句预测
    text = [' '.join(segment(text))]
    cvect = read_model('./result/cvect.model')
    lda = read_model('./result/lda/lda.model')

    ### 预测lda主题
    text_cvect = cvect.transform(text)
    topic = lda.transform(text_cvect)[0]
    topic = np.argmax(topic)
    print(f'text所属主体：topic_{topic}')

    ### 预测svm，knn，决策树
    tfidf = read_model('./result/tfidf.model')
    svm_model = read_model('./result/svm/svm_tfidf_model.pkl')
    knn_model = read_model('./result/knn/knn_tfidf_model.pkl')
    dtree_model = read_model('./result/dtree/dtree_tfidf_model.pkl')
    text_tfidf = tfidf.transform(text)
    svm_result = svm_model.predict_proba(text_tfidf)[0]
    print(f'svm result: {svm_result}')

    knn_result = knn_model.predict_proba(text_tfidf)[0]
    print(f'knn result: {knn_result}')

    dtree_result = dtree_model.predict_proba(text_tfidf)[0]
    print(f'dtree result: {dtree_result}')

if __name__ == '__main__':
    main()
    predict()