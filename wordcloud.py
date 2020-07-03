import collections
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plt_wordcloud():
    allword = pos + neg
    object_list = []
    for x in allword:
        for word in x.split():
            if word:
                object_list.append(word)

    word_counts = collections.Counter(object_list)  # 对分词做词频统计
    word_counts_top10 = word_counts.most_common(10)  # 获取前10最高频的词
    print(word_counts_top10)  # 输出检查

    font = './image/STKAITI.TTF'
    mask = np.array(Image.open("./image/alice.png"))
    wc = WordCloud(mask=mask,
                   font_path=font,
                   mode='RGBA',
                   background_color='white',
                   max_words=200).generate_from_frequencies(word_counts)

    # 显示词云
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    # 保存到文件
    wc.to_file('./result/wordcloud_alice.png')

    font = './image/STKAITI.TTF'
    wc = WordCloud(font_path=font,
                   mode='RGBA',
                   background_color='white',
                   max_words=200).generate_from_frequencies(word_counts)

    # 显示词云
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # 保存到文件
    wc.to_file('./result/wordcloud_none.png')


plt_wordcloud()