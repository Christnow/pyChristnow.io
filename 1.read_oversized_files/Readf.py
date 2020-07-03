import traceback
import pandas as pd

class Readf():
    '''
    生成器迭代器按行读取超大文件
    '''
    def __init__(self, file):

        self.file = file

    def __iter__(self):

        try:
            with open(self.file, 'r', encoding='utf-8') as f:
                for i, j in enumerate(f):
                    yield j.strip()
        except Exception:
            traceback.print_exc()


class Readp():
    '''
    Pandas读取文件 apply和map处理
    '''
    def __init__(self, file):

        self.file = file

    def read(self):

        sp = re.compile("[,; ；]+")
        names = ['zt', 'cid', 'cname', 'keywords', 'll', 'le']
        df = pd.read_csv(self.file, sep='|', header=None,
                         names=names)[names].dropna()
        df['zt'] = df['zt'].apply(lambda s: list(filter(None, sp.split(s))))
        df['cid'] = df['cid'].apply(lambda s: list(filter(None, sp.split(s))))
        df['cname'] = df['cname'].apply(
            lambda s: list(filter(None, sp.split(s))))
        df['keywords'] = df['keywords'].apply(
            lambda s: list(filter(None, sp.split(s))))
        df['ll'] = df['ll'].apply(lambda s: list(filter(None, sp.split(s))))
        df['le'] = df['le'].apply(lambda s: s.split())
        return df[(df['zt'].map(len) > 0) & (df['cid'].map(len) > 0) &
                  (df['cname'].map(len) > 0) & (df['keywords'].map(len) > 0) &
                  (df['ll'].map(len) > 0) & (df['le'].map(len) > 0)]


####df = df[(True ^ df['keyword'].isnull())]  ###去除nan值

if __name__ == '__main__':

    file = './r.txt'
    for i in Readf(file):
        print(i)
    text = list(filter(lambda x: len(x) >= 5, Readf(file)))
    print(text)