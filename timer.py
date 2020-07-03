import time


class timeiter(object):
    __slots__ = ['desc', 'start_time', 'end_time', 'total_time']

    def __init__(self, desc=''):
        self.desc = desc

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.total_time = round(self.end_time - self.start_time, 8)
        print(f'{self.desc}: {self.total_time}')


if __name__ == '__main__':
    with timeiter(desc='process range time is'):
        s = [x for x in range(10000)]
        with timeiter(desc='one range time is'):
            b = [x for x in range(100000)]
