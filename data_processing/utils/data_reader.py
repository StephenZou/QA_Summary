from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        # 此处f1等同于f1.readlines()
        for line in f1:
            words += line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            # 统计最低词频以上的词，如果已经降序排列，则后续词频都会小于min_count， 所以continue要换成break
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    vocab = [(v, i) for i,v in enumerate(result)]
    # reverse_vocab = [(i,v) for i,v in enumerate(result)]
    return vocab