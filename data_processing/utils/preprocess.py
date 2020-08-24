import pandas as pd
import os
from data_processing.utils.tokenizer import segment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']

def remove_words(words_list):
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list

def parse_data(train_path, test_path):
    # 读取文件
    train_df = pd.read_csv(train_path, encoding='utf-8')
    # 去除Report列中的NaN值
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    # 用空格填充NaN值
    train_df.fillna('', inplace=True)
    # 将question列和dialogue列链接起来
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is ', len(train_x))
    # 清洗数据，去除固定的词
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def preprocess_sentence(sentence):
    # 分词
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line