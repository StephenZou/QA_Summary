from data_processing.utils.preprocess import parse_data
from data_processing.utils.data_reader import read_data, build_vocab, save_word_dict
from data_processing.utils.build_w2v import build
from data_processing.utils.dataset_split import train_val_split

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SET_X_PATH = '{}/datasets/train_set.seg_x.txt'
TRAIN_SET_Y_PATH = '{}/datasets/train_set.seg_y.txt'
TEST_SET_X_PATH = '{}/datasets/test_set.seg_x.txt'
VAL_SET_X_PATH = '{}/datasets/val_set.seg_x.txt'
VAL_SET_Y_PATH = '{}/datasets/val_set.seg_y.txt'

if __name__ == "__main__":
    parse_data('{}/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/AutoMaster_TestSet.csv'.format(BASE_DIR))
    lines = read_data(TRAIN_SET_X_PATH.format(BASE_DIR),
                      TRAIN_SET_Y_PATH.format(BASE_DIR),
                      TEST_SET_X_PATH.format(BASE_DIR))
    vocab = build_vocab(lines)
    save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))
    build(TRAIN_SET_X_PATH.format(BASE_DIR),
          TRAIN_SET_Y_PATH.format(BASE_DIR),
          TEST_SET_X_PATH.format(BASE_DIR),
          out_path='{}/datasets/word2vec.txt'.format(BASE_DIR),
          sentence_path='{}/datasets/sentences.txt'.format(BASE_DIR))
    train_val_split(TRAIN_SET_X_PATH.format(BASE_DIR),
                    TRAIN_SET_Y_PATH.format(BASE_DIR),
                    VAL_SET_X_PATH.format(BASE_DIR),
                    VAL_SET_Y_PATH.format(BASE_DIR))