import tensorflow as tf

from pgn_model.models.pgn import PGN
from pgn_model.models.sequence_to_sequence import SequenceToSequence
from utils.batcher import Vocab, batcher
from utils.train_helper import train_model
from .test_helper import search_decode
from tqdm import tqdm
from data_processing.utils.data_utils import get_result_filename
import pandas as pd
import pprint
import os
from rouge import Rouge
from transformer_pgn.models.transformer import PGN_TRANSFORMER


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    # model = SequenceToSequence(params)
    # model = PGN(params)
    model = PGN_TRANSFORMER(params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    # 防止训练过程中中断，不需要重新训练
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


def test(params):
    assert params["mode"].lower() == "test" or params["mode"].lower() == "eval" , "change training mode to 'test' or 'eval'"
    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])


    # b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
    ckpt = tf.train.Checkpoint(SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    results = []
    results = predict_result(model, params, vocab, params['test_save_dir'])
    print("results个数："+str(len(results)))
    return results


def predict_result(model, params, vocab, result_save_path):
    print("Creating the batcher ...")
    dataset = batcher(vocab, params)
    results = search_decode(model, dataset, vocab, params)
    if params["mode"] == "test":
        results = list(map(lambda x: x.replace(" ", ""), results))
        save_predict_result(results, params)
    elif params["mode"] == "eval":
        # save_eval_result(results, params)
        results = results[:166]
        save_eval_result(results, params)
    return results


def save_eval_result(results, params):
    # eval_df = pd.DataFrame(data=results)
    result_save_path = os.path.join(params['test_save_dir'], 'val_set.predict_y.txt')
    # eval_df.to_csv(result_save_path)
    with open(result_save_path, 'w') as f:
        for result in results:
            f.write(result+'\r\n')


def save_predict_result(results, params):
    # 读取结果
    test_df = pd.read_csv(params['test_x_dir'])
    test_df['Prediction'] = results[:20000]
    test_df = test_df['QID', 'Prediction']
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


def evaluate(params):
    results = test(params)
    preds = []
    with open(os.path.join(params["test_save_dir"], "val_set.seg_y.txt"), encoding='utf-8') as f:
        for line in f.readlines():
            preds.append(line)
    rouge = Rouge()
    scores = rouge.get_scores(preds, results, avg=True)
    print("\n\n")
    pprint.pprint(scores)
