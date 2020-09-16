import tensorflow as tf

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


class Vocab:
    def __init__(self, vocab_file, max_size):
        self.word2id = {UNKNOWN_TOKEN: 0, PAD_TOKEN: 1, START_DECODING: 2, STOP_DECODING: 3}
        self.id2word = {0: UNKNOWN_TOKEN, 1: PAD_TOKEN, 2: START_DECODING, 3: STOP_DECODING}
        self.count = 4
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(r'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, '
                                    r'but %s is' % w)

                if w in self.word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading."
                          % (max_size, self.count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" %
              (self.count, self.id2word[self.count - 1]))

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


# 如果解码序列未达到最大解码长度，则增加stop标记，如果超过最大解码长度，则按最大解码长度截断。
def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:
        inp = inp[:max_len]
        target = target[:max_len]
    else:
        target.append(stop_id)
    assert len(inp) == len(target)
    return inp, target


def example_generator(vocab, train_x_path, train_y_path, eval_x_path, test_x_path,
                      max_enc_len, max_dec_len, mode, batch_size, decode_type, beam_size):
    if mode == "train":
        dataset_train_x = tf.data.TextLineDataset(train_x_path)
        dataset_train_y = tf.data.TextLineDataset(train_y_path)
        train_dataset = tf.data.Dataset.zip((dataset_train_x, dataset_train_y))

        for raw_record in train_dataset:
            article = raw_record[0].numpy().decode("utf-8")
            abstract = raw_record[1].numpy().decode("utf-8")

            start_decoding = vocab.word_to_id(START_DECODING)
            stop_decoding = vocab.word_to_id(STOP_DECODING)

            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)
            sample_encoder_pad_mask = [1 for _ in range(enc_len)]
            enc_input = [vocab.word_to_id(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            abstract_sentences = [""]
            abstract_words = abstract.split()
            abs_ids = [vocab.word_to_id(w) for w in abstract_words]
            abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs);
            dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)
            _,target = get_dec_inp_targ_seqs(abs_ids_extend_vocab,max_dec_len, start_decoding, stop_decoding)

            dec_len = len(dec_input)

            sample_decoder_pad_mask = [1 for _ in range(dec_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_len": dec_len,
                "dec_input": dec_input,
                "target": target,
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract_sentences,
                "sample_decoder_pad_mask": sample_decoder_pad_mask,
                "sample_encoder_pad_mask": sample_encoder_pad_mask
            }
            yield output

    if mode == "test" or mode == "eval":
        # file_path = ''
        if mode == "test":
            file_path = test_x_path
        else:
            file_path = eval_x_path
        test_dataset = tf.data.TextLineDataset(file_path)
        for raw_record in test_dataset:
            article = raw_record.numpy().decode('utf-8')
            article_words = article.split()[:max_enc_len]
            enc_len = len(article_words)

            enc_input = [vocab.word_to_id(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

            sample_encoder_pad_mask = [1 for _ in range(enc_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": 40,
                "article": article,
                "abstract": '',
                "abstract_sents": [],
                "sample_decoder_pad_mask": [],
                "sample_encoder_pad_mask": sample_encoder_pad_mask
            }
            if decode_type == 'greedy':
                yield output
            else:
                for _ in range(beam_size):
                    yield output


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids=[]
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i==unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size()+article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def batch_generator(generator, vocab, train_x_path, train_y_path, eval_x_path,
                    test_x_path, max_enc_len, max_dec_len, batch_size, mode, decode_type, beam_size):
    dataset = tf.data.Dataset.from_generator(lambda: generator(vocab, train_x_path, train_y_path, eval_x_path,
                                                               test_x_path,max_enc_len, max_dec_len,
                                                               mode, batch_size, decode_type, beam_size),
                                             output_types={
                                                 "enc_len": tf.int32,
                                                 "enc_input": tf.int32,
                                                 "enc_input_extend_vocab": tf.int32,
                                                 "article_oovs": tf.string,
                                                 "dec_input": tf.int32,
                                                 "target": tf.int32,
                                                 "dec_len": tf.int32,
                                                 "article": tf.string,
                                                 "abstract": tf.string,
                                                 "abstract_sents": tf.string,
                                                 "sample_decoder_pad_mask": tf.int32,
                                                 "sample_encoder_pad_mask": tf.int32
                                             },
                                             output_shapes={
                                                 "enc_len": [],
                                                 "enc_input": [None],
                                                 "enc_input_extend_vocab": [None],
                                                 "article_oovs": [None],
                                                 "dec_input": [None],
                                                 "target": [None],
                                                 "dec_len": [],
                                                 "article": [],
                                                 "abstract": [],
                                                 "abstract_sents": [None],
                                                 "sample_decoder_pad_mask": [None],
                                                 "sample_encoder_pad_mask": [None]
                                             })
    def update(entry):
        return ({"enc_input": entry["enc_input"],
                 "extended_enc_input": entry["enc_input_extend_vocab"],
                 "article_oovs": entry["article_oovs"],
                 "enc_len": entry["enc_len"],
                 "article": entry["article"],
                 "max_oov_len": tf.shape(entry["article_oovs"])[1],
                 "sample_encoder_pad_mask": entry["sample_encoder_pad_mask"]},

                {"dec_input": entry["dec_input"],
                 "dec_target": entry["target"],
                 "dec_len": entry["dec_len"],
                 "abstract": entry["abstract"],
                 "sample_decoder_pad_mask": entry["sample_decoder_pad_mask"]})

    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=({
                                       "enc_len": [],
                                       "enc_input": [None],
                                       "enc_input_extend_vocab": [None],
                                       "article_oovs": [None],
                                       "dec_input": [max_dec_len],
                                       "target": [max_dec_len],
                                       "dec_len": [],
                                       "article": [],
                                       "abstract": [],
                                       "abstract_sents": [None],
                                       "sample_decoder_pad_mask": [max_dec_len],
                                       "sample_encoder_pad_mask": [None]
                                   }),
                                   padding_values={"enc_len": -1,
                                                   "enc_input": 1,
                                                   "enc_input_extend_vocab": 1,
                                                   "article_oovs": b'',
                                                   "dec_input": 1,
                                                   "target": 1,
                                                   "dec_len": -1,
                                                   "article": b'',
                                                   "abstract": b'',
                                                   "abstract_sents": b'',
                                                   "sample_decoder_pad_mask": 0,
                                                   "sample_encoder_pad_mask": 0},
                                   drop_remainder=True)
    dataset = dataset.map(update)
    return dataset



def batcher(vocab, hpm):
    dataset = batch_generator(example_generator, vocab, hpm["train_seg_x_dir"], hpm["train_seg_y_dir"],
                              hpm["eval_seg_x_dir"],hpm["test_seg_x_dir"], hpm["max_enc_len"],
                              hpm["max_dec_len"], hpm["batch_size"], hpm["mode"], hpm['decode_type'], hpm['beam_size'])
    return dataset
