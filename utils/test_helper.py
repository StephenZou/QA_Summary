import tensorflow as tf
from tqdm import tqdm


UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

def search_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    results = []
    sample_size = 0
    if params["mode"] == "test":
        sample_size = 20000
    elif params["mode"] == "eval":
        sample_size = 166
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算

    results = []
    if params['decode_type'] == 'greedy':
        steps_epoch = sample_size // batch_size + 1
        for i in tqdm(range(steps_epoch)):
            enc_data, _ = next(iter(dataset))
            results += batch_greedy_decode(model, enc_data, vocab, params)
    elif params['decode_type'] == 'beam':
        # steps_epoch = sample_size
        # for i in tqdm(range(steps_epoch)):
        #     enc_data, _ = next(iter(dataset))
        #     results.append(batch_beam_decode(model, enc_data, vocab, params))
        for batch in dataset:
            enc_data = batch[0]
            results.append(batch_beam_decode(model, enc_data, vocab, params))
    return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]

    predicts = [''] * batch_size
    inputs = tf.convert_to_tensor(batch_data)
    enc_output, enc_hidden = model.call_encoder(inputs)

    dec_hidden = enc_hidden
    dec_input = tf.constant([2] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)

    context_vector, _ = model.attention(dec_hidden, enc_output)
    for t in range(params['max_dec_len']):
        _, pred, dec_hidden = model.decoder(dec_input, context_vector)
        context_vector, _ = model.attention(dec_hidden, enc_output)
        predicted_ids = tf.argmax(pred, axis=1).numpy()
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '
        dec_input = tf.expand_dims(predicted_ids, 1)
    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        if '[STOP]' in predict:
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results


class Hypothesis:
    def __init__(self, tokens, log_probs):
        self.tokens = tokens
        self.log_probs = log_probs
        self.abstract = ""
    def extend(self, token, log_prob):
        return Hypothesis(tokens=self.tokens+[token],
                          log_probs=self.log_probs+[log_prob])
    @property
    def latest_token(self):
        return self.tokens[-1]
    @property
    def tot_log_prob(self):
        return sum(self.log_probs)
    @property
    def avg_log_prob(self):
        return self.tot_log_prob/len(self.tokens)


def print_top_k(hyp, k, vocab, batch):
    text = " ".join([vocab.id_to_word(int(index)) for index in batch[0]])
    print('\nhyp.text :{}'.format(text))
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp.abstract = " ".join([vocab.id_to_word(index) for index in k_hyp.tokens])
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


def batch_beam_decode(model, enc_data, vocab, params):
    start_index = vocab.word_to_id(START_DECODING)
    stop_index = vocab.word_to_id(STOP_DECODING)
    unk_index = vocab.word_to_id(UNKNOWN_TOKEN)
    batch_size = params['batch_size']
    def decode_onestep(context_vector, dec_input, enc_output, prev_output_tokens, batch_size, repetition_penalty):
        # final_dists, _ = model(enc_outputs, dec_input, dec_state)
        _, pred, dec_hidden = model.decoder(dec_input, context_vector)
        context_vector, _ = model.attention(dec_hidden, enc_output)
        pred = enforce_repetition_penalty_(pred, batch_size, prev_output_tokens, repetition_penalty)
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(pred), k=params["beam_size"]*2) # (batch_size, beam_size*2)
        top_k_log_probs = tf.math.log(top_k_probs)
        results = {"top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "context_vector": context_vector}
        return results
    batch_data = enc_data["enc_input"]
    inputs = tf.convert_to_tensor(batch_data)
    enc_output, enc_hidden = model.call_encoder(inputs)

    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0]) for _ in range(batch_size)]

    results = []
    steps = 0
    dec_hidden = enc_hidden
    context_vector, _ = model.attention(dec_hidden, enc_output)
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]
        latest_tokens = [t if t in range(params['vocab_size']) else unk_index for t in latest_tokens]
        prev_output_tokens = [h.tokens for h in hyps]
        # hiddens = [h.hidden for h in hyps]
        dec_input = tf.expand_dims(latest_tokens, axis=1)
        # dec_hidden = tf.stack(hiddens, axis=0)

        returns = decode_onestep(context_vector, dec_input, enc_output, prev_output_tokens, batch_size, params["repetition_penalty"])

        # dec_hidden = returns["dec_hidden"]
        context_vector = returns["context_vector"]
        top_k_ids = returns["top_k_ids"] # (batch_size, beam_size*2)
        top_k_log_probs = returns["top_k_log_probs"]

        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        for i in range(num_orig_hyps):
            # len(hyps) = batch_size
            h = hyps[i]
            for j in range(params["beam_size"]*2):
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j])
                all_hyps.append(new_hyp)
        # all_hyps: batch_size*beam_size*2
        hyps = []
        # 按照概率排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期， 遇到句尾，添加到结果集
                if steps >= params["min_dec_steps"]:
                    results.append(h)
            else:
                hyps.append(h)
            # 如果最优的未完成句子等于beam_size，或是最优完成句子达到beam_size
            if len(hyps) == params["beam_size"] or len(results) == params["beam_size"]:
                break
        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    # print_top_k(hyps_sorted, 3, vocab, enc_data)

    best_hyp = hyps_sorted[0]
    abstract = "".join([vocab.id_to_word(index) for index in best_hyp.tokens])
    if STOP_DECODING in abstract:
        abstract = abstract[:abstract.index(STOP_DECODING)]
    if START_DECODING in abstract:
        abstract = abstract.replace(START_DECODING, '')
    return abstract


def enforce_repetition_penalty_(scores, batch_size, prev_output_tokens, repetition_penalty):
    scores_variable = tf.Variable(scores)
    for i in range(batch_size):
        for previous_token in set(prev_output_tokens[i]):
            if scores_variable[i, previous_token] < 0:
                scores_variable[i, previous_token].assign(scores_variable[i, previous_token]*repetition_penalty)
            else:
                scores_variable[i, previous_token].assign(scores_variable[i, previous_token]/repetition_penalty)
    return tf.convert_to_tensor(scores_variable)
