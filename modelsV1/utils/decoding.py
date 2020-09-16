import tensorflow as tf

def calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    vocab_dists = [p_gen*dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1-p_gen)*dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    extended_size = vocab_size + batch_oov_len # 总词汇表大小加上oov词汇表大小
    extra_zeros = tf.zeros((batch_size, batch_oov_len))

    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    batch_nums = tf.range(0, limit=batch_size) # batch_nums=[0,1,2,3,4]
    batch_nums = tf.expand_dims(batch_nums, 1) # batch_nums=[[0],[1],[2],[3],[4]]

    attn_len = tf.shape(_enc_batch_extend_vocab)[1] # 3
    batch_nums = tf.tile(batch_nums, [1, attn_len]) # batch_nums = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)

    shape=[batch_size, extended_size]
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # 最终合并vocab distribution 和 attn distribution
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists