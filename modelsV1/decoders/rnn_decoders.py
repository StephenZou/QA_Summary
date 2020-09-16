import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.Wc = tf.keras.layers.Dense(units)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # PGN网络，加入enc_padding_mask, use_coverage, pre_coverage
    def call(self, dec_hidden, enc_output, enc_padding_mask, use_coverage=False, pre_coverage = None):
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1) # shape=(256, 1, 256)

        def masked_attention(score):
            attn_dist = tf.squeeze(score, axis=2)
            attn_dist = tf.nn.softmax(attn_dist, axis = 1)
            mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
            attn_dist *= mask
            mask_sums = tf.reduce_sum(attn_dist, axis = 1)
            attn_dist = attn_dist / tf.reshape(mask_sums, [-1,1])
            attn_dist = tf.expand_dims(attn_dist, axis = 2)
            return attn_dist

        if use_coverage and pre_coverage is not None:
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis) + self.Wc(pre_coverage)))
            attn_dist = masked_attention(e)
            # coverage为t时刻之前的attn_weight的加和，所以仅需要当前的attn_weight加上pre_coverage即可
            coverage = attn_dist + pre_coverage
        else:
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
            attn_dist = masked_attention(e)
            if use_coverage:
                # t0时刻
                coverage = attn_dist
            else:
                coverage = []

        context_vector = tf.reduce_sum(attn_dist*enc_output, axis=1)
        return context_vector, tf.squeeze(attn_dist, -1), coverage
        # seq2seq模型部分
        # score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))  # score:(256,200,1)
        # attn_dist = tf.nn.softmax(score, axis=1)  # attn_dist:(256,200,1)
        # context_vector = attn_dist * enc_output  # enc_output: (256, 200, 256)
        # context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(256, 256)
        # return context_vector, tf.squeeze(attn_dist, -1)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, context_vector):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))

        out = self.fc(output)
        return x, out, state

class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state)+
                             self.w_i_reduce(dec_inp)+self.w_c_reduce(context_vector))