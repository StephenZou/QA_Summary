import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super().__init__()
        self.batch_sz = batch_sz
        # 双向
        self.enc_units = enc_units//2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        # return_sequences为True返回整个序列，形如(samples, timesteps, output_dim)，为False返回序列的最后一个输出, 形如(samples, output_dim);
        # return_state:除了返回output外，是否还要返回hidden state。
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1) #shape(256,128)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden) #output:(256,200,256) forward_state:(256,128) backward_state:(256,128)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
