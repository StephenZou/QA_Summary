import tensorflow as tf
from modelsV1.encoders import rnn_encoders
from modelsV1.decoders import rnn_decoders
from data_processing.utils.data_utils import load_word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = rnn_encoders.Encoder(params['vocab_size'],
                                            params['embed_size'],
                                            params['enc_units'],
                                            params['batch_size'],
                                            self.embedding_matrix)
        self.attention = rnn_decoders.BahdanauAttention(params['attn_units'])
        self.decoder = rnn_decoders.Decoder(params['vocab_size'],
                                            params['embed_size'],
                                            params['dec_units'],
                                            params['batch_size'],
                                            self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_inp, dec_hidden, dec_tar, epsilon):
        predictions = []
        attentions = []
        context_vector, _ = self.attention(dec_hidden, enc_output)  # dec_hidden(256,256) enc_output(256,200,256)
        pred = '[START]'
        for t in range(dec_tar.shape[1]):
            # Teacher Forcing
            _, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),  context_vector)
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)
            predictions.append(pred)
            attentions.append(attn_dist)
        return tf.stack(predictions, 1), dec_hidden