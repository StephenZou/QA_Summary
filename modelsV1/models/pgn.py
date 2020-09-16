import tensorflow as tf
from modelsV1.encoders import rnn_encoders
from modelsV1.decoders import rnn_decoders
from data_processing.utils.data_utils import load_word2vec
from modelsV1.utils import decoding

class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = rnn_encoders.Encoder(params["vocab_size"], params["embed_size"], params["enc_units"],
                                            params["batch_size"], self.embedding_matrix)
        self.attention = rnn_decoders.BahdanauAttention(params["attn_units"])
        self.decoder = rnn_decoders.Decoder(params["vocab_size"], params["embed_size"], params["dec_units"],
                                            params["batch_size"], self.embedding_matrix)
        self.pointer = rnn_decoders.Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp,
             dec_inp, batch_oov_len, enc_padding_mask, use_coverage, pre_coverage):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []

        context_vector, attention, coverage_next = self.attention(dec_hidden, enc_output, enc_padding_mask, use_coverage, pre_coverage)
        for t in range(dec_inp.shape[1]):
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:,t], axis=1), context_vector)
            context_vector, attention, coverage_next = self.attention(dec_hidden, enc_output, enc_padding_mask, use_coverage, coverage_next)
            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
            predictions.append(pred)
            attentions.append(attention)
            coverages.append(coverage_next)
            p_gens.append(p_gen)
        final_dists = decoding.calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len, self.params["vocab_size"],self.params["batch_size"])
        if self.params["mode"] == "train":
            outputs = dict(logits=final_dists, dec_hidden=dec_hidden, attentions=attentions, coverages=coverages, p_gens=p_gens)
        else:
            outputs = dict(logits=tf.stack(final_dists, 1),
                           dec_hidden=dec_hidden,
                           attentions=tf.stack(attentions, 1),
                           coverages=tf.stack(coverages,1),
                           p_gens=tf.stack(p_gens,1))
        return outputs
