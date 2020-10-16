import tensorflow as tf
import time
from pgn_model.utils.losses import loss_function
from transformer_pgn.schedules.lr_schedules import CustomSchedule
from transformer_pgn.layers.transformer import create_mask

START_DECODING = '[START]'


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def train_model(model, dataset, params, ckpt, ckpt_manager):
    learning_rate = CustomSchedule(params["d_model"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    # 定义损失函数 改为PGN网络，注释掉
    # def loss_function(real, pred):
    #     mask = tf.math.logical_not(tf.math.equal(real, 1))
    #     dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)
    #
    #     loss_ = loss_object(real, pred)
    #     mask = tf.cast(mask, dtype=loss_.dtype)
    #     loss_ *= mask  # 去掉padding词
    #     loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
    #     return tf.reduce_mean(loss_)


    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, padding_mask):
        with tf.GradientTape() as tape:
            # enc_output, enc_hidden = model.call_encoder(enc_inp)
            # dec_hidden = enc_hidden
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(enc_inp, dec_inp)
            # pred, _ = model(enc_output, dec_inp, dec_hidden, dec_tar)
            outputs = model(enc_inp,
                            enc_extended_inp,
                            batch_oov_len,
                            dec_inp,
                            params['training'],
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask)
            pred = outputs["logits"]
            loss = loss_function(dec_tar, pred)
            # loss = loss_function(dec_tar,
            #                      outputs,
            #                      padding_mask,
            #                      params["cov_loss_wt"],
            #                      params["is_coverage"])

        # variables = model.encoder.trainable_variables +\
        #             model.attention.trainable_variables + \
        #             model.decoder.trainable_variables + \
        #             model.pointer.trainable_variables
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        for batch in dataset:
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[0]["extended_enc_input"],  # shape=(16, 200)
                              batch[1]["dec_input"],  # shape=(16, 50)
                              batch[1]["dec_target"],  # shape=(16, 50)
                              batch[0]["max_oov_len"],
                              batch[1]["sample_decoder_pad_mask"])
            # loss = train_step(batch[0]["enc_input"],
            #                   batch[0]["extended_enc_input"],
            #                   batch[1]["dec_input"],
            #                   batch[1]["dec_target"],
            #                   batch[0]["max_oov_len"],
            #                   batch[0]["sample_encoder_pad_mask"],
            #                   batch[1]["sample_decoder_pad_mask"])
            step += 1
            total_loss += loss
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, step, total_loss/step))
        if total_loss/step < best_loss:
            best_loss = total_loss/step
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))