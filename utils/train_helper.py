import tensorflow as tf
import time

START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params['learning_rate'])
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 1))
        dec_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.float32), axis=-1)

        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask  # 去掉padding词
        loss_ = tf.reduce_sum(loss_, axis=-1)/dec_lens
        return tf.reduce_mean(loss_)

    def train_step(enc_inp, dec_tar, dec_inp):
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            pred, _ = model(enc_output, dec_inp, dec_hidden, dec_tar)
            loss = loss_function(dec_tar, pred)

        variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables
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
            loss = train_step(batch[0]["enc_input"],
                              batch[1]["dec_target"],
                              batch[1]["dec_input"])
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