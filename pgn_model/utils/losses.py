import tensorflow as tf


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def loss_function(real, outputs, padding_mask, cov_loss_wt, use_coverage):
    pred = outputs["logits"]
    attn_dists = outputs["attentions"]
    if use_coverage:
        loss = pgn_log_loss_function(real, pred, padding_mask) + cov_loss_wt*_coverage_loss(attn_dists, padding_mask)
        return loss
    else:
        return seq2seq_loss_function(real, pred, padding_mask)


def seq2seq_loss_function(real, pred, padding_mask):
    loss = 0
    for t in range(real.shape[1]):
        loss_=loss_object(real[:,t], pred[:, t])
        mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)
        loss += loss_
    return loss/real.shape[1]


def pgn_log_loss_function(real, final_dists, padding_mask):
    loss_per_step=[]
    batch_nums=tf.range(0, limit=real.shape[0])
    for dec_step, dist in enumerate(final_dists):
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)
        gold_probs=tf.gather_nd(dist, indices)
        losses = -tf.math.log(gold_probs)
        loss_per_step.append(losses)
    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss

def _mask_and_avg(values, padding_mask):
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)
    values_per_step=[v*padding_mask[:, dec_step] for dec_step,v in enumerate(values)]
    values_per_ex=sum(values_per_step)/dec_lens
    return tf.reduce_mean(values_per_ex)

def _coverage_loss(attn_dists, padding_mask):
    coverage = tf.zeros_like(attn_dists[0])
    covlosses=[]
    for a in attn_dists:
        covlosse = tf.reduce_sum(tf.minimum(a, coverage),axis=1)
        covlosses.append(covlosse)
        coverage += a
    coverage_loss=_mask_and_avg(covlosses, padding_mask)
    return coverage_loss