import tensorflow as tf
from transformer_pgn.layers.position import positional_encoding


def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
        输出，注意力权重
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True) # transpose_b:将k转置

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    # 将mask放在缩放的向量上，这里的mask为什么要乘以-1e9？？？？
    if mask is not None:
        scaled_attention_logits += (mask*-1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_headers):
        super().__init__()
        self.num_headers = num_headers
        self.d_model = d_model

        assert d_model % num_headers == 0

        self.depth = self.d_model // self.num_headers
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x,(batch_size, -1, self.num_headers, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) #(batch_size, seq_len_q, num_headers, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # token embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(vocab_size, d_model)

    def call(self, x):
        embed_x = self.embedding(x)
        embed_x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # 根据语料长度截取
        embed_x += self.pos_encoding[:, :tf.shape(x)[1], :]
        return embed_x


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] #(batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # 形成三角矩阵
    mask=1-tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_mask(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask