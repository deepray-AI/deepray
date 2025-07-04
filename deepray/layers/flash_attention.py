import math
import tensorflow as tf
from deepray.custom_ops.flash_attn import gen_flash_attention_ops


class FlashAttentionLayer(tf.keras.layers.Layer):
  """
  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra
  Paper: https://arxiv.org/abs/2205.14135
  https://github.com/HazyResearch/flash-attention

  FlashAttention currently supports:
    Turing, Ampere, Ada, or Hopper GPUs (e.g., H100, A100, RTX 3090, T4).
    fp16 and bf16 (bf16 requires Ampere, Ada, or Hopper GPUs).
    Head dimensions that are multiples of 8, up to 128 (e.g., 8, 16, 24, ...).
    Head dim > 64 backward requires A100 or H100.

  Tensor shapes:
    query: [BatchSize(B), SequenceLength(S), NumHeads(H), DimHead(K)]
    key: [BatchSize(B), SequenceLength(S), NumHeads(H), DimHead(K)]
    value:[BatchSize(B), SequenceLength(S), NumHeads(H), DimHead(K)]
    output: 3d tensor, (BatchSize, SequenceLength, NumHeads*DimHead)
  """

  def __init__(
    self,
    max_query_length,
    max_key_length,
    num_heads,
    dim_head,
    dropout_rate=0.0,
    is_causal=False,
    num_splits=1,
    dtype=tf.half,
    **kwargs,
  ):
    """vim
    Args:
      max_query_length: maximum query sequence length
      max_key_length: maximum key sequence length
      num_heads: number of heads
      dim_head: Head dimensions that are multiples of 8,
                up to 128 (e.g., 8, 16, 24, ..., 128).
                Head dim > 64 backward requires A100 or H100.
      dropout_rate(float): Dropout probability;
                           if greater than 0.0, dropout is applied
      is_causal(bool): If true,
                       assumes causal attention masking and errors
                       if both attn_mask and is_causal
      num_splits(int): How many SMs per attention matrix.
                       SelfAttention default is 1
    """
    super(FlashAttentionLayer, self).__init__(**kwargs)
    self.num_heads = num_heads
    if dim_head % 8 != 0:
      raise ValueError(
        "Head dimensions that are multiples of 8,"
        "up to 128 (e.g., 8, 16, 24, ..., 128)."
        "Head dim > 64 backward requires A100 or H100."
        "You set to %s" % dim_head
      )
    self.dim_head = dim_head
    self.dropout_rate = dropout_rate
    self.is_causal = is_causal
    self.softmax_scale = 1.0 / math.sqrt(self.dim_head)
    self.num_splits = num_splits
    self.max_query_length = max_query_length
    self.max_key_length = max_key_length
    self.fa_type = dtype

  def call(self, query, key, value, mask=None, **kwargs):
    """
    Args:
        query: Query tensor, dtype: `tf.bfloat16, tf.float16`
        key: Key tensor, dtype: `tf.bfloat16, tf.float16`
        value: Value tensor, dtype: `tf.bfloat16, tf.float16`
        mask: attention mask, dtyoe: `tf.float32`
    Tensor shapes:
        inputs[0] [BatchSize(B), SequenceLength(S), NumHeads(H), DimHead(K)]
                HiddenSize = NumHeads(H) x DimHead(K)
    Returns:
         shape=[BatchSize(B), SequenceLength(S), NumHeads(H), DimHead(K)]
    """
    # input query: shape=(B, S, H, K)
    # input key: shape=(B, S, H, K)
    # input value: shape=(B, S, H, K)

    # query, key, value: shape=[B x S, H, K]

    if mask is not None:

      def calculate(mask):
        cu_seqlens = tf.reduce_sum(mask, -1)
        max_length = tf.reduce_max(cu_seqlens)
        max_length = tf.cast(max_length, tf.int32)
        cu_seqlens = tf.cast(cu_seqlens, tf.float32)
        cu_seqlens = tf.cumsum(cu_seqlens, axis=0)
        cu_seqlens = tf.cast(cu_seqlens, tf.int32)
        cu_seqlens = tf.concat([tf.constant([0]), cu_seqlens], 0)
        return cu_seqlens, max_length

      query_mask = tf.reduce_sum(query, [2, 3])
      query_mask = tf.ones_like(query_mask, dtype=tf.int32)
      cu_seqlens_q, max_len_q = calculate(query_mask)
      mask = tf.cast(mask, tf.int32)
      cu_seqlens_k, max_len_k = calculate(mask)
      mask = tf.reshape(mask, [-1])
      indices = tf.where(tf.not_equal(mask, 0))

      query = tf.reshape(query, [-1, self.num_heads, self.dim_head])
      key = tf.reshape(key, [-1, self.num_heads * self.dim_head])
      key = tf.gather(key, indices)
      key = tf.reshape(key, [-1, self.num_heads, self.dim_head])
      value = tf.reshape(value, [-1, self.num_heads * self.dim_head])
      value = tf.gather(value, indices)
      value = tf.reshape(value, [-1, self.num_heads, self.dim_head])
    else:
      query = tf.reshape(query, [-1, self.num_heads, self.dim_head])
      key = tf.reshape(key, [-1, self.num_heads, self.dim_head])
      value = tf.reshape(value, [-1, self.num_heads, self.dim_head])

    if mask is None:
      cu_seqlens_k = tf.constant([i * self.max_key_length for i in range(512 + 1)])
      cu_seqlens_q = tf.constant([i * self.max_query_length for i in range(512 + 1)])
      max_len_q = self.max_query_length
      max_len_k = self.max_key_length
    return_softmax = False
    zero_tensors = False
    # [B X S, H, K] => [B X S, H, K]
    # The attention of the recommendation system currently does not require causal
    attn_weight = gen_flash_attention_ops.fmha_forward(
      query,
      key,
      value,
      cu_seqlens_q,
      cu_seqlens_k,
      max_len_q,
      max_len_k,
      self.dropout_rate,
      self.softmax_scale,
      zero_tensors,
      self.is_causal,
      return_softmax,
      self.num_splits,
    )

    # output attn_weight: [B, S, H, K]
    attn_weight = tf.reshape(attn_weight, [-1, self.max_query_length, self.num_heads, self.dim_head])
    tf.logging.info("self attention output shape {}".format(attn_weight))
    return attn_weight

  def compute_output_shape(self, input_shape):
    return input_shape[0][:2] + (self.num_heads, self.dim_head)

  def get_config(self):
    config = {"dropout_rate": self.dropout_rate}
    base_config = super(FlashAttentionLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
