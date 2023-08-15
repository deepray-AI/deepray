import math

import tensorflow as tf
from absl import flags
from tensorflow import keras
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Dense, Flatten
from tensorflow.keras.layers import Layer
from tensorflow_recommenders_addons import dynamic_embedding as de

from deepray.layers.bucketize import Hash
from deepray.layers.embedding import DynamicEmbedding
from deepray.utils.data.feature_map import FeatureMap

FLAGS = flags.FLAGS


class KMaxPooling(Layer):

  def __init__(self, k, dim):
    super(KMaxPooling, self).__init__()
    self.k = k
    self.dim = dim

  def forward(self, X):
    index = X.topk(self.k, dim=self.dim)[1].sort(dim=self.dim)[0]
    output = X.gather(self.dim, index)
    return output


class CCPM(keras.Model):
  """
  """

  def __init__(self, conv_kernel_width=(6, 5, 3), conv_filters=(4, 4, 4), embed_reg=1e-6, *args, **kwargs):
    super(CCPM, self).__init__()
    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map
    self.sparse_feat_len = self.feature_map[(self.feature_map['ftype'] == "Categorical")].shape[0]
    self.conv_len = len(conv_filters)  # 卷积层数

    # KMaxPooling
    self.p = []
    for i in range(1, self.conv_len + 1):
      if i < self.conv_len:
        k = max(1, int((1 - pow(i / self.conv_len, self.conv_len - i)) * self.sparse_feat_len))
        self.p.append(k)
      else:
        self.p.append(3)
    self.max_pooling_list = [KMaxPooling(k, dim=2) for k in self.p]

    self.padding_list = [ZeroPadding2D(padding=(0, conv_kernel_width[i] - 1)) for i in range(self.conv_len)]
    self.conv_list = [
        Conv2D(filters=conv_filters[i], kernel_size=(1, conv_kernel_width[i])) for i in range(self.conv_len)
    ]

    self.flatten = Flatten()
    self.dense = Dense(units=1)

  def build(self, input_shape):
    self.embedding_layers = {}
    self.hash_long_kernel = {}
    for name, dim, voc_size, hash_size, dtype in self.feature_map[(self.feature_map['ftype'] == "Categorical")][[
        "name", "dim", "voc_size", "hash_size", "dtype"
    ]].values:
      if not math.isnan(hash_size):
        self.hash_long_kernel[name] = Hash(int(hash_size))
        voc_size = int(hash_size)

      if not FLAGS.use_horovod:
        self.embedding_layers[name] = DynamicEmbedding(
            embedding_size=dim,
            # mini_batch_regularizer=l2(feature.emb_reg_l2),
            # mask_value=feature.default_value,
            key_dtype=dtype,
            value_dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform(),
            name='embedding_' + name,
            # init_capacity=800000,
            kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver())
        )
      else:
        import horovod.tensorflow as hvd

        gpu_device = ["GPU:0"]
        mpi_size = hvd.size()
        mpi_rank = hvd.rank()
        self.embedding_layers[name] = de.keras.layers.HvdAllToAllEmbedding(
            # mpi_size=mpi_size,
            embedding_size=dim,
            key_dtype=dtype,
            value_dtype=tf.float32,
            initializer=tf.keras.initializers.GlorotUniform(),
            devices=gpu_device,
            init_capacity=800000,
            name='embedding_' + name,
            kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))
        )

      # self.embedding_layers[name] = Embedding(
      #   embedding_dim=dim,
      #   vocabulary_size=voc_size,
      #   name='embedding_' + name)

  def call(self, inputs, training=None, mask=None):
    embedding_out = []
    for name, hash_size in self.feature_map[(self.feature_map['ftype'] == "Categorical")][["name", "hash_size"]].values:
      input_tensor = inputs[name]
      if not math.isnan(hash_size):
        input_tensor = self.hash_long_kernel[name](input_tensor)
      input_tensor = self.embedding_layers[name](input_tensor)
      embedding_out.append(input_tensor)

    s = tf.stack(embedding_out, axis=-1)  # B, Dim, number_of_feature
    # 先扩充channel维度
    s = tf.expand_dims(s, axis=3)

    for i in range(self.conv_len):
      # padding
      s = self.padding_list[i](s)
      # conv , batch,embed_dim,width,channel
      r = self.conv_list[i](s)
      s = self.max_pooling_list[i](r)
      s = relu(s)

    outputs = self.dense(self.flatten(s))
    outputs = sigmoid(outputs)
    return outputs
