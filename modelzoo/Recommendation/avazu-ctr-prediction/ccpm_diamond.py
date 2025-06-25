import tensorflow as tf
from absl import flags
from tensorflow import keras
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Dense, Flatten
from tensorflow.keras.layers import Layer

from deepray.layers.embedding import DiamondEmbedding
from deepray.utils.data.feature_map import FeatureMap


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
    fold_columns = [
        "hour", "id", "C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
        "app_category", "device_id", "device_ip", "device_model", "device_type", "device_conn_type", "C14", "C15",
        "C16", "C17", "C18", "C19", "C20", "C21"
    ]
    self.embedding_group = DiamondEmbedding(self.feature_map, fold_columns)

  def call(self, inputs, training=None, mask=None):
    features = self.embedding_group(inputs)
    embedding_out = list(features.values())
    s = tf.stack(embedding_out, axis=-1)  # B, number_of_feature, Dim
    # 先扩充channel维度
    # s = tf.expand_dims(s, axis=3)

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
