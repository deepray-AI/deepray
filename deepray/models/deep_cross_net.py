import tensorflow as tf
import tensorflow.keras.layers as layers

from deepray.layers.cross import CrossNet

categorical_columns = []
for feature, vocabulary_list in CATEGORIES_Dict.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
    key=feature, vocabulary_list=vocabulary_list, num_oov_buckets=1
  )
  categorical_columns.append(tf.feature_column.embedding_column(cat_col, 16))

numberical_columns = []
for feature in NUMERICAL_FEATURES:
  num_col = tf.feature_column.numeric_column(feature, default_value=0, dtype=tf.dtypes.float32, normalizer_fn=scale_fn)
  numberical_columns.append(num_col)

categorical_columns_layer = tf.keras.layers.DenseFeatures(categorical_columns)
numberical_columns_layer = tf.keras.layers.DenseFeatures(numberical_columns)


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__(name="my_model")
    self.input_1 = categorical_columns_layer
    self.input_2 = numberical_columns_layer
    self.deep1 = layers.Dense(400)
    self.deep2 = layers.Dense(400)
    self.relu1 = layers.ReLU()
    self.relu2 = layers.ReLU()
    self.dropout1 = layers.Dropout(0.1)
    self.dropout2 = layers.Dropout(0.1)
    self.norm1 = layers.BatchNormalization()
    self.norm2 = layers.BatchNormalization()
    self.sigmoid = layers.Dense(1, activation="sigmoid")
    self.cross = CrossNet(layer_num=3)

  def call(self, inputs, training=None):
    c = self.input_1(inputs)
    n = self.input_2(inputs)
    # n = tf.reshape(tf.reshape(n, [-1, 13, 1]) * tf.ones([16], tf.float32), [-1, 16 * 13])
    l = tf.keras.layers.concatenate([c, n])
    l_shape = l.shape

    # cross
    x = self.cross(l)

    # deep
    y = self.deep1(l)
    y = self.norm1(y, training=training)
    y = self.relu1(y)
    y = self.dropout1(y)
    y = self.deep2(l)
    y = self.norm2(y, training=training)
    y = self.relu2(y)
    y = self.dropout2(y)
    z = tf.keras.layers.concatenate([x, y])

    return self.sigmoid(z)
