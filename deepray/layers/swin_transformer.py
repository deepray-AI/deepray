import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from deepray.layers.attention import WindowAttention

"""
## The complete Swin Transformer model

Finally, we put together the complete Swin Transformer by replacing the standard multi-head
attention (MHA) with shifted windows attention. As suggested in the
original paper, we create a model comprising of a shifted window-based MHA
layer, followed by a 2-layer MLP with GELU nonlinearity in between, applying
`LayerNormalization` before each MSA layer and each MLP, and a residual
connection after each of these layers.

Notice that we only create a simple MLP with 2 Dense and
2 Dropout layers. Often you will see models using ResNet-50 as the MLP which is
quite standard in the literature. However in this paper the authors use a
2-layer MLP with GELU nonlinearity in between.
"""


def window_partition(x, window_size):
  _, height, width, channels = x.shape
  patch_num_y = height // window_size
  patch_num_x = width // window_size
  x = tf.reshape(
    x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
  )
  x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
  windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
  return windows


def window_reverse(windows, window_size, height, width, channels):
  patch_num_y = height // window_size
  patch_num_x = width // window_size
  x = tf.reshape(
    windows,
    shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
  )
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, shape=(-1, height, width, channels))
  return x


class DropPath(layers.Layer):
  def __init__(self, drop_prob=None, **kwargs):
    super().__init__(**kwargs)
    self.drop_prob = drop_prob

  def call(self, x):
    input_shape = tf.shape(x)
    batch_size = input_shape[0]
    rank = x.shape.rank
    shape = (batch_size,) + (1,) * (rank - 1)
    random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
    path_mask = tf.floor(random_tensor)
    output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
    return output


class SwinTransformer(layers.Layer):
  def __init__(
      self,
      dim,
      num_patch,
      num_heads,
      window_size=7,
      shift_size=0,
      num_mlp=1024,
      qkv_bias=True,
      dropout_rate=0.0,
      **kwargs,
  ):
    super().__init__(**kwargs)

    self.dim = dim  # number of input dimensions
    self.num_patch = num_patch  # number of embedded patches
    self.num_heads = num_heads  # number of attention heads
    self.window_size = window_size  # size of window
    self.shift_size = shift_size  # size of window shift
    self.num_mlp = num_mlp  # number of MLP nodes

    self.norm1 = layers.LayerNormalization(epsilon=1e-5)
    self.attn = WindowAttention(
      dim,
      window_size=(self.window_size, self.window_size),
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      dropout_rate=dropout_rate,
    )
    self.drop_path = DropPath(dropout_rate)
    self.norm2 = layers.LayerNormalization(epsilon=1e-5)

    self.mlp = keras.Sequential(
      [
        layers.Dense(num_mlp, activation=tf.keras.activations.gelu),
        layers.Dropout(dropout_rate),
        layers.Dense(dim),
        layers.Dropout(dropout_rate),
      ]
    )

    if min(self.num_patch) < self.window_size:
      self.shift_size = 0
      self.window_size = min(self.num_patch)

  def build(self, input_shape):
    if self.shift_size == 0:
      self.attn_mask = None
    else:
      height, width = self.num_patch
      h_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      w_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
      )
      mask_array = np.zeros((1, height, width, 1))
      count = 0
      for h in h_slices:
        for w in w_slices:
          mask_array[:, h, w, :] = count
          count += 1
      mask_array = tf.convert_to_tensor(mask_array)

      # mask array to windows
      mask_windows = window_partition(mask_array, self.window_size)
      mask_windows = tf.reshape(
        mask_windows, shape=[-1, self.window_size * self.window_size]
      )
      attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
        mask_windows, axis=2
      )
      attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
      attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
      self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

  def call(self, x):
    height, width = self.num_patch
    _, num_patches_before, channels = x.shape
    x_skip = x
    x = self.norm1(x)
    x = tf.reshape(x, shape=(-1, height, width, channels))
    if self.shift_size > 0:
      shifted_x = tf.roll(
        x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
      )
    else:
      shifted_x = x

    x_windows = window_partition(shifted_x, self.window_size)
    x_windows = tf.reshape(
      x_windows, shape=(-1, self.window_size * self.window_size, channels)
    )
    attn_windows = self.attn(x_windows, mask=self.attn_mask)

    attn_windows = tf.reshape(
      attn_windows, shape=(-1, self.window_size, self.window_size, channels)
    )
    shifted_x = window_reverse(
      attn_windows, self.window_size, height, width, channels
    )
    if self.shift_size > 0:
      x = tf.roll(
        shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
      )
    else:
      x = shifted_x

    x = tf.reshape(x, shape=(-1, height * width, channels))
    x = self.drop_path(x)
    x = x_skip + x
    x_skip = x
    x = self.norm2(x)
    x = self.mlp(x)
    x = self.drop_path(x)
    x = x_skip + x
    return x
