import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from deepray.layers.swin_transformer import SwinTransformer

"""
### Build the model

We put together the Swin Transformer model.
"""

"""
## Model training and evaluation

### Extract and embed patches

We first create 3 layers to help us extract, embed and merge patches from the
images on top of which we will later use the Swin Transformer class we built.
"""


class PatchExtract(layers.Layer):
  def __init__(self, patch_size, **kwargs):
    super().__init__(**kwargs)
    self.patch_size_x = patch_size[0]
    self.patch_size_y = patch_size[0]

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
      images=images,
      sizes=(1, self.patch_size_x, self.patch_size_y, 1),
      strides=(1, self.patch_size_x, self.patch_size_y, 1),
      rates=(1, 1, 1, 1),
      padding="VALID",
    )
    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
  def __init__(self, num_patch, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.num_patch = num_patch
    self.proj = layers.Dense(embed_dim)
    self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

  def call(self, patch):
    pos = tf.range(start=0, limit=self.num_patch, delta=1)
    return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
  def __init__(self, num_patch, embed_dim):
    super().__init__()
    self.num_patch = num_patch
    self.embed_dim = embed_dim
    self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

  def call(self, x):
    height, width = self.num_patch
    _, _, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=(-1, height, width, C))
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = tf.concat((x0, x1, x2, x3), axis=-1)
    x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
    return self.linear_trans(x)


class BaseModel():
  def __init__(self,
               input_shape=(32, 32, 3),
               patch_size=(2, 2),  # 2-by-2 sized patches
               dropout_rate=0.03,  # Dropout rate
               num_heads=8,  # Attention heads
               embed_dim=64,  # Embedding dimension
               num_mlp=256,  # MLP layer size
               qkv_bias=True,  # Convert embedded patches to query, key, and values with a learnable additive value
               window_size=2,  # Size of attention window
               shift_size=1,  # Size of shifting window
               image_dimension=32,  # Initial image size
               ):
    num_patch_x = input_shape[0] // patch_size[0]
    num_patch_y = input_shape[1] // patch_size[1]

    self.crop = layers.RandomCrop(image_dimension, image_dimension)
    self.extract = PatchExtract(patch_size)
    self.embedding = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)
    self.swin0 = SwinTransformer(
      dim=embed_dim,
      num_patch=(num_patch_x, num_patch_y),
      num_heads=num_heads,
      window_size=window_size,
      shift_size=0,
      num_mlp=num_mlp,
      qkv_bias=qkv_bias,
      dropout_rate=dropout_rate,
    )
    self.swin1 = SwinTransformer(
      dim=embed_dim,
      num_patch=(num_patch_x, num_patch_y),
      num_heads=num_heads,
      window_size=window_size,
      shift_size=shift_size,
      num_mlp=num_mlp,
      qkv_bias=qkv_bias,
      dropout_rate=dropout_rate,
    )
    self.merging = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)
    self.input_shape = input_shape

  def __call__(self, num_classes, *args, **kwargs):
    input = layers.Input(self.input_shape)
    x = self.crop(input)
    x = layers.RandomFlip("horizontal")(x)
    x = self.extract(x)
    x = self.embedding(x)
    x = self.swin0(x)
    x = self.swin1(x)
    x = self.merging(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(input, output)


"""
### Train on CIFAR-100

We train the model on CIFAR-100. Here, we only train the model
for 40 epochs to keep the training time short in this example.
In practice, you should train for 150 epochs to reach convergence.
"""
