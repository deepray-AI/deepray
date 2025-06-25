# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fashion-MNIST dataset."""

import gzip
import os
import sys

import numpy as np
import tensorflow as tf
from absl import flags
from keras.src.utils.data_utils import get_file

from deepray.datasets.datapipeline import DataPipeline

flags.FLAGS([
  sys.argv[0],
  "--num_train_examples=60000",
])


class FashionMNIST(DataPipeline):
  def __init__(self):
    """Loads the Fashion-MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
    along with a test set of 10,000 images. This dataset can be used as
    a drop-in replacement for MNIST.

    The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | T-shirt/top |
    |   1   | Trouser     |
    |   2   | Pullover    |
    |   3   | Dress       |
    |   4   | Coat        |
    |   5   | Sandal      |
    |   6   | Shirt       |
    |   7   | Sneaker     |
    |   8   | Bag         |
    |   9   | Ankle boot  |

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of grayscale image data with shapes
      `(60000, 28, 28)`, containing the training data.

    **y_train**: uint8 NumPy array of labels (integers in range 0-9)
      with shape `(60000,)` for the training data.

    **x_test**: uint8 NumPy array of grayscale image data with shapes
      (10000, 28, 28), containing the test data.

    **y_test**: uint8 NumPy array of labels (integers in range 0-9)
      with shape `(10000,)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    ```

    License:
      The copyright for Fashion-MNIST is held by Zalando SE.
      Fashion-MNIST is licensed under the [MIT license](
      https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

    """
    super().__init__()
    dirname = os.path.join("datasets", "fashion-mnist")
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
      "train-labels-idx1-ubyte.gz",
      "train-images-idx3-ubyte.gz",
      "t10k-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz",
    ]

    self.paths = []
    for fname in files:
      self.paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))

  def __len__(self):
    pass

  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    if is_training:
      with gzip.open(self.paths[0], "rb") as lbpath:
        y = np.frombuffer(lbpath.read(), np.uint8, offset=8)

      with gzip.open(self.paths[1], "rb") as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y), 28, 28)
    else:
      with gzip.open(self.paths[2], "rb") as lbpath:
        y = np.frombuffer(lbpath.read(), np.uint8, offset=8)

      with gzip.open(self.paths[3], "rb") as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y), 28, 28)

    dataset = tf.data.Dataset.from_tensor_slices((
      tf.cast(x[..., tf.newaxis] / 255.0, tf.float32),
      tf.cast(y, tf.int64),
    ))
    dataset = dataset.repeat(flags.FLAGS.epochs).shuffle(10000).batch(batch_size)
    return dataset
