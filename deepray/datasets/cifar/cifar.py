# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""CIFAR100 small images classification dataset."""

import sys
import os
import _pickle as cPickle
from keras import backend

import numpy as np
import tensorflow as tf
from absl import flags
from keras.src.utils.data_utils import get_file
from tensorflow import keras
from deepray.datasets.datapipeline import DataPipeline

flags.FLAGS([
  sys.argv[0],
  "--num_train_examples=60000",
])


class CIFAR(DataPipeline):
  def load_batch(self, fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data.

    Args:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, "rb") as f:
      d = cPickle.load(f, encoding="bytes")
      # decode utf8
      d_decoded = {}
      for k, v in d.items():
        d_decoded[k.decode("utf8")] = v
      d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


class CIFAR10(CIFAR):
  def __init__(self, **kwargs):
    """Loads the CIFAR10 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | airplane    |
    |   1   | automobile  |
    |   2   | bird        |
    |   3   | cat         |
    |   4   | deer        |
    |   5   | dog         |
    |   6   | frog        |
    |   7   | horse       |
    |   8   | ship        |
    |   9   | truck       |

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of grayscale image data with shapes
      `(50000, 32, 32, 3)`, containing the training data. Pixel values range
      from 0 to 255.

    **y_train**: uint8 NumPy array of labels (integers in range 0-9)
      with shape `(50000, 1)` for the training data.

    **x_test**: uint8 NumPy array of grayscale image data with shapes
      `(10000, 32, 32, 3)`, containing the test data. Pixel values range
      from 0 to 255.

    **y_test**: uint8 NumPy array of labels (integers in range 0-9)
      with shape `(10000, 1)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    ```
    """
    super().__init__(**kwargs)
    dirname = "cifar-10-batches-py"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    self.path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=(  # noqa: E501
        "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
      ),
    )

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, *args, **kwargs):
    if is_training:
      num_train_samples = 50000

      x = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
      y = np.empty((num_train_samples,), dtype="uint8")

      for i in range(1, 6):
        fpath = os.path.join(self.path, "data_batch_" + str(i))
        (
          x[(i - 1) * 10000 : i * 10000, :, :, :],
          y[(i - 1) * 10000 : i * 10000],
        ) = self.load_batch(fpath)
    else:
      fpath = os.path.join(self.path, "test_batch")
      x, y = self.load_batch(fpath)

    y = np.reshape(y, (len(y), 1))

    if backend.image_data_format() == "channels_last":
      x = x.transpose(0, 2, 3, 1)

    num_classes = 10

    y = keras.utils.to_categorical(y, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((x / 255.0, y))
    dataset = dataset.repeat(flags.FLAGS.epochs).shuffle(10000).batch(batch_size)
    return dataset


class CIFAR100(CIFAR):
  def __init__(self, label_mode="fine", **kwargs):
    """Loads the CIFAR100 dataset.

    This is a dataset of 50,000 32x32 color training images and
    10,000 test images, labeled over 100 fine-grained classes that are
    grouped into 20 coarse-grained classes. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Args:
      label_mode: one of "fine", "coarse". If it is "fine" the category labels
        are the fine-grained labels, if it is "coarse" the output labels are the
        coarse-grained superclasses.

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of grayscale image data with shapes
      `(50000, 32, 32, 3)`, containing the training data. Pixel values range
      from 0 to 255.

    **y_train**: uint8 NumPy array of labels (integers in range 0-99)
      with shape `(50000, 1)` for the training data.

    **x_test**: uint8 NumPy array of grayscale image data with shapes
      `(10000, 32, 32, 3)`, containing the test data. Pixel values range
      from 0 to 255.

    **y_test**: uint8 NumPy array of labels (integers in range 0-99)
      with shape `(10000, 1)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    ```
    """
    super().__init__(**kwargs)
    if label_mode not in ["fine", "coarse"]:
      raise ValueError(f'`label_mode` must be one of `"fine"`, `"coarse"`. Received: label_mode={label_mode}.')

    dirname = "cifar-100-python"
    origin = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    self.path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=(  # noqa: E501
        "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
      ),
    )
    self.label_mode = label_mode

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, *args, **kwargs):
    if is_training:
      fpath = os.path.join(self.path, "train")

    else:
      fpath = os.path.join(self.path, "test")

    x, y = self.load_batch(fpath, label_key=self.label_mode + "_labels")
    y = np.reshape(y, (len(y), 1))
    if backend.image_data_format() == "channels_last":
      x = x.transpose(0, 2, 3, 1)

    num_classes = 100

    y = keras.utils.to_categorical(y, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((x / 255.0, y))
    dataset = dataset.repeat(flags.FLAGS.epochs).shuffle(10000).batch(batch_size)
    return dataset
