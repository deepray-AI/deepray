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
"""MNIST handwritten digits dataset."""

import sys

import numpy as np
import tensorflow as tf
from absl import flags
from keras.src.utils.data_utils import get_file

from deepray.datasets.datapipeline import DataPipeline
from deepray.utils.horovod_utils import get_rank, get_world_size


class Mnist(DataPipeline):
  def __init__(self, path="mnist.npz"):
    """Loads the MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    More info can be found at the
    [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

    Args:
      path: path where to cache the dataset locally
        (relative to `~/.keras/datasets`).

    Returns:
      Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train**: uint8 NumPy array of grayscale image data with shapes
      `(60000, 28, 28)`, containing the training data. Pixel values range
      from 0 to 255.

    **y_train**: uint8 NumPy array of digit labels (integers in range 0-9)
      with shape `(60000,)` for the training data.

    **x_test**: uint8 NumPy array of grayscale image data with shapes
      (10000, 28, 28), containing the test data. Pixel values range
      from 0 to 255.

    **y_test**: uint8 NumPy array of digit labels (integers in range 0-9)
      with shape `(10000,)` for the test data.

    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    ```

    License:
      Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
      which is a derivative work from original NIST datasets.
      MNIST dataset is made available under the terms of the
      [Creative Commons Attribution-Share Alike 3.0 license.](
      https://creativecommons.org/licenses/by-sa/3.0/)
    """
    super().__init__()

    flags.FLAGS([
      sys.argv[0],
      "--num_train_examples=60000",
    ])

    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    self.path = get_file(
      path,
      origin=origin_folder + "mnist.npz",
      file_hash=(  # noqa: E501
        "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
      ),
    )

  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    with np.load(self.path, allow_pickle=True) as f:
      if is_training:
        image, label = f["x_train"], f["y_train"]
      else:
        image, label = f["x_test"], f["y_test"]

    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(image[..., tf.newaxis] / 255.0, tf.float32), label))
    if self.use_horovod:
      # For multi-host training, we want each hosts to always process the same
      # subset of files.  Each host only sees a subset of the entire dataset,
      # allowing us to cache larger datasets in memory.
      dataset = dataset.shard(num_shards=get_world_size(), index=get_rank())
    dataset = dataset.repeat(epochs).shuffle(10000).batch(batch_size)
    return dataset
