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
r"""Script to process the Imagenet dataset and upload to gcs.

To run the script setup a virtualenv with the following libraries installed.
- `gcloud`: Follow the instructions on
  [cloud SDK docs](https://cloud.google.com/sdk/downloads) followed by
  installing the python api using `pip install gcloud`.
- `google-cloud-storage`: Install with `pip install google-cloud-storage`
- `tensorflow`: Install with `pip install tensorflow`

Once you have all the above libraries setup, you should register on the
[Imagenet website](http://image-net.org/download-images) and download the
ImageNet .tar files. It should be extracted and provided in the format:
- Training images: train/n03062245/n03062245_4620.JPEG
- Validation Images: validation/ILSVRC2012_val_00000001.JPEG
- Validation Labels: synset_labels.txt

To run the script to preprocess the raw dataset as TFRecords and upload to gcs,
run the following command:

```
python3 imagenet_to_gcs.py \
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/IMAGENET_DIR" \
  --raw_data_dir="path/to/imagenet"
```

"""

import math
import os
import random
from typing import Iterable, List, Mapping, Union, Tuple
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

flags.DEFINE_string("gcs_output_path", None, "GCS path for uploading the dataset.")
flags.DEFINE_string("local_scratch_dir", None, "Scratch directory path for temporary files.")
flags.DEFINE_string(
  "raw_data_dir",
  None,
  "Directory path for raw Imagenet dataset. Should have train and validation subdirectories inside it.",
)

LABELS_FILE = "synset_labels.txt"

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = "train"
VALIDATION_DIRECTORY = "validation"


def _check_or_create_dir(directory: str):
  """Checks if directory exists otherwise creates it."""
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)


def _int64_feature(value: Union[int, Iterable[int]]) -> tf.train.Feature:
  """Inserts int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value: Union[bytes, str]) -> tf.train.Feature:
  """Inserts bytes features into Example proto."""
  if isinstance(value, str):
    value = bytes(value, "utf-8")
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(
  filename: str, image_buffer: str, label: int, synset: str, height: int, width: int
) -> tf.train.Example:
  """Builds an Example proto for an ImageNet example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto

  """
  colorspace = "RGB"
  channels = 3
  image_format = "JPEG"

  example = tf.train.Example(
    features=tf.train.Features(
      feature={
        "image/height": _int64_feature(height),
        "image/width": _int64_feature(width),
        "image/colorspace": _bytes_feature(colorspace),
        "image/channels": _int64_feature(channels),
        "image/class/label": _int64_feature(label),
        "image/class/synset": _bytes_feature(synset),
        "image/format": _bytes_feature(image_format),
        "image/filename": _bytes_feature(os.path.basename(filename)),
        "image/encoded": _bytes_feature(image_buffer),
      }
    )
  )
  return example


def _is_png(filename: str) -> bool:
  """Determines if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.

  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  return "n02105855_2933.JPEG" in filename


def _is_cmyk(filename: str) -> bool:
  """Determines if file contains a CMYK JPEG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.

  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  denylist = set([
    "n01739381_1309.JPEG",
    "n02077923_14822.JPEG",
    "n02447366_23489.JPEG",
    "n02492035_15739.JPEG",
    "n02747177_10752.JPEG",
    "n03018349_4028.JPEG",
    "n03062245_4620.JPEG",
    "n03347037_9675.JPEG",
    "n03467068_12171.JPEG",
    "n03529860_11437.JPEG",
    "n03544143_17228.JPEG",
    "n03633091_5218.JPEG",
    "n03710637_5125.JPEG",
    "n03961711_5286.JPEG",
    "n04033995_2932.JPEG",
    "n04258138_17003.JPEG",
    "n04264628_27969.JPEG",
    "n04336792_7448.JPEG",
    "n04371774_5854.JPEG",
    "n04596742_4225.JPEG",
    "n07583066_647.JPEG",
    "n13037406_4650.JPEG",
  ])
  return os.path.basename(filename) in denylist


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format="rgb", quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format="rgb", quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data: bytes) -> tf.Tensor:
    """Converts a PNG compressed image to a JPEG Tensor."""
    return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data: bytes) -> tf.Tensor:
    """Converts a CMYK image to RGB Tensor."""
    return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data: bytes) -> tf.Tensor:
    """Decodes a JPEG image."""
    image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename: str, coder: ImageCoder) -> Tuple[str, int, int]:
  """Processes a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.

  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.

  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, "rb") as f:
    image_data = f.read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    logging.info("Converting PNG to JPEG for %s", filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    logging.info("Converting CMYK to RGB for %s", filename)
    image_data = coder.cmyk_to_rgb(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(
  coder: ImageCoder,
  output_file: str,
  filenames: Iterable[str],
  synsets: Iterable[Union[str, bytes]],
  labels: Mapping[str, int],
):
  """Processes and saves a list of images as TFRecords.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set.
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: map of string to integer; id for all synset labels.

  """
  writer = tf.python_io.TFRecordWriter(output_file)

  for filename, synset in zip(filenames, synsets):
    image_buffer, height, width = _process_image(filename, coder)
    label = labels[synset]
    example = _convert_to_example(filename, image_buffer, label, synset, height, width)
    writer.write(example.SerializeToString())

  writer.close()


def _process_dataset(
  filenames: Iterable[str],
  synsets: Iterable[str],
  labels: Mapping[str, int],
  output_directory: str,
  prefix: str,
  num_shards: int,
) -> List[str]:
  """Processes and saves list of images as TFRecords.

  Args:
    filenames: iterable of strings; each string is a path to an image file.
    synsets: iterable of strings; each string is a unique WordNet ID.
    labels: map of string to integer; id for all synset labels.
    output_directory: path where output files should be created.
    prefix: string; prefix for each file.
    num_shards: number of chunks to split the filenames into.

  Returns:
    files: list of tf-record filepaths created from processing the dataset.

  """
  _check_or_create_dir(output_directory)
  chunksize = int(math.ceil(len(filenames) / num_shards))
  coder = ImageCoder()

  files = []

  for shard in range(num_shards):
    chunk_files = filenames[shard * chunksize : (shard + 1) * chunksize]
    chunk_synsets = synsets[shard * chunksize : (shard + 1) * chunksize]
    output_file = os.path.join(output_directory, "%s-%.5d-of-%.5d" % (prefix, shard, num_shards))
    _process_image_files_batch(coder, output_file, chunk_files, chunk_synsets, labels)
    logging.info("Finished writing file: %s", output_file)
    files.append(output_file)
  return files


def convert_to_tf_records(raw_data_dir: str, local_scratch_dir: str) -> Tuple[List[str], List[str]]:
  """Converts the Imagenet dataset into TF-Record dumps."""

  # Shuffle training records to ensure we are distributing classes
  # across the batches.
  random.seed(0)

  def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order

  # Glob all the training files
  training_files = tf.io.gfile.glob(os.path.join(raw_data_dir, TRAINING_DIRECTORY, "*", "*.JPEG"))

  # Get training file synset labels from the directory name
  training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]
  training_synsets = list(map(lambda x: bytes(x, "utf-8"), training_synsets))

  training_shuffle_idx = make_shuffle_idx(len(training_files))
  training_files = [training_files[i] for i in training_shuffle_idx]
  training_synsets = [training_synsets[i] for i in training_shuffle_idx]

  # Glob all the validation files
  validation_files = sorted(tf.io.gfile.glob(os.path.join(raw_data_dir, VALIDATION_DIRECTORY, "*.JPEG")))

  # Get validation file synset labels from labels.txt
  validation_synsets = tf.gfile.FastGFile(os.path.join(raw_data_dir, LABELS_FILE), "rb").read().splitlines()

  # Create unique ids for all synsets
  labels = {v: k + 1 for k, v in enumerate(sorted(set(validation_synsets + training_synsets)))}

  # Create training data
  logging.info("Processing the training data.")
  training_records = _process_dataset(
    training_files,
    training_synsets,
    labels,
    os.path.join(local_scratch_dir, TRAINING_DIRECTORY),
    TRAINING_DIRECTORY,
    TRAINING_SHARDS,
  )

  # Create validation data
  logging.info("Processing the validation data.")
  validation_records = _process_dataset(
    validation_files,
    validation_synsets,
    labels,
    os.path.join(local_scratch_dir, VALIDATION_DIRECTORY),
    VALIDATION_DIRECTORY,
    VALIDATION_SHARDS,
  )

  return training_records, validation_records


def run(raw_data_dir: str, gcs_output_path: str, local_scratch_dir: str, gcs_upload: bool = False):
  """Runs the ImageNet preprocessing and uploading to GCS.

  Args:
    raw_data_dir: str, the path to the folder with raw ImageNet data.
    gcs_upload: bool, whether or not to upload to GCS.
    gcs_output_path: str, the GCS bucket to write to.
    local_scratch_dir: str, the local directory path.
    client: An optional storage client.

  """

  if gcs_upload and gcs_output_path is None:
    raise ValueError("GCS output path must be provided.")
  elif gcs_upload and not gcs_output_path.startswith("gs://"):
    raise ValueError("GCS output path must start with gs://")

  if raw_data_dir is None:
    raise AssertionError(
      "The ImageNet download path is no longer supported. Please download "
      "and extract the .tar files manually and provide the `raw_data_dir`."
    )

  # Convert the raw data into tf-records
  training_records, validation_records = convert_to_tf_records(
    raw_data_dir=raw_data_dir, local_scratch_dir=local_scratch_dir
  )


def main(_):
  run(raw_data_dir=FLAGS.raw_data_dir, gcs_output_path=FLAGS.gcs_output_path, local_scratch_dir=FLAGS.local_scratch_dir)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  app.run(main)
