#!/usr/bin/env python
# @Time    : 2021/8/9 3:41 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>

import abc
import os
import urllib.request
from enum import Enum

import pandas as pd
import tensorflow as tf
from absl import flags, logging

import deepray
from deepray.utils.data.feature_map import FeatureMap

pd.set_option("max_colwidth", 800)
pd.set_option("max_colwidth", 800)
pd.set_option("display.max_rows", None)
os.environ["TF_CPP_VMODULE"] = "dataset=1"  # let hybridbackend logs reading filename.

ROOT_PATH = os.path.dirname(deepray.__file__)

IS_TRAINING = Enum("is_training", ("Train", "Valid", "Test"))


class DataPipeline(object):
  def __init__(self, context: tf.distribute.InputContext = None, **kwargs):
    # super().__init__(**kwargs)
    self.built = False
    self.use_horovod = flags.FLAGS.use_horovod
    self.context = context
    self.feature_map = FeatureMap().feature_map
    # self.conf = Foo(flags.FLAGS.conf_file).conf
    self.url = None
    self.prebatch_size = kwargs.get("prebatch_size", None)

  @abc.abstractmethod
  def __len__(self):
    pass

  @abc.abstractmethod
  def build(self):
    raise NotImplementedError("build: not implemented!")

  @classmethod
  def read_list_from_file(cls, filename):
    file_list = tf.io.gfile.glob(filename)
    if len(file_list) <= 0:
      logging.error(f"glob pattern {filename} did not match any files, process exits.")
      exit(0)
    # assert len(file_list) > 0, \
    #     'glob pattern {} did not match any files'.format(filename)
    return file_list

  @abc.abstractmethod
  def parser(self, record):
    """
    must be defined in subclass
    """
    raise NotImplementedError("build: not implemented!")

  @abc.abstractmethod
  def build_dataset(
    self, batch_size, input_file_pattern=None, is_training=True, epochs=1, shuffle=False, *args, **kwargs
  ):
    """
    must be defined in subclass
    """
    raise NotImplementedError("build_dataset: not implemented!")

  def __call__(self, batch_size=None, input_file_pattern=None, is_training=True, *args, **kwargs):
    """Gets a closure to create a dataset."""
    return self.build_dataset(
      batch_size=self.context.get_per_replica_batch_size(batch_size) if self.context else batch_size,
      input_file_pattern=input_file_pattern,
      is_training=is_training,
      epochs=1,
      *args,
      **kwargs,
    )

  def maybe_download(self, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
      filename, _ = urllib.request.urlretrieve(self.url, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
      print("Found and verified", filename)
    else:
      print(statinfo.st_size)
      raise Exception("Failed to verify " + self.url + ". Can you get to it with a browser?")
    return filename

  def _dataset_options(self, input_files):
    options = tf.data.Options()

    # When `input_files` is a path to a single file or a list
    # containing a single path, disable auto sharding so that
    # same input file is sent to all workers.
    if isinstance(input_files, str) or len(input_files) == 1:
      options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    else:
      """ Constructs tf.data.Options for this dataset. """
      options.experimental_optimization.parallel_batch = True
      options.experimental_slack = True
      options.threading.max_intra_op_parallelism = 1
      options.experimental_optimization.map_parallelization = True

    return options

  def train_test_split(self, arrays, test_size=0.33, shuffle=False):
    from sklearn.model_selection import train_test_split

    random_state = flags.FLAGS.random_seed if flags.FLAGS.random_seed else 1024
    X_train, X_test = train_test_split(arrays, test_size=test_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test
