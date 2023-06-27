#!/usr/bin/env python
# @Time    : 2021/8/9 3:41 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>

import abc
import multiprocessing
import os
import urllib.request
from enum import Enum

import pandas as pd
import tensorflow as tf
from absl import flags
from absl import logging

import deepray
from deepray.utils.data.feature_map import FeatureMap

pd.set_option("max_colwidth", 800)
pd.set_option("max_colwidth", 800)
pd.set_option("display.max_rows", None)
os.environ["TF_CPP_VMODULE"] = "dataset=1"  # let hybridbackend logs reading filename.

ROOT_PATH = os.path.dirname(deepray.__file__)

FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel_parse", multiprocessing.cpu_count(), "Number of parallel parsing")
flags.DEFINE_integer("shuffle_buffer", None, "Size of shuffle buffer")
flags.DEFINE_integer("prefetch_buffer", 16, "Size of prefetch buffer")
flags.DEFINE_integer("parallel_reads_per_file", None, "Number of parallel reads per file")
flags.DEFINE_integer("interleave_cycle", 16, "Number of interleaved inputs")
flags.DEFINE_integer("interleave_block", 2, "Number of interleaved block_length inputs")
flags.DEFINE_float("neg_sample_rate", 0.0, "")
flags.DEFINE_string("conf_file", os.getcwd() + "/conf/dp.yaml", "configuration in file.")

IS_TRAINING = Enum('is_training', ('Train', 'Valid', 'Test'))


class DataPipeLine:

  def __init__(self,
               use_horovod=False,
               context: tf.distribute.InputContext = None,
               **kwargs):
    self.use_horovod = use_horovod
    self.context = context
    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map
    # self.conf = Foo(FLAGS.conf_file).conf
    self.url = None

  @abc.abstractmethod
  def __len__(self):
    pass

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
  def build_dataset(self, input_file_pattern,
                    batch_size,
                    is_training=True,
                    prebatch_size=0,
                    epochs=1,
                    shuffle=True,
                    *args, **kwargs):
    """
    must be defined in subclass
    """
    raise NotImplementedError("build_dataset: not implemented!")

  def __call__(self, input_file_pattern=None, batch_size=None, is_training=None, prebatch_size=0, *args, **kwargs):
    """Gets a closure to create a dataset."""

    return self.build_dataset(
      input_file_pattern=input_file_pattern,
      batch_size=self.context.get_per_replica_batch_size(batch_size) if self.context else batch_size,
      is_training=is_training,
      epochs=FLAGS.epochs,
      *args, **kwargs
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
