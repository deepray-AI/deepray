#!/usr/bin/env python
# @Time    : 2021/12/7 8:44 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os

import horovod.tensorflow as hvd
import pandas as pd
import tensorflow as tf
from absl import logging, flags

from deepray.design_patterns import SingletonType

FLAGS = flags.FLAGS


class FeatureMap(metaclass=SingletonType):

  def __init__(self, feature_map, black_list=None, white_list=None):
    # Read YAML file
    # with open(os.path.join(os.path.dirname(__file__), feature_file), encoding="utf-8") as stream:
    #   try:
    #     self.conf = yaml.safe_load(stream)
    #   except yaml.YAMLError as exc:
    #     logging.error(exc)
    self._feature_file = feature_map
    self._black_list = black_list
    self._white_list = white_list if white_list else FLAGS.white_list
    self.feature_map = self.get_summary()
    if (not FLAGS.use_horovod or hvd.rank() == 0) and self.feature_map is not None:
      logging.info(
          "\n" +
          self.feature_map.loc[:,
                               ~self.feature_map.columns.isin(["bucket_boundaries", "vocabulary_list"])].to_markdown()
      )

  def get_summary(self):
    if not tf.io.gfile.exists(self._feature_file):
      logging.info(f"File not exists: {self._feature_file}")
      return None
    with tf.io.gfile.GFile(self._feature_file, mode="r") as f:
      file_name, file_extension = os.path.splitext(self._feature_file)
      if file_extension == ".csv":
        feature_map = pd.read_csv(
            f,
            dtype={
                "code": int,
                "name": "string",
                "dtype": "string",
                "ftype": "string",
                "dim": "uint32",
                "length": float,
                "voc_size": float,
                "lr": "float32",
                "optimizer": "string",
                "storage_type": "string",
                "composition_size": "string",
                "ev_filter": "string",
            },
        ).fillna(
            value={
                "code": -1,
                "length": 1.0,
                "voc_size": 0.0,
                "lr": 0.0,
                "optimizer": "",
                "storage_type": "",
                "composition_size": "",
                "ev_filter": "",
            }
        )
      elif file_extension == ".tsv":
        feature_map = pd.read_csv(
            f,
            sep='\t',
            header=None,
            usecols=[i for i in range(13)],
            names=[
                "code", "name", "length", "dtype", "gpercentile", "gcov", "geva", "bpercentile", "bcov", "beva",
                "def_valu", "fea_tag", "dim"
            ],
            dtype={
                "code": "string",
                "name": "string",
                "length": float,
                "dtype": "string",
                # "ftype": "string",
                "gpercentile": "string",
                "geva": "string",
                "bpercentile": "string",
                "bcov": "string",
                "beva": "string",
                "def_valu": "string",
                "fea_tag": "string"
            },
        ).fillna(
            value={
                # "code": "",
                # "name": "",
                "length": 1.0,
                # "dtype": "",
                # "gpercentile": "",
                # "fea_geva": "",
                # "fea_bpercentile": "",
                # "fea_bcov": "",
                # "fea_beva": "",
                "def_valu": "",
                # "fea_tag": ""
            }
        )
      else:
        ValueError(f"Not support format for {f}")
    if self._black_list:
      with open(self._black_list) as f:
        black_feature_list = [feature.strip() for feature in f]
      feature_map = feature_map[~feature_map["name"].isin(black_feature_list)]

    if self._white_list:
      white_feature_list = []
      if os.path.isfile(self._white_list):
        print(f'{self._white_list} is a file.')
        with open(self._white_list) as f:
          white_feature_list += [feature.strip() for feature in f]
      elif os.path.isdir(self._white_list):
        print(f'{self._white_list} is a directory.')
        for used_features in os.listdir(self._white_list):
          filename = os.path.join(self._white_list, used_features)
          with open(filename) as f:
            white_feature_list += [feature.strip() for feature in f]
      else:
        print(f'{self._white_list} is neither a file nor a directory.')

      feature_map = feature_map[feature_map["name"].isin(white_feature_list)]

    # Convert these columns to int if they exist in the DataFrame
    for column in [
        'length',
        'voc_size',
        # 'composition_size'
    ]:
      if column in feature_map.columns:
        feature_map[column] = feature_map[column].astype(int)

    return feature_map
