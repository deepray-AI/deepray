#!/usr/bin/env python
# @Time    : 2021/12/7 8:44 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import yaml
import pandas as pd
import tensorflow as tf
from absl import logging, flags

from deepray.design_patterns import SingletonType
from deepray.utils.horovod_utils import is_main_process


class FeatureMap(metaclass=SingletonType):
  def __init__(self):
    if flags.FLAGS.config_file:
      # Read YAML file
      with open(flags.FLAGS.config_file, encoding="utf-8") as stream:
        try:
          self.yaml_conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
          logging.error(exc)
    if flags.FLAGS.feature_map and tf.io.gfile.exists(flags.FLAGS.feature_map):
      self.feature_map = self.get_summary(
        feature_map=flags.FLAGS.feature_map, black_list=flags.FLAGS.black_list, white_list=flags.FLAGS.white_list
      )
      if is_main_process():
        logging.info("Used features map:")
        print(
          "\n"
          + self.feature_map.loc[
            :, ~self.feature_map.columns.isin(["bucket_boundaries", "vocabulary_list"])
          ].to_markdown()
        )
    else:
      logging.info(f"feature_map file not exists: {flags.FLAGS.feature_map}")
      self.feature_map = None

  def get_summary(self, feature_map, black_list=None, white_list=None):
    with tf.io.gfile.GFile(feature_map, mode="r") as f:
      file_name, file_extension = os.path.splitext(feature_map)
      sep = None
      if file_extension == ".csv":
        sep = ","
      elif file_extension == ".tsv":
        sep = "\t"
      else:
        ValueError(f"Not support format for {f}")
      feature_map = pd.read_csv(
        f,
        sep=sep,
        dtype={
          "code": int,
          "name": "string",
          "dtype": "string",
          "ftype": "string",
          "dim": "uint32",
          "length": float,
          "voc_size": float,
        },
      ).fillna(
        value={
          "code": -1,
          "length": 1.0,
          "voc_size": 0.0,
        }
      )
    if black_list:
      with open(black_list) as f:
        black_feature_list = [feature.strip() for feature in f]
      feature_map = feature_map[~feature_map["name"].isin(black_feature_list)]

    if white_list:
      white_feature_list = []
      if os.path.isfile(white_list):
        print(f"{white_list} is a file.")
        with open(white_list) as f:
          white_feature_list += [feature.strip() for feature in f]
      elif os.path.isdir(white_list):
        print(f"{white_list} is a directory.")
        for used_features in os.listdir(white_list):
          filename = os.path.join(white_list, used_features)
          with open(filename) as f:
            white_feature_list += [feature.strip() for feature in f]
      else:
        print(f"{white_list} is neither a file nor a directory.")

      feature_map = feature_map[feature_map["name"].isin(white_feature_list)]

    # Convert these columns to int if they exist in the DataFrame
    for column in [
      "length",
      "voc_size",
    ]:
      if column in feature_map.columns:
        feature_map[column] = feature_map[column].astype(int)

    return feature_map
