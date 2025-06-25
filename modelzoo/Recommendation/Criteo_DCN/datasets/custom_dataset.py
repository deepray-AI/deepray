# -*- coding:utf-8 -*-
import tensorflow as tf
from absl import flags

from deepray.datasets.parquet_pipeline.ali_parquet_dataset import ParquetPipeline


class CustomParquetPipeline(ParquetPipeline):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.column_names = list(self.feature_map["name"])
    self._label_name = self.feature_map[(self.feature_map["ftype"] == "Label")].iloc[0]["name"]
    self.numerical_features = self.feature_map[(self.feature_map["ftype"] == "Numerical")]["name"].tolist()
    self.categorical_features = self.feature_map[(self.feature_map["ftype"] == "Categorical")]["name"].tolist()

  def parser(self, record):
    if self.numerical_features:
      record["dense_features"] = tf.concat(
        [tf.reshape(record.pop(key), [-1, 1]) for key in self.numerical_features], axis=1
      )
    for fea in self.categorical_features:
      record[fea] = tf.reshape(record[fea], [-1])

    target = record.pop(self._label_name)
    return record, target
