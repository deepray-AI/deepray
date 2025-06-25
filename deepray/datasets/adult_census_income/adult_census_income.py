import os
import sys

import tensorflow as tf
from absl import flags

from deepray.datasets.datapipeline import DataPipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(dir_path, "feature_map.csv")):
  FLAGS([
    sys.argv[0],
    f"--feature_map={dir_path}/feature_map.csv",
  ])


class Adult_census_income(DataPipeline):
  def __init__(self, data_path="/workspaces/dataset/census/adult.csv"):
    super().__init__()

    LABEL = "income"
    NUMERICAL_FEATURES = self.feature_map.loc[self.feature_map["ftype"] == "Numerical", "name"].tolist()
    CATEGORY_FEATURES = self.feature_map.loc[self.feature_map["ftype"] == "Categorical", "name"].tolist()

    df = pd.read_csv(data_path)
    df[LABEL] = (df[LABEL].apply(lambda x: ">50K" in x)).astype(int)

    for feat in CATEGORY_FEATURES:
      lbe = LabelEncoder()
      df[feat] = lbe.fit_transform(df[feat])
    # Feature normilization
    mms = MinMaxScaler(feature_range=(0, 1))
    df[NUMERICAL_FEATURES] = mms.fit_transform(df[NUMERICAL_FEATURES])

    for feat in CATEGORY_FEATURES:
      print(f"{feat}: {(df[feat].max() + 1,)}")

    self.train_df, self.valid_df = train_test_split(df, test_size=0.2, random_state=1024)

    FLAGS([
      sys.argv[0],
      f"--num_train_examples={self.train_df.shape[0]}",
    ])

  def build_dataset(self, input_file_pattern, batch_size, is_training=True, epochs=1, shuffle=True, *args, **kwargs):
    if is_training:
      target = self.train_df.pop("income")
      dataset = tf.data.Dataset.from_tensor_slices((dict(self.train_df), target))
    else:
      target = self.valid_df.pop("income")
      dataset = tf.data.Dataset.from_tensor_slices((dict(self.valid_df), target))

    dataset = dataset.repeat(epochs).shuffle(10000).batch(batch_size)
    return dataset
