#!/usr/bin/env python
# @Time    : 2023/9/26 2:50 PM
# @Author  : Hailin.Fu
# @license : Copyright(C),  <hailin.fu@>
import os
import sys

from absl import app, flags

from arsenal_parquet_dataset.custom_dataset import CustomArsenalParquetDataset
from custom_model import MatchModel


def runner(argv=None):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  if len(argv) <= 1:
    argv = [
        sys.argv[0],
        "--batch_size=8",
        "--dataset=gs_rank_e2e",
        "--epochs=1",
        "--run_eagerly=False",
        "--use_dynamic_embedding=True",
        f"--feature_map={dir_path}/feature_map.csv",
        "--model_dir=/code/fuhailin/arsenal_tfra_accelerate/gs_rank_tfra_accelerate_test/latest",
    ]
  if argv:
    FLAGS(argv, known_only=True)

  data_pipe = CustomArsenalParquetDataset(dataset_name=FLAGS.dataset, partitions=[{'ds': "2023-09-06"}])
  test_files_list = data_pipe.get_dataset_files()
  test_ds = data_pipe(input_file_pattern=test_files_list[-1], batch_size=FLAGS.batch_size)
  model = MatchModel(pretrain=FLAGS.pretrain, training=False).build()
  model.load_weights(os.path.join(FLAGS.model_dir, "variables/variables"))

  for x, y in test_ds.take(1):
    x = x.pop("lid")
    preds = model(x)
    print(preds)


if __name__ == "__main__":
  app.run(runner)
