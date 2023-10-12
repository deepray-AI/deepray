# -*- coding: utf-8 -*-
import tempfile
import os
from absl import app, flags
import tensorflow as tf
from dcn_v2 import Ranking
from deepray.utils.export import export_to_savedmodel
from deepray.utils.data.file_io import recursive_copy
from deepray.datasets.criteo import CriteoTsvReader

FLAGS = flags.FLAGS


def main(_):
  model = Ranking(interaction="cross", training=False)
  data_pipe = CriteoTsvReader(use_synthetic_data=True)
  train_dataset = data_pipe(FLAGS.train_data, batch_size=1, is_training=True)

  for x, y in train_dataset.take(1):
    preds = model(x)

  tmp_path = tempfile.mkdtemp(dir='.')

  src = os.path.join(FLAGS.model_dir, "export_main")
  dest = os.path.join(FLAGS.model_dir, "export_main_optimized")

  recursive_copy(src, dest)

  export_to_savedmodel(model, savedmodel_dir=tmp_path)

  if tf.io.gfile.exists(os.path.join(dest, "saved_model.pb")):
    tf.io.gfile.remove(os.path.join(dest, "saved_model.pb"))
    tf.io.gfile.copy(
        os.path.join(tmp_path + "_main", "saved_model.pb"), os.path.join(dest, "saved_model.pb"), overwrite=True
    )


if __name__ == "__main__":
  app.run(main)
