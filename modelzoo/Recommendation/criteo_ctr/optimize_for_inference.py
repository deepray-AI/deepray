# -*- coding: utf-8 -*-
import logging
import os
import tempfile

import tensorflow as tf
from absl import app, flags

from dcn_v2 import Ranking
from deepray.datasets.criteo import CriteoTsvReader
from deepray.utils.export import export_to_savedmodel

FLAGS = flags.FLAGS


def main(_):
  model = Ranking(interaction="cross", training=False)
  data_pipe = CriteoTsvReader(use_synthetic_data=True)

  # Why do we perfer to use only one example to rebuild model?
  #
  train_dataset = data_pipe(FLAGS.train_data, batch_size=1, is_training=True)
  for x, y in train_dataset.take(1):
    preds = model(x)

  tmp_path = tempfile.mkdtemp(dir='/tmp/')

  src = os.path.join(FLAGS.model_dir, "export_main")

  export_to_savedmodel(model, savedmodel_dir=tmp_path)

  file = os.path.join(src, "saved_model.pb")
  if tf.io.gfile.exists(file):
    tf.io.gfile.remove(file)
    logging.info(f"Replace optimized saved_modle.pb for {file}")
    tf.io.gfile.copy(os.path.join(tmp_path + "_main", "saved_model.pb"), file, overwrite=True)


if __name__ == "__main__":
  app.run(main)
