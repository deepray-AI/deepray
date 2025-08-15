import os

# reduce number of threads
os.environ["TF_NUM_INTEROP_THREADS"] = "1"  # 设置操作之间的线程数
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # 设置单个操作内部的线程数
from absl import flags

import deepray as dp
import tf_keras as keras
from deepray.layers.embedding_variable import EmbeddingVariable
import tensorflow as tf
from deepray.layers.mlp import MLP
from typing import Dict
from deepray.core.trainer import Trainer
from deepray.callbacks import ModelCheckpoint
from deepray.utils.export import export_to_savedmodel
import numpy as np
from deepray.callbacks.training_speed import TrainingSpeed

np.set_printoptions(suppress=True)


class TestModel(keras.Model):
  def __init__(self, training=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # breakpoint()
    self.embedding_layer = EmbeddingVariable(
      embedding_dim=8,
      key_dtype="int64",
      value_dtype=tf.float32,
      initializer=keras.initializers.TruncatedNormal(seed=flags.FLAGS.random_seed),
      # initializer=keras.initializers.Constant(value=0.1),
      name="ev_emb",
      with_unique=False,
      storage_type="DRAM",
      # storage_type="HBM_DRAM",
    )
    self._top_stack = MLP(hidden_units=[6, 1], activations=[None, "sigmoid"], name="testDense")

  def call(self, inputs: Dict[str, tf.Tensor], training=None, mask=None) -> tf.Tensor:
    feature_interaction_output = self.embedding_layer(inputs)
    # print(feature_interaction_output.numpy()[0])
    prediction = self._top_stack(feature_interaction_output)
    # prediction = tf.keras.activations.sigmoid(feature_interaction_output)
    return prediction


def main():
  pid = os.getpid()
  # 验证设置
  print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
  print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
  input("pid: " + str(pid) + ", press enter to continue")
  # optimizer = dp.optimizers.SGD(flags.FLAGS.learning_rate)
  # optimizer = dp.optimizers.Adagrad(learning_rate=flags.FLAGS.learning_rate)

  optimizer = dp.optimizers.Adam(learning_rate=flags.FLAGS.learning_rate)
  # optimizer = dp.optimizers.AdamAsync(learning_rate=flags.FLAGS.learning_rate)
  dataset = tf.data.Dataset.from_tensor_slices((
    [8, 8, -0, 8, -2, 8451010344448425984, 8451006525551411200, 3, 5, 4, 6, 7],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
  )).batch(flags.FLAGS.batch_size)
  model = TestModel()

  trainer = Trainer(model=model, optimizer=optimizer, loss="binary_crossentropy", metrics=["AUC"], jit_compile=False)

  trainer.fit(
    x=dataset,
    callbacks=[
      TrainingSpeed(),
      ModelCheckpoint(),
    ],
  )

  savedmodel_path = export_to_savedmodel(trainer.main_model)
  print(savedmodel_path)


if __name__ == "__main__":
  dp.runner(main)
