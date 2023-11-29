# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""unit tests of dynamic embedding optimizer ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import shutil

from absl import app, flags
import tensorflow as tf

try:
  from tensorflow_recommenders_addons import dynamic_embedding as de
except:
  pass

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import monitored_session
from tensorflow.python.training import training_util
try:
  from tensorflow.keras.optimizers.legacy import Adam
except:
  from tensorflow.keras.optimizers import Adam

import deepray

FLAGS = flags.FLAGS
FLAGS.use_horovod = True
FLAGS.keras_use_ctl = True
FLAGS.use_dynamic_embedding = True
FLAGS.epochs = 1
FLAGS.batch_size = 8
FLAGS.steps_per_summary = 2
FLAGS.save_checkpoint_steps = 2
FLAGS.stop_steps = 10

from deepray.core.base_trainer import Trainer
from deepray.layers.dynamic_embedding import DistributedDynamicEmbedding, DynamicEmbeddingOption
from deepray.utils.export import export_to_savedmodel

default_config = config_pb2.ConfigProto(allow_soft_placement=True, gpu_options=config_pb2.GPUOptions(allow_growth=True))


def get_emb_sequential_model(emb_t, *args, **kwargs):
  l0 = tf.keras.layers.InputLayer(input_shape=(None,), dtype=dtypes.int64)
  l1 = emb_t(*args, **kwargs)
  l2 = tf.keras.layers.Dense(8, 'relu', kernel_initializer='zeros')
  l3 = tf.keras.layers.Dense(1, 'sigmoid', kernel_initializer='zeros')
  if emb_t == tf.keras.layers.Embedding:
    model = tf.keras.Sequential([l0, l1, l2, l3])
  elif emb_t == DistributedDynamicEmbedding:
    model = tf.keras.Sequential([l0, l1, l2, l3])
  else:
    raise TypeError('Unsupported embedding layer {}'.format(emb_t))
  return model


class HorovodTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def test_adam_minimize_trainable(self):
    tf.keras.backend.clear_session()
    keras_base_opt = Adam(1.0)
    keras_test_opt = Adam(1.0)
    self.common_minimize_trainable_v2(keras_base_opt, keras_test_opt, name="keras_adam")

  @test_util.run_all_in_graph_and_eager_modes
  def test_all_to_all_embedding_trainable(self):
    # TODO: Resolve the conflict arising from the 'save' function incompatibility with TensorFlow 2.11.
    if (tf.__version__ == "2.11.0" or tf.__version__ == "2.11.1"):
      self.skipTest("The save function doesn't work with TF 2.11, skip the test.")
    keras_base_opt = Adam(1.0)
    keras_test_opt = Adam(1.0)
    self.common_all_to_all_embedding_trainable_v2(keras_base_opt, keras_test_opt, name="keras_adam")

  def common_minimize_trainable_v2(self, base_opt, test_opt, name):
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest("Skip the test for horovod because it's not available.")
    try:
      from tensorflow_recommenders_addons import dynamic_embedding as de
    except (NotFoundError):
      self.skipTest("Skip the test for TFRA DE because it's not available.")

    tf.config.set_soft_device_placement(True)

    hvd.init()

    # These cases need 2 GPUs at least if available.
    logical_devices = tf.config.list_logical_devices('GPU')
    _device = "GPU" if len(logical_devices) >= hvd.size() else "CPU"
    _device_id = hvd.local_rank() if _device == "GPU" and len(logical_devices) >= 2 else 0

    if _device == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_id)

    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    for dtype, run_step, dim in itertools.product([dtypes.float32], [1], [10]):
      print("device=", "/{}:{}".format(_device, _device_id))
      with tf.device("/{}:{}".format(_device, _device_id)):
        x = tf.random.uniform(shape=[32, dim])
        y = tf.zeros([32, 1])

        with tf.GradientTape() as tape_base:
          base_weight = tf.compat.v1.get_variable(name="base_weights", initializer=tf.ones([10, 1]))

          base_logits = tf.nn.relu(math_ops.matmul(x, base_weight))
          base_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=base_logits)

          # grad_base = tape_base.gradient(base_loss, base_weight)
          # base_opt.
        base_opt_op = base_opt.minimize(base_loss, var_list=[base_weight], tape=tape_base)

        with tf.GradientTape() as tape_test:
          test_weight = tf.compat.v1.get_variable(name="test_weights", initializer=tf.ones([10, 1]))

          test_logits = tf.nn.relu(math_ops.matmul(x, test_weight))
          test_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=test_logits)

        grads_and_vars = test_opt._compute_gradients(test_loss, var_list=[test_weight], tape=tape_test)
        var_list = []
        aggregated_grad = []
        for grad, var in grads_and_vars:
          var_list.append(var)
          aggregated_grad.append(hvd.allreduce(grad, op=hvd.Sum))
        aggregated_grads_and_vars = zip(aggregated_grad, var_list)
        test_opt_op = test_opt.apply_gradients(aggregated_grads_and_vars)

      with monitored_session.MonitoredTrainingSession(is_chief=True, config=default_config) as sess:
        for _ in range(run_step):
          sess.run(base_opt_op)
          sess.run(test_opt_op)

        self.assertAllCloseAccordingToType(
            sess.run(base_weight),
            sess.run(test_weight),
            msg="Cond:{},{},{}".format(dtype, run_step, dim),
        )

  def common_all_to_all_embedding_trainable_v2(self, base_opt, test_opt, name):
    try:
      import horovod.tensorflow as hvd
    except (NotFoundError):
      self.skipTest("Skip the test for horovod because it's not available.")
    try:
      from tensorflow_recommenders_addons import dynamic_embedding as de
    except (NotFoundError):
      self.skipTest("Skip the test for TFRA DE because it's not available.")

    tf.config.set_soft_device_placement(True)

    hvd.init()

    # These cases need 2 GPUs at least if available.
    logical_devices = tf.config.list_logical_devices('GPU')
    _device = "GPU" if len(logical_devices) >= hvd.size() else "CPU"
    _device_id = hvd.local_rank() if _device == "GPU" and len(logical_devices) >= 2 else 0

    if _device == "GPU":
      os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_id)
      _de_device = 'HBM'
    else:
      _de_device = 'DRAM'

    base_opt = de.DynamicEmbeddingOptimizer(base_opt, synchronous=True)
    test_opt = test_opt
    init = tf.keras.initializers.Zeros()
    batch_size = FLAGS.batch_size
    for dtype, run_step, dim in itertools.product([dtypes.float32], [FLAGS.stop_steps], [10]):
      print("device=", "/{}:{}".format(_device, _device_id))
      with tf.device("/{}:{}".format(_device, _device_id)):
        total_data_num = batch_size * run_step
        x = math_ops.range(0, total_data_num, dtype=dtypes.int64)
        x = tf.reshape(x, (-1, 1))
        y = tf.zeros((total_data_num, 1), dtype=dtypes.float32)
        train_input_fn = tf.data.Dataset.from_tensor_slices((x, y)).cache().repeat()
        train_input_fn = train_input_fn.batch(batch_size)

        base_model = get_emb_sequential_model(
            DistributedDynamicEmbedding,
            embedding_dim=dim,
            key_dtype=tf.int64,
            value_dtype=dtype,
            initializer=init,
            de_option=DynamicEmbeddingOption(device=_de_device),
            name='all2all_emb'
        )
        test_model = get_emb_sequential_model(
            tf.keras.layers.Embedding,
            input_dim=batch_size * run_step,
            output_dim=dim,
            embeddings_initializer=init,
            name='tf_emb'
        )
        loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        base_trainer = Trainer(model=base_model, optimizer=base_opt, loss=loss_func)
        test_trainer = Trainer(model=test_model, optimizer=test_opt, loss=loss_func)

        hvd_model_dir = "/tmp/hvd_save_restore" + str(hvd.size()) + str(run_step) + str(
            dim
        )  # All ranks should share same save directory
        tf_model_dir = "/tmp/tf_save_restore" + str(hvd.size()) + str(run_step) + str(
            dim
        )  # All ranks should share same save directory

        FLAGS.model_dir = tf_model_dir
        test_trainer.fit(train_input=train_input_fn,)

        FLAGS.model_dir = hvd_model_dir
        base_trainer.fit(train_input=train_input_fn,)

      self.assertAllCloseAccordingToType(
          base_model.layers[1].weights[0],
          test_model.layers[1].weights[0],
          msg="Cond:{},{},{}".format(dtype, run_step, dim),
      )

      self.assertAllCloseAccordingToType(
          base_model.layers[2].weights[0],
          test_model.layers[2].weights[0],
          msg="Cond:{},{},{}".format(dtype, run_step, dim),
      )

      a2aemb_size = base_model.layers[0].emb.params.size()

      export_to_savedmodel(base_model)

      del base_model
      del base_trainer
      del base_opt
      tf.keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()
      new_base_model = get_emb_sequential_model(
          DistributedDynamicEmbedding,
          embedding_dim=dim,
          key_dtype=tf.int64,
          value_dtype=dtype,
          initializer=init,
          de_option=DynamicEmbeddingOption(device=_de_device,),
          name='all2all_emb'
      )
      new_base_opt = de.DynamicEmbeddingOptimizer(Adam(1.0), synchronous=True)
      FLAGS.init_checkpoint = [os.path.join(FLAGS.model_dir, 'ckpt_main_model')]
      FLAGS.stop_steps = 0
      new_base_trainer = Trainer(model=new_base_model, optimizer=new_base_opt, loss=loss_func)
      new_base_trainer.fit(train_input=train_input_fn,)

      new_a2aemb_size = new_base_model.layers[0].emb.params.size()
      self.assertEqual(a2aemb_size, new_a2aemb_size)

      hvd.join()  # Sync for avoiding files conflict
      tf.keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()
      new_base_model.load_weights(FLAGS.model_dir + '/export_main/variables/variables')
      new_a2aemb_size = new_base_model.layers[0].emb.params.size()
      self.assertEqual(a2aemb_size, new_a2aemb_size)
      hvd.join()  # Sync for avoiding files conflict


if __name__ == "__main__":
  app.run(test.main())
