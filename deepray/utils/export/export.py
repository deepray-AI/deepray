# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Convenience functions for exporting models as SavedModels or other types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import logging, flags


from deepray.utils.horovod_utils import is_main_process, get_world_size, get_rank

FLAGS = flags.FLAGS


def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32, batch_size=1):
  """Returns a input_receiver_fn that can be used during serving.

  This expects examples to come through as float tensors, and simply
  wraps them as TensorServingInputReceivers.

  Arguably, this should live in tf.estimator.export. Testing here first.

  Args:
    shape: list representing target size of a single example.
    dtype: the expected datatype for the input example
    batch_size: number of input tensors that will be passed for prediction

  Returns:
    A function that itself returns a TensorServingInputReceiver.
  """

  def serving_input_receiver_fn():
    # Prep a placeholder where the input example will be fed in
    features = tf.compat.v1.placeholder(dtype=dtype, shape=[batch_size] + shape, name='input_tensor')

    return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors=features)

  return serving_input_receiver_fn


def export_to_savedmodel(model):
  savedmodel_dir = os.path.join(FLAGS.model_dir, 'export')
  os.makedirs(savedmodel_dir, exist_ok=True)
  logging.info(f"save pb model to:{savedmodel_dir}, without optimizer & traces")

  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  if not os.path.exists(savedmodel_dir):
    os.mkdir(savedmodel_dir)

  if is_main_process():
    tf.saved_model.save(model, savedmodel_dir, options=options)
  else:
    de_dir = os.path.join(savedmodel_dir, "variables", "TFRADynamicEmbedding")
    for var in model.variables:
      if hasattr(var, "params"):
        # save other rank's embedding weights
        var.params.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())
        # save opt weights
        # opt_de_vars = var.params.optimizer_vars.as_list(
        # ) if hasattr(var.params.optimizer_vars, "as_list") else var.params.optimizer_vars
        # for opt_de_var in opt_de_vars:
        #   opt_de_var.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())


def export_for_serving(model, export_model):
  export_dir = os.path.join(FLAGS.model_dir, 'serving')
  os.makedirs(export_dir, exist_ok=True)
  logging.info(f"save pb model to:{export_dir}, without optimizer & traces")

  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  def save_spec():
    # tf 2.6 以上版本
    if hasattr(model, 'save_spec'):
      return model.save_spec()
    else:
      arg_specs = list()
      kwarg_specs = dict()
      for i in model.inputs:
        arg_specs.append(i.type_spec)
      return [arg_specs], kwarg_specs

  @tf.function
  def serve(*args, **kwargs):
    return model(*args, **kwargs)

  arg_specs, kwarg_specs = save_spec()

  if is_main_process():
    tf.keras.models.save_model(
        model,
        export_dir,
        overwrite=True,
        include_optimizer=False,
        options=options,
        signatures={'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)},
    )
  else:
    de_dir = os.path.join(export_dir, "variables", "TFRADynamicEmbedding")
    for var in model.variables:
      if hasattr(var, "params"):
        var.params.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())

  if is_main_process():
    # 修改计算图变成单机版
    from tensorflow.python.saved_model import save as tf_save
    # save_and_return_nodes函数用来覆盖save_model函数生成的saved_model.pb文件，重写计算图
    tf_save.save_and_return_nodes(
        obj=export_model,
        export_dir=export_dir,
        options=options,
        signatures={'serving_default': serve.get_concrete_function(*arg_specs, **kwarg_specs)},
        experimental_skip_checkpoint=True
    )
