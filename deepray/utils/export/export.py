# Copyright 2023 The Deepray Authors. All Rights Reserved.
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
import re
import sys
import tempfile
from typing import Optional, Union, Dict, Text, List

import horovod.tensorflow as hvd
import tensorflow as tf
from absl import logging, flags
from keras.engine import data_adapter
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

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


def export_to_checkpoint(saver: Union[tf.train.Checkpoint, tf.train.CheckpointManager], checkpoint_number=None):

  def helper(name, _saver):
    """Saves model to with provided checkpoint prefix."""
    latest_checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'ckpt_' + name))
    match = re.search(r"(?<=ckpt-)\d+", latest_checkpoint_file) if latest_checkpoint_file else None
    latest_step_ckpt = int(match.group()) if match else -1

    if latest_step_ckpt != checkpoint_number:
      save_path = _saver.save(checkpoint_number)
      logging.info('Saved checkpoint to {}'.format(save_path))

  if is_main_process():
    if isinstance(saver, dict):
      for name, _saver in saver.items():
        helper(name, _saver)
    else:
      helper(name="main", _saver=saver)


def export_to_savedmodel(
    model: Union[tf.keras.Model, Dict[Text, tf.keras.Model]],
    savedmodel_dir: Optional[Text] = None,
    checkpoint_dir: Optional[Union[Text, Dict[Text, Text]]] = None,
    restore_model_using_load_weights: bool = False
) -> Text:
  """Export keras model for serving which does not include the optimizer.

  Arguments:
      model: Keras model object to export.
      savedmodel_dir: Path to which exported model will be saved.
      checkpoint_dir: Path from which model weights will be loaded, if
        specified.
      restore_model_using_load_weights: Whether to use checkpoint.restore() API
        for custom checkpoint or to use model.load_weights() API.
        There are 2 different ways to save checkpoints. One is using
        tf.train.Checkpoint and another is using Keras model.save_weights().
        Custom training loop implementation uses tf.train.Checkpoint API
        and Keras ModelCheckpoint callback internally uses model.save_weights()
        API. Since these two API's cannot be used toghether, model loading logic
        must be taken into account how model checkpoint was saved.

  Raises:
    ValueError when model is not specified.
  """

  if FLAGS.use_dynamic_embedding and FLAGS.use_horovod:
    try:
      rank_array = hvd.allgather_object(get_rank(), name='check_tfra_ranks')
      assert len(set(rank_array)) == get_world_size()
    except:
      raise ValueError(f"Shouldn't place {sys._getframe().f_code.co_name} only in the main_process when use TFRA and Horovod.")

  def helper(name, _model: tf.keras.Model, _checkpoint_dir):
    _savedmodel_dir = os.path.join(FLAGS.model_dir, 'export') if savedmodel_dir is None else savedmodel_dir
    _savedmodel_dir = f"{_savedmodel_dir}_{name}"
    os.makedirs(_savedmodel_dir, exist_ok=True)

    if _checkpoint_dir:
      # Keras compile/fit() was used to save checkpoint using
      # model.save_weights().
      if restore_model_using_load_weights:
        model_weight_path = os.path.join(_checkpoint_dir, 'checkpoint')
        assert tf.io.gfile.exists(model_weight_path)
        _model.load_weights(model_weight_path)

      # tf.train.Checkpoint API was used via custom training loop logic.
      else:
        checkpoint = tf.train.Checkpoint(model=_model)

        # Restores the model from latest checkpoint.
        latest_checkpoint_file = tf.train.latest_checkpoint(_checkpoint_dir)
        assert latest_checkpoint_file
        logging.info('Checkpoint file %s found and restoring from '
                     'checkpoint', latest_checkpoint_file)
        checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()

    options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA']) if FLAGS.use_dynamic_embedding else None

    if is_main_process():
      tf.saved_model.save(_model, _savedmodel_dir, options=options)
    else:
      de_dir = os.path.join(_savedmodel_dir, "variables", "TFRADynamicEmbedding")
      for var in _model.variables:
        if hasattr(var, "params"):
          # save other rank's embedding weights
          var.params.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())
          # save opt weights
          # opt_de_vars = var.params.optimizer_vars.as_list(
          # ) if hasattr(var.params.optimizer_vars, "as_list") else var.params.optimizer_vars
          # for opt_de_var in opt_de_vars:
          #   opt_de_var.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())

    if is_main_process():
      logging.info(f"save pb model to: {_savedmodel_dir}, without optimizer & traces")

    return _savedmodel_dir

  if isinstance(model, dict):
    ans = []
    for name, _model in model.items():
      _dir = helper(name, _model, _checkpoint_dir=checkpoint_dir[name] if checkpoint_dir else None)
      ans.append(_dir)
    prefix_path = longestCommonPrefix(ans)
    logging.info(f"Export multiple models to {prefix_path}*")
    return prefix_path
  else:
    return helper(name="main", _model=model, _checkpoint_dir=checkpoint_dir)


def optimize_for_inference(
    model: Union[tf.keras.Model, Dict[Text, tf.keras.Model]],
    dataset: tf.data.Dataset,
    savedmodel_dir: Text,
) -> None:
  x, y, z = data_adapter.unpack_x_y_sample_weight(next(iter(dataset)))
  preds = model(x)
  logging.info(preds)

  def helper(_model, path):
    tmp_path = tempfile.mkdtemp(dir='/tmp/')
    export_to_savedmodel(_model, savedmodel_dir=tmp_path)
    file = os.path.join(path, "saved_model.pb")
    if tf.io.gfile.exists(file):
      tf.io.gfile.remove(file)
      logging.info(f"Replace optimized saved_modle.pb for {file}")
      tf.io.gfile.copy(os.path.join(tmp_path + "_main", "saved_model.pb"), file, overwrite=True)
    else:
      raise FileNotFoundError(f"{file} does not exist.")

  if isinstance(model, dict):
    for name, _model in model.items():
      src = savedmodel_dir + name
      helper(_model, src)
  else:
    helper(model, savedmodel_dir)


class SavedModel:

  def __init__(self, model_dir, precision):
    if FLAGS.use_dynamic_embedding:
      from tensorflow_recommenders_addons import dynamic_embedding as de
      de.enable_inference_mode()

    self.saved_model_loaded = tf.saved_model.load(model_dir, tags=[tag_constants.SERVING])
    self.graph_func = self.saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.precision = tf.float16 if precision == "amp" else tf.float32

    if not FLAGS.run_eagerly:
      self._infer_step = tf.function(self.infer_step)
    else:
      self._infer_step = self.infer_step

  def __call__(self, x, **kwargs):
    return self._infer_step(x)

  def infer_step(self, x):
    output = self.graph_func(**x)
    return output


class TFTRTModel:

  def export_model(self, model_dir, prec, tf_trt_model_dir=None):
    loaded_model = tf.saved_model.load(model_dir)
    signature = loaded_model.signatures['serving_default']
    logging.info(signature)
    # input_shape = [1, 384]
    # dummy_input = tf.constant(tf.zeros(input_shape, dtype=tf.int32))
    # x = [
    #   tf.constant(dummy_input, name='input_word_ids'),
    #   tf.constant(dummy_input, name='input_mask'),
    #   tf.constant(dummy_input, name='input_type_ids'),
    # ]
    # _ = model(x)

    trt_prec = trt.TrtPrecisionMode.FP32 if prec == "fp32" else trt.TrtPrecisionMode.FP16
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=trt.TrtConversionParams(precision_mode=trt_prec),
    )
    converter.convert()
    tf_trt_model_dir = tf_trt_model_dir or f'/tmp/tf-trt_model_{prec}'
    converter.save(tf_trt_model_dir)
    logging.info(f"TF-TRT model saved at {tf_trt_model_dir}")

  def __init__(self, model_dir, precision):
    temp_tftrt_dir = f"/tmp/tf-trt_model_{precision}"
    self.export_model(model_dir, precision, temp_tftrt_dir)
    saved_model_loaded = tf.saved_model.load(temp_tftrt_dir, tags=[tag_constants.SERVING])
    logging.info(f"TF-TRT model loaded from {temp_tftrt_dir}")
    self.graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.precision = tf.float16 if precision == "amp" else tf.float32

  def __call__(self, x, **kwargs):
    return self.infer_step(x)

  @tf.function
  def infer_step(self, x):
    output = self.graph_func(**x)
    return output


def longestCommonPrefix(strs: List[str]) -> str:
  if not strs:
    return ""

  length, count = len(strs[0]), len(strs)
  for i in range(length):
    c = strs[0][i]
    if any(i == len(strs[j]) or strs[j][i] != c for j in range(1, count)):
      return strs[0][:i]

  return strs[0]
