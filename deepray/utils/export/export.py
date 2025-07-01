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

import inspect
import os
import re
import tempfile
from typing import Optional, Union, Dict, Text, List

import tensorflow as tf
from absl import flags
from packaging.version import parse

if parse(tf.__version__.replace("-tf", "+tf")) < parse("2.11"):
  from keras.engine import data_adapter
elif parse(tf.__version__) > parse("2.16.0"):
  from tf_keras.src.engine import data_adapter
else:
  from keras.src.engine import data_adapter
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from deepray.utils import logging_util
from deepray.utils.horovod_utils import is_main_process, get_world_size, get_rank

logger = logging_util.get_logger()


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
    features = tf.compat.v1.placeholder(dtype=dtype, shape=[batch_size] + shape, name="input_tensor")

    return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors=features)

  return serving_input_receiver_fn


def export_to_checkpoint(saver: Union[tf.train.Checkpoint, tf.train.CheckpointManager], checkpoint_number=None):
  def helper(name, _saver):
    """Saves model to with provided checkpoint prefix."""
    latest_checkpoint_file = tf.train.latest_checkpoint(os.path.join(flags.FLAGS.model_dir, "ckpt_" + name))
    match = re.search(r"(?<=ckpt-)\d+", latest_checkpoint_file) if latest_checkpoint_file else None
    latest_step_ckpt = int(match.group()) if match else -1

    if latest_step_ckpt != checkpoint_number:
      save_path = _saver.save(checkpoint_number)
      logger.info("Saved checkpoint to {}".format(save_path))

  def _save_fn():
    if isinstance(saver, dict):
      for name, _saver in saver.items():
        helper(name, _saver)
    else:
      helper(name="main", _saver=saver)

  if flags.FLAGS.use_horovod and flags.FLAGS.use_dynamic_embedding:
    _save_fn()
  else:
    _save_fn()


def export_to_savedmodel(
  model: Union[tf.keras.Model, Dict[Text, tf.keras.Model]],
  savedmodel_dir: Optional[Text] = None,
  checkpoint_dir: Optional[Union[Text, Dict[Text, Text]]] = None,
  restore_model_using_load_weights: bool = False,
  include_optimizer: bool = False,
  signatures=None,
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

  if flags.FLAGS.use_dynamic_embedding and flags.FLAGS.use_horovod:
    try:
      import horovod.tensorflow as hvd

      rank_array = hvd.allgather_object(get_rank(), name="check_tfra_ranks")
      assert len(set(rank_array)) == get_world_size()
    except:
      raise ValueError(f"Shouldn't place {inspect.stack()[0][3]} only in the main_process when use TFRA and Horovod.")

  def helper(name, _model: tf.keras.Model, _checkpoint_dir):
    _savedmodel_dir = os.path.join(flags.FLAGS.model_dir, "export") if savedmodel_dir is None else savedmodel_dir
    if get_world_size() > 1:
      _savedmodel_dir = f"{_savedmodel_dir}_{name}_{get_rank()}"
    else:
      _savedmodel_dir = f"{_savedmodel_dir}_{name}"
    os.makedirs(_savedmodel_dir, exist_ok=True)

    if _checkpoint_dir:
      # Keras compile/fit() was used to save checkpoint using
      # model.save_weights().
      if restore_model_using_load_weights:
        model_weight_path = os.path.join(_checkpoint_dir, "checkpoint")
        assert tf.io.gfile.exists(model_weight_path)
        _model.load_weights(model_weight_path)

      # tf.train.Checkpoint API was used via custom training loop logic.
      else:
        checkpoint = tf.train.Checkpoint(model=_model)

        # Restores the model from latest checkpoint.
        latest_checkpoint_file = tf.train.latest_checkpoint(_checkpoint_dir)
        assert latest_checkpoint_file
        logger.info("Checkpoint file %s found and restoring from checkpoint", latest_checkpoint_file)
        checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()

    if flags.FLAGS.use_dynamic_embedding:
      try:
        from tensorflow_recommenders_addons import dynamic_embedding as de

        de.keras.models.de_save_model(
          _model, _savedmodel_dir, overwrite=True, include_optimizer=include_optimizer, signatures=signatures
        )
      except:
        # Compatible with TFRA version before commit 460b50847d459ebbf91b30ea0f9499fbc7ed9da0
        def _check_de_var_with_fs_saver(_var):
          try:
            from tensorflow_recommenders_addons import dynamic_embedding as de

            # This function only serves FileSystemSaver.
            return (
              hasattr(_var, "params")
              and hasattr(_var.params, "_created_in_class")
              and _var.params._saveable_object_creator is not None
              and isinstance(_var.params.kv_creator.saver, de.FileSystemSaver)
            )
          except:
            return False

        de_dir = os.path.join(_savedmodel_dir, "variables", "TFRADynamicEmbedding")
        options = tf.saved_model.SaveOptions(namespace_whitelist=["TFRA"])
        if is_main_process():
          for var in _model.variables:
            _is_dump = _check_de_var_with_fs_saver(var)
            if _is_dump:
              de_var = var.params
              if hasattr(de_var, "saveable"):
                de_var.saveable._saver_config.save_path = de_dir
          tf.saved_model.save(_model, export_dir=_savedmodel_dir, signatures=signatures, options=options)
        else:
          for var in _model.variables:
            _is_dump = _check_de_var_with_fs_saver(var)
            if _is_dump:
              de_var = var.params
              a2a_emb = de_var._created_in_class
              # save other rank's embedding weights
              var.params.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())
              # save opt weights
              if include_optimizer:
                de_opt_vars = (
                  a2a_emb.optimizer_vars.as_list()
                  if hasattr(a2a_emb.optimizer_vars, "as_list")
                  else a2a_emb.optimizer_vars
                )
                for de_opt_var in de_opt_vars:
                  de_opt_var.save_to_file_system(dirpath=de_dir, proc_size=get_world_size(), proc_rank=get_rank())
    else:
      if is_main_process():
        tf.saved_model.save(_model, export_dir=_savedmodel_dir, signatures=signatures)

    if is_main_process():
      logger.info(f"save pb model to: {_savedmodel_dir}, without optimizer & traces")

    return _savedmodel_dir

  if isinstance(model, dict):
    ans = []
    for name, _model in model.items():
      _dir = helper(name, _model, _checkpoint_dir=checkpoint_dir[name] if checkpoint_dir else None)
      ans.append(_dir)
    prefix_path = longestCommonPrefix(ans)
    logger.info(f"Export multiple models to {prefix_path}*")
    return prefix_path
  else:
    return helper(name="main", _model=model, _checkpoint_dir=checkpoint_dir)


def optimize_for_inference(
  model: Union[tf.keras.Model, Dict[Text, tf.keras.Model]],
  savedmodel_dir: Text,
  dataset: tf.data.Dataset = None,
  signatures=None,
) -> None:
  x = None
  if dataset:
    x, y, z = data_adapter.unpack_x_y_sample_weight(next(iter(dataset)))
    if isinstance(model, dict):
      for name, _model in model.items():
        if "main" in name:
          preds = _model(x)
          logger.debug(preds)
    else:
      preds = model(x)
      logger.debug(preds)

  def helper(_model, path):
    tmp_path = tempfile.mkdtemp(dir="/tmp/")
    export_to_savedmodel(_model, savedmodel_dir=tmp_path, signatures=signatures)
    file = os.path.join(path, "saved_model.pb")
    if tf.io.gfile.exists(file):
      tf.io.gfile.remove(file)
      logger.info(f"Replace optimized saved_modle.pb for {file}")
      tf.io.gfile.copy(os.path.join(tmp_path + "_main", "saved_model.pb"), file, overwrite=True)
    else:
      raise FileNotFoundError(f"{file} does not exist.")

  if isinstance(model, dict):
    for name, _model in model.items():
      if dataset:
        if "main" in name:
          preds = _model(x)
          logger.info(preds)
      src = savedmodel_dir + name
      helper(_model, src)
  else:
    if dataset:
      preds = model(x)
      logger.info(preds)
    helper(model, savedmodel_dir)


class SavedModel:
  def __init__(self, model_dir, precision):
    if flags.FLAGS.use_dynamic_embedding:
      from tensorflow_recommenders_addons import dynamic_embedding as de

      de.enable_inference_mode()

    self.saved_model_loaded = tf.saved_model.load(model_dir, tags=[tag_constants.SERVING])
    self.graph_func = self.saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.precision = tf.float16 if precision == "amp" else tf.float32

    if not flags.FLAGS.run_eagerly:
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
    signature = loaded_model.signatures["serving_default"]
    logger.info(signature)
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
    tf_trt_model_dir = tf_trt_model_dir or f"/tmp/tf-trt_model_{prec}"
    converter.save(tf_trt_model_dir)
    logger.info(f"TF-TRT model saved at {tf_trt_model_dir}")

  def __init__(self, model_dir, precision):
    temp_tftrt_dir = f"/tmp/tf-trt_model_{precision}"
    self.export_model(model_dir, precision, temp_tftrt_dir)
    saved_model_loaded = tf.saved_model.load(temp_tftrt_dir, tags=[tag_constants.SERVING])
    logger.info(f"TF-TRT model loaded from {temp_tftrt_dir}")
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
