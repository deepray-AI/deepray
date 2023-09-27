# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to save models."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os
import typing

import tensorflow as tf
from absl import logging, flags
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

FLAGS = flags.FLAGS


def export_bert_model(
    model_export_path: typing.Text,
    model: tf.keras.Model,
    checkpoint_dir: typing.Optional[typing.Text] = None,
    restore_model_using_load_weights: bool = False
) -> None:
  """Export keras model for serving which does not include the optimizer.

  Arguments:
      model_export_path: Path to which exported model will be saved.
      model: Keras model object to export.
      checkpoint_dir: Path from which model weights will be loaded, if
        specified.
      restore_model_using_load_weights: Whether to use checkpoint.restore() API
        for custom checkpoint or to use model.load_weights() API.
        There are 2 different ways to save checkpoints. One is using
        tf.train.Checkpoint and another is using Keras model.save_weights().
        Custom training loop implementation uses tf.train.Checkpoint API
        and Keras ModelCheckpoint callback internally uses model.save_weights()
        API. Since these two API's cannot be used toghether, model loading logic
        must be take into account how model checkpoint was saved.

  Raises:
    ValueError when either model_export_path or model is not specified.
  """
  if not model_export_path:
    raise ValueError('model_export_path must be specified.')
  if not isinstance(model, tf.keras.Model):
    raise ValueError('model must be a tf.keras.Model object.')

  if checkpoint_dir:
    # Keras compile/fit() was used to save checkpoint using
    # model.save_weights().
    if restore_model_using_load_weights:
      model_weight_path = os.path.join(checkpoint_dir, 'checkpoint')
      assert tf.io.gfile.exists(model_weight_path)
      model.load_weights(model_weight_path)

    # tf.train.Checkpoint API was used via custom training loop logic.
    else:
      checkpoint = tf.train.Checkpoint(model=model)

      # Restores the model from latest checkpoint.
      latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
      assert latest_checkpoint_file
      logging.info('Checkpoint file %s found and restoring from '
                   'checkpoint', latest_checkpoint_file)
      checkpoint.restore(latest_checkpoint_file).assert_existing_objects_matched()

  model.save(model_export_path, include_optimizer=False, save_format='tf')


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
