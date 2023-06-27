import os
from collections import defaultdict
from itertools import chain

import tensorflow as tf
from absl import logging
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import clip_ops

from deepray.layers.core import DNN
from deepray.layers.mtl import CGC

epsilon = backend_config.epsilon

from .base_model import BaseModel


class CGCModel(BaseModel):

  def __call__(self, nn_hidden_units=(256, 128, 1), nn_l2_reg=0.0, nn_dropout=0.0, nn_use_bn=False, is_training=True, *args, **kwargs):
    target_label_table, eva_target_label_table, class_weight_table = self.get_target()
    targets = list(target_label_table.keys())
    num_targets = len(targets)

    gate_input_dim = self.conf.gate_input_dim if self.conf.gate_input_dim else 1
    input_list, nn_dense_features = self.get_input_and_dense_features(is_training, self.get_nn_compo(), targets + ['share'], extra_dim=gate_input_dim)
    #         print("nn_dense_features:", nn_dense_features)
    nn_target_inputs = list()
    for target in targets + ['share']:
      nn_target_inputs.append(tf.concat(nn_dense_features[target], axis=-1, name=f"{target}_input"))
    nn_input = tf.concat(nn_target_inputs, axis=1, name="cgc_input")
    nn_input_gate = tf.concat(nn_dense_features['extra'], axis=-1, name="gate_input")

    # output target
    output_dict = dict()
    output_weights = []
    output_tensors = []

    logging.info(f'class_weight_table: {class_weight_table}')

    num_experts_task = num_targets + 1 if self.conf.num_experts is None else self.conf.num_experts
    cgc_output = CGC(num_tasks=num_targets, num_experts_task=num_experts_task, num_experts_share=num_experts_task, units=[self.conf.units], output_share=False, name="cgc")([nn_input, nn_input_gate])
    target_outputs = tf.split(cgc_output, num_or_size_splits=num_targets, axis=1)
    logging.info(f'[CGC] gate_input_dim: {gate_input_dim}, num_targets: {num_targets}, num_experts_task: {num_experts_task}')

    bayes_inputs = dict()
    for i, target in enumerate(target_label_table):
      bayes_inputs[target] = DNN((self.conf.units, 64), name=f'bayes_{target}',
                                 l2_reg=nn_l2_reg,
                                 output_activation='relu',
                                 dropout_rate=nn_dropout,
                                 use_bn=nn_use_bn)(tf.squeeze(target_outputs[i], axis=1))

    for target, target_conf in self.conf.target.items():
      target_input = bayes_inputs[target]
      if "bayes" in target_conf and target_conf["bayes"] in bayes_inputs:
        bayes_input = bayes_inputs[target_conf["bayes"]]
        target_input = Concatenate(axis=-1, name=f'{target}_n_{target_conf["bayes"]}')([target_input, bayes_input])
      nn_target_output = DNN((1,), name=f'dnn_{target}', l2_reg=nn_l2_reg, dropout_rate=nn_dropout, use_bn=nn_use_bn)(target_input)
      # out = ClipActivation('sigmoid', name=f"clip_act_{target}")(nn_target_output)

      tout = tf.keras.layers.Activation(tf.nn.sigmoid)(nn_target_output)
      epsilon_ = constant_op.constant(epsilon(), dtype=tout.dtype.base_dtype)
      output = clip_ops.clip_by_value(tout, epsilon_, 1. - epsilon_)
      output_dict[target] = tf.keras.layers.Lambda(lambda x: x, name=target)(output)

      # output_dict[target] = Lambda(lambda x: x, name=target)(output)
      output_weights.append(class_weight_table[target])
      output_tensors.append(output_dict[target])

    # output predict
    if len(output_tensors) > 1:
      weight_sum = sum(output_weights)
      output_weights = [w / weight_sum for w in output_weights]
      if 'multiply' in self.conf.target_fusion_type:
        predict_out = tf.reduce_prod(tf.pow(tf.concat(output_tensors, axis=-1), output_weights), name="predict_multiply", axis=-1, keepdims=True)
      else:  # add
        predict_out = tf.reduce_sum(tf.multiply(tf.concat(output_tensors, axis=-1), output_weights), name="predict_add", axis=-1, keepdims=True)
    else:
      predict_out = output_tensors[0]
    output_dict['predict'] = Lambda(lambda x: x, name="predict")(predict_out)

    # for mtl fusion plugin
    i = 0
    for target in ['det', 'addf', 'chat', 'success', 'refuse']:
      if target in output_dict:
        predict_name = 'predict_%d' % i
        if target == 'refuse':
          output_dict[predict_name] = tf.keras.layers.Lambda(lambda x: 1.0 - x, name=predict_name)(output_dict[target])
        else:
          output_dict[predict_name] = tf.keras.layers.Lambda(lambda x: x, name=predict_name)(output_dict[target])
        i += 1

    # output eva target & metrics
    metrics = dict()
    weighted_metrics = dict()
    for key, config in self.conf.evaluate_target.items():
      target = config['target'] if 'target' in config else 'predict'
      if target in output_dict:
        metric = tf.keras.metrics.AUC(num_thresholds=1000, summation_method='minoring', name='auc')
        # pr_metric = tf.keras.metrics.AUC(num_thresholds=1000, summation_method='minoring', name='pr_auc', curve='PR')
        if 'weighted' in config and config['weighted']:
          weighted_metrics[key] = metric
        else:
          metrics[key] = metric
      else:
        target = 'predict'
      output_dict[key] = Lambda(lambda x: x, name=key)(output_dict[target])

    logging.info(f'class_weight_table: {class_weight_table}, output_dict: {output_dict}')
    return tf.keras.Model(inputs=input_list, outputs=output_dict)

  # 生成 Input 及 FFM & NN Feature
  def get_input_and_dense_features(self, is_training, nn_comp, targets=None, extra_dim=0):
    # Inputs
    used_features = list(chain.from_iterable(self.field_dict.values()))
    conti_features = self.conti_fea_dict()
    features = dict()
    num_targets = 1 if not targets else len(targets)
    for fname in used_features:
      code, dtype, length, value = self.fea_code[fname], self.fea_dtype[fname], self.fea_length[fname], self.fea_def_valu_dict[fname]

      use_hash = False
      emb_size = 0
      vocab = None
      boundaries = None
      emb_name = fname
      if fname in self.conf.hash_fea_bucket_size_dict:
        use_hash = True
        emb_size = self.conf.hash_fea_bucket_size_dict[fname]
      elif fname in self.conf.id_fea_bucket_size_dict:
        emb_size = self.conf.id_fea_bucket_size_dict[fname]
      elif fname in self.cate_fea_dict:
        if self.conf.emb_reuse:
          emb_name = self.fea_tag_dict[fname]
        vocab = self.cate_fea_dict[fname]
        emb_size = len(vocab) + 1
      elif fname in conti_features:
        boundaries = conti_features[fname]
        emb_size = len(boundaries) + 1

      features[fname] = self.make_feature(name=fname, code=code, dtype=dtype, len=length, default_value=value,
                                          use_hash=use_hash, vocab=vocab, boundaries=boundaries,
                                          emb_name=emb_name, emb_size=emb_size)
    input_dict = self.input_from_features(features.values())

    # NN features
    nn_features = []
    nn_cnt = nn_comp
    emb_dim_byname = dict()
    for field, fea_list in self.field_dict.items():
      if field not in nn_cnt:
        continue
      emb_dims = nn_cnt[field]
      if len(emb_dims) < len(fea_list):
        emb_dims = emb_dims + [emb_dims[-1]] * (len(fea_list) - len(emb_dims))

      for i, fea_name in enumerate(fea_list):
        feature = features[fea_name]
        emb_name = feature.emb_name
        emb_dim = emb_dims[i]
        if self.conf.emb_reuse:
          if emb_name in emb_dim_byname and emb_dim_byname[emb_name] != emb_dim:
            logging.warn(f"[EMBED REUSE] {feature.name}@{emb_name} from {emb_dim} to {emb_dim_byname[emb_name]}")
            emb_dim = emb_dim_byname[emb_name]
          emb_dim_byname[emb_name] = emb_dim
        nn_features.append(self.make_feature(f=feature,
                                             emb_dim=emb_dim * num_targets + (extra_dim if extra_dim > 0 else 0),
                                             emb_split=[emb_dim] * num_targets + ([extra_dim] if extra_dim > 0 else [])))
    emb_dict = self.embedding_from_feature(nn_features, is_training)
    nn_dense_features = self.dense_from_columns(nn_features, input_dict, emb_dict)

    if num_targets > 1:
      feas_split_dense = defaultdict(list)
      for feature in nn_features:
        fea_name = feature.name
        fea_dense = nn_dense_features[fea_name]
        fea_denses = tf.split(fea_dense, num_or_size_splits=feature.emb_split, axis=-1, name=f"split_{fea_name}")
        for i, target in enumerate(targets):
          feas_split_dense[target].append(fea_denses[i])
        if extra_dim > 0:
          feas_split_dense['extra'].append(fea_denses[-1])

      return list(input_dict.values()), feas_split_dense

    return list(input_dict.values()), list(nn_dense_features.values())

  # 读取nn的特征交叉信息
  def get_nn_compo(self):
    nn_cnt = dict()
    nn_path = os.path.join(self.conf.conf_path, self.conf.network_compo_dir, 'nn')
    with open(nn_path, 'r') as fr:
      for line in fr.readlines():
        line = line.strip().split('\t')
        fields = line[0].split(',')
        fea_len = list(map(lambda x: int(x), line[1].split(',')))
        for field in fields:
          if field not in nn_cnt:
            nn_cnt[field] = fea_len
    return nn_cnt
