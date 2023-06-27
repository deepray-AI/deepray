# -*- coding:utf-8 -*-

import tensorflow as tf
from absl import logging, flags
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten, Lambda
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config
from tensorflow.python.ops import clip_ops

from .base_model import BaseModel

epsilon = backend_config.epsilon
FLAGS = flags.FLAGS


class TowerNewTFRA(BaseModel):
  def __call__(self, nn_hidden_units=(256, 128, 64), nn_l2_reg=0.0, nn_dropout=0.0, nn_use_bn=False, is_training=True, *args, **kwargs):
    self._nn_hidden_units = nn_hidden_units
    self._is_training = is_training

    self.targets = list(self.target_label_table.keys())
    self.input_dict = self.input_from_features()
    features = self.build_features()
    output_dict = self.build_network(features=features)
    model = tf.keras.Model(inputs=self.input_dict, outputs=output_dict)
    return model

  def build_network(self, flags=None, features=None):
    geek_nn_dense_features, job_nn_dense_features = self.get_input_and_dense_features(features, self._is_training, self.get_geek_nn_compo(), self.get_job_nn_compo(), self.targets, extra_dim=0)

    #         print("input_list:", len(input_list))
    #         print("geek nn:", len(geek_nn_dense_features))
    #         print("job nn:", len(job_nn_dense_features))

    x_job = Flatten()(Concatenate(axis=-1)(job_nn_dense_features))
    x_geek = Flatten()(Concatenate(axis=-1)(geek_nn_dense_features))
    for i, n in enumerate(self._nn_hidden_units):
      x_job = tf.keras.layers.Dense(n, activation='relu')(x_job)
      x_geek = tf.keras.layers.Dense(n, activation='relu')(x_geek)
    #             if nn_dropout:
    #                 x_job = tf.keras.layers.Dropout(nn_dropout[i])(x_job)
    #                 x_geek = tf.keras.layers.Dropout(nn_dropout[i])(x_geek)

    x_job = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x_job)
    x_geek = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x_geek)
    predict_out = tf.keras.layers.Dot(axes=-1, normalize=False)([x_job, x_geek])

    epsilon_ = constant_op.constant(epsilon(), dtype=predict_out.dtype.base_dtype)
    predict_out = clip_ops.clip_by_value(predict_out, epsilon_, 1. - epsilon_)

    # output target
    output_dict = dict()
    output_dict['addf'] = Lambda(lambda x: x, name="addf")(predict_out)
    output_dict['predict'] = Lambda(lambda x: x, name="predict")(predict_out)
    output_dict['predict_0'] = Lambda(lambda x: x, name="predict_0")(predict_out)
    output_dict['job_vec'] = tf.keras.layers.Lambda(lambda x: x, name='job_vec')(x_job)
    output_dict['geek_vec'] = tf.keras.layers.Lambda(lambda x: x, name='geek_vec')(x_geek)

    for i, target in enumerate(self.target_label_table):
      output_dict[target] = Lambda(lambda x: x, name=target)(predict_out)

    # output eva target & metrics
    print("conf.evaluate_target:", self.conf.evaluate_target)

    for key, config in self.conf.evaluate_target.items():
      target = config['target'] if 'target' in config else 'predict'
      if target in output_dict:
        pass
      else:
        target = 'predict'
      output_dict[key] = Lambda(lambda x: x, name=key)(output_dict[target])

    logging.info(f'output_dict: {output_dict}')
    return output_dict

  # 生成 NN Feature
  def get_input_and_dense_features(self, features, is_training, geek_comp, job_comp, targets=None, extra_dim=0):
    # NN features
    id_features = []
    nn_features = []
    all_features = []
    geek_cnt = geek_comp
    job_cnt = job_comp
    emb_dim_by_name = dict()
    num_targets = 1 if not targets else len(targets)
    geek_feature_set = set()
    for field, fea_list in self.field_dict.items():
      if field in geek_cnt:
        emb_dims = geek_cnt[field]
      elif field in job_cnt:
        emb_dims = job_cnt[field]
      else:
        continue

      if len(emb_dims) < len(fea_list):
        emb_dims = emb_dims + [emb_dims[-1]] * (len(fea_list) - len(emb_dims))

      for i, fea_name in enumerate(fea_list):
        if field in geek_cnt:
          geek_feature_set.add(fea_name)

        feature = features[fea_name]
        emb_name = feature.emb_name
        emb_dim = emb_dims[i]
        if self.conf.emb_reuse:
          if emb_name in emb_dim_by_name and emb_dim_by_name[emb_name] != emb_dim:
            logging.warn(f"[EMBED REUSE] {feature.name}@{emb_name} from {emb_dim} to {emb_dim_by_name[emb_name]}")
            emb_dim = emb_dim_by_name[emb_name]
          emb_dim_by_name[emb_name] = emb_dim
        if feature.emb_dynamic:
          id_features.append(self.make_feature(f=feature,
                                               emb_dim=emb_dim * num_targets + (extra_dim if extra_dim > 0 else 0),
                                               emb_split=[emb_dim] * num_targets + ([extra_dim] if extra_dim > 0 else [])))
        else:
          nn_features.append(self.make_feature(f=feature,
                                               emb_dim=emb_dim * num_targets + (extra_dim if extra_dim > 0 else 0),
                                               emb_split=[emb_dim] * num_targets + ([extra_dim] if extra_dim > 0 else [])))
        all_features.append(self.make_feature(f=feature,
                                              emb_dim=emb_dim * num_targets + (extra_dim if extra_dim > 0 else 0),
                                              emb_split=[emb_dim] * num_targets + ([extra_dim] if extra_dim > 0 else [])))

    emb_dict = self.embedding_from_feature(all_features, is_training)
    id_dense_features = self.dense_from_columns_id(id_features, emb_dict)
    print("id:", id_dense_features)
    nn_dense_features = self.dense_from_columns(nn_features, emb_dict)
    nn_dense_features.update(id_dense_features)

    geek_dense_features = []
    job_dense_features = []
    i = 0
    for emb_name, feas in nn_dense_features.items():
      if emb_name in geek_feature_set:
        geek_dense_features.append(feas)
      else:
        job_dense_features.append(feas)

    return geek_dense_features, job_dense_features
