import abc
import datetime
import math
import os
import sys
from collections import OrderedDict, defaultdict
from collections import namedtuple

import tensorflow as tf
from absl import logging, flags
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import backend as K

from deepray.layers.bucketize import Bucketize
from deepray.layers.embedding import DynamicEmbedding
from deepray.layers.seq import Pooling
from deepray.utils.data.feature_map import FeatureMap
from deepray.utils.data.input_meta import InputMeta

# if FLAGS.use_dynamic_embedding:
from tensorflow_recommenders_addons import dynamic_embedding as de

Feature = namedtuple(
    'Feature', [
        'name', 'code', 'dtype', 'len', 'default_value', 'use_hash', 'ids', 'vocab', 'boundaries', 'emb_name',
        'emb_size', 'emb_dim', 'emb_reg_l1', 'emb_split', 'emb_reg_l2', 'emb_init', 'emb_mask', 'emb_dynamic',
        'trainable', 'combiner', 'group'
    ]
)


class BaseModel():

  def __init__(self):
    super().__init__()
    self.conf = InputMeta().conf
    self.conf_version = InputMeta().conf_version
    self.field_dict = self.get_fea_field_dict()
    self.feature_map = FeatureMap(feature_map=FLAGS.feature_map, black_list=FLAGS.black_list).feature_map

    self.fea_gpercentile_dict, fea_gcov_dict, fea_geva_dict, self.fea_bpercentile_dict, fea_bcov_dict, fea_beva_dict, self.fea_tag_dict, self.cate_fea_dict, id_fea_dict, search_vocab_list, word_fea_dict = self.get_feature_meta(
    )

    self.save_path = self.get_save_model_path(self.conf.out_path)

    fine_tune = FLAGS.fine_tune
    if fine_tune:
      self.save_path = os.path.join(self.save_path, fine_tune)
      os.makedirs(self.save_path, exist_ok=True)

    if os.path.exists(self.save_path) and not self.conf.only_predict:
      path_to_pb = os.path.join(self.save_path, "saved_model.pb")
      path_to_pbtxt = os.path.join(self.save_path, "saved_model.pbtxt")
      if os.path.exists(path_to_pb) or os.path.exists(path_to_pbtxt):
        logging.info('[MODEL] Model already exists! would OVERWRITE! %s' % self.save_path)
        # sys.exit(0)

    # input相关
    self.use_fea_list = self.feature_map['name'].values.tolist()

    self.hash_fea_dict, self.id_fea_dict = self.conf.hash_fea_bucket_size_dict, self.conf.id_fea_bucket_size_dict
    self.conti_fea_cut = None
    if self.conf.bussiness == 'boss':
      self.conti_fea_cut = self.fea_bpercentile_dict
    elif self.conf.bussiness == 'geek':
      self.conti_fea_cut = self.fea_gpercentile_dict
    else:
      logging.info(f'system exits with code -1! unknown bussiness: {self.conf.bussiness}')
      sys.exit(-1)
    if self.conf.extra_conti_fea_cut:
      self.conti_fea_cut = {**self.conti_fea_cut, **self.conf.extra_conti_fea_cut}
    logging.info(f'conti_fea_cut: {len(self.conti_fea_cut)}')
    logging.info(f'cate_fea_dict: {len(self.cate_fea_dict)}')
    logging.info(f'use_fea_list: {len(self.use_fea_list)}')
    # output相关
    self.target_label_table, self.eva_target_label_table, self.class_weight_table = self.get_target()

  @abc.abstractmethod
  def build_network(self, flags=None, features=None):
    """
    must defined in subclass
    """
    raise NotImplementedError("build_network: not implemented!")

  def get_target(self):
    target_label_table = dict()
    eva_target_label_table = dict()
    class_weight_table = dict()
    for key, value in self.conf.target.items():
      sample_weight = value['total_sample_weight'] if 'total_sample_weight' in value else value['sample_weight']
      label_key = []
      label_value = []
      for k, v in sample_weight.items():
        label_key.append(k)
        label_value.append(0 if v < 0 else 1)

      # 兼容之前的格式，之后会下掉
      if 'label' in value:
        label_key = []
        label_value = []
        for k, v in value['label'].items():
          label_key.append(k)
          label_value.append(v)

      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=label_key, values=label_value, key_dtype=tf.string, value_dtype=tf.int64, name=key + "_target_lookup_1"
      )
      target_label_table[key] = tf.lookup.StaticHashTable(initializer, default_value=0, name=key + "_target_lookup")
      class_weight_table[key] = value["class_weight"]

    for key, value in self.conf.evaluate_target.items():
      target_label = value['label']
      initializer = tf.lookup.KeyValueTensorInitializer(
          keys=list(target_label.keys()),
          values=list(target_label.values()),
          key_dtype=tf.string,
          value_dtype=tf.int64,
          name=key + "_target_lookup_1"
      )
      eva_target_label_table[key] = tf.lookup.StaticHashTable(initializer, default_value=0, name=key + "_target_lookup")
      class_weight_table[key] = 0

    class_weight_table["predict"] = 0

    return target_label_table, eva_target_label_table, class_weight_table

  def get_save_model_path(self, basedir):
    end_date = FLAGS.end_date
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    save_date = "+%02d%02d" % (end_dt.month, end_dt.day)
    save_path = os.path.join(basedir, save_date)
    os.makedirs(save_path, exist_ok=True)
    return save_path

  def get_fea_field_dict(self):
    fea_field_dict = dict()
    fea_field_path = os.path.join(self.conf.conf_path, 'fea_field')
    for file in os.listdir(fea_field_path):
      filename = os.path.join(fea_field_path, file)
      val = []
      #         logger.info("#" + file + "#")
      if not os.path.isfile(filename):
        logging.info('warning: %s is a directory!' % file)
        continue
      with open(filename, 'r') as fr:
        while True:
          line = fr.readline().strip()
          if line == '':
            break
          if line == 'NULL':
            continue
          val.append(line)
          # logging.info(line)
      fea_field_dict[file] = val
    return fea_field_dict

  def get_feature_tag(self):
    path = os.path.join(self.conf.common_config, self.conf.tag_file)
    logging.info(f'fea_tag_path: {path}')
    fea_tag_name_dict = dict()
    f = open(path)
    for line in f.readlines():
      line = line.strip().split('\t')
      if len(line) < 2:
        continue
      fea_tag_name_dict[line[0]] = line[1]
    return fea_tag_name_dict

  def get_cate_fea_vocab_list(self):
    path = os.path.join(self.conf.common_config, self.conf.vocab_dir)
    logging.info(f'fea_vocab_path: {path}')
    cate_fea_vocab_list = dict()
    # cate_fea_path = os.path.join(self.conf.project_path, 'self.conf', 'conf_common', self.conf.cate_fea_dir)

    # logging.info('reading category_feature')
    for file in os.listdir(path):
      filename = os.path.join(path, file)
      val = []
      # logging.info(file)
      if os.path.isdir(filename):
        logging.info('warning: %s is a directory!' % file)
        continue
      with open(filename, 'r') as f:
        for line in f:
          line = line.strip()
          if line == '' or line == 'NULL':
            continue
          val.append(int(line))
      cate_fea_vocab_list[file] = val
    # sys.exit()
    return cate_fea_vocab_list

  def get_search_vocab_list(self):
    path = os.path.join(self.conf.common_config, self.conf.search_dir)
    logging.info(f'search_vocab_path: {path}')
    search_vocab_list = dict()

    for file in os.listdir(path):
      filename = os.path.join(path, file)
      val = []
      # logging.info(file)
      if os.path.isdir(filename):
        logging.info('warning: %s is a directory!' % file)
        continue
      with open(filename, 'r') as f:
        for line in f:
          line = line.strip()
          if line == '' or line == 'NULL':
            continue
          val.append(line.strip().encode('utf-8'))
      search_vocab_list[file] = val
    # sys.exit()
    return search_vocab_list

  def get_feature_meta(self):
    # 299     boss_ret_num_1d3        1       float32 [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 16.0, 28.0, 505.0] 0.3836  0.5177|0.5037|0.5328|0.5377|0.5233|0.521.5233|0.5211       [1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 11.0, 16.0, 26.0, 502.0] 0.6171  0.5|0.5062|0.5|0.5016|0.5|0.4929        0       ,8,102,38,

    # fea_code               string
    # fea_name               string
    # fea_len                string
    # fea_dtype              string
    # gpercentile            string
    # gcov                   string
    # geva                   string
    # bpercentile            string
    # bcov                   string
    # beva                   string
    # def_valu               string
    # fea_tag                string

    path = os.path.join(self.conf.common_config, self.conf.meta_file)
    logging.info(f'fea_meta_path: {path}')

    cate_fea_vocab_list = self.get_cate_fea_vocab_list()
    fea_tag_name_dict = self.get_feature_tag()
    search_vocab_list = self.get_search_vocab_list()

    cols = 'fea_code,fea_name,fea_len,fea_dtype,gpercentile,gcov,geva,bpercentile,bcov,beva,def_valu,fea_tag,dim'
    cols = cols.split(',')
    logging.info(cols)

    cate_fea_dict = dict()
    word_fea_dict = dict()
    fea_id_set = {'job_id', 'boss_id', 'exp_id', 'geek_id', 'addf_id'}
    id_fea_dict = dict()
    word_tag_set = {'word'}

    fea_gpercentile_dict = dict()
    fea_gcov_dict = dict()
    fea_geva_dict = dict()
    fea_bpercentile_dict = dict()
    fea_bcov_dict = dict()
    fea_beva_dict = dict()
    fea_tag_dict = dict()

    f = open(path)
    for line in f.readlines():
      line = line.strip().split('\t')

      # logging.info(line)
      if len(line) < len(cols):
        logging.info(f'column len not match! please check input: {line}')
        continue
      if line[2] == '-1' and not line[2].isdigit():
        logging.info(f'fea_len = -1! please check input: {line}')
        continue
      if line[3] not in ('int64', 'float32', 'string'):
        logging.info(f'invalaid dytype! please check input: {line}')
        continue
      if int(line[2]) > 1 and not line[11]:
        logging.info(f'empty fea_tag! required fea_tag when vector occurs but found null! please check input: {line}')
        continue
      fea_name = line[1]
      fea_len = int(line[2])
      fea_dtype = line[3]
      fea_gpercentile = line[4]
      fea_gcov = line[5]
      fea_geva = line[6]
      fea_bpercentile = line[7]
      fea_bcov = line[8]
      fea_beva = line[9]
      fea_tag = line[11].strip().split(',')
      for tag in fea_tag:
        if tag in fea_tag_name_dict:
          tag_name = fea_tag_name_dict[tag]
          fea_tag_dict[fea_name] = tag_name
        if tag in fea_id_set:
          id_fea_dict[fea_name] = tag
        if tag in word_tag_set:
          word_fea_dict[fea_name] = tag

      if fea_name in fea_tag_dict:
        tag_name = fea_tag_dict[fea_name]
        if tag_name in cate_fea_vocab_list:
          cate_fea_dict[fea_name] = cate_fea_vocab_list[tag_name]
        else:
          logging.warn(f'tag {tag_name} NOT EXIST for cat feature {fea_name}')
          cate_fea_dict[fea_name] = []

      if fea_len == 1 and fea_dtype in ('int64', 'float32') and fea_name not in fea_tag_dict:
        if fea_gpercentile.startswith('[') or (
            fea_gpercentile not in ['', '\"\"'] and not fea_gpercentile.startswith('{')
        ):
          fea_gpercentile = list(set(map(lambda x: float(x), fea_gpercentile.strip('[').strip(']').split(','))))
          fea_gpercentile.sort()
          fea_gpercentile_dict[fea_name] = fea_gpercentile
        if fea_bpercentile.startswith('[') or (
            fea_bpercentile not in ['', '\"\"'] and not fea_bpercentile.startswith('{')
        ):
          fea_bpercentile = list(set(map(lambda x: float(x), fea_bpercentile.strip('[').strip(']').split(','))))
          fea_bpercentile.sort()
          fea_bpercentile_dict[fea_name] = fea_bpercentile
      if fea_gcov != '':
        fea_gcov_dict[fea_name] = fea_gcov
        fea_geva_dict[fea_name] = fea_geva
      if fea_bcov != '':
        fea_bcov_dict[fea_name] = fea_bcov
        fea_beva_dict[fea_name] = fea_beva

    return fea_gpercentile_dict, fea_gcov_dict, fea_geva_dict, fea_bpercentile_dict, fea_bcov_dict, fea_beva_dict, fea_tag_dict, cate_fea_dict, id_fea_dict, search_vocab_list, word_fea_dict

  def make_feature(self, f=None, **kwargs):
    if not f:
      f = Feature(
          None, 0, tf.int64, 1, -3, False, False, None, None, None, 0, 1, 0.00001, None, 0.00001, 'truncated_normal',
          False, False, True, 'mean', None
      )
    copy = f._asdict()
    copy.update(kwargs)
    return Feature(**copy)

  def input_from_features(self):
    input_dict = OrderedDict()
    for code, fname, dtype, length in self.feature_map[["code", "name", "dtype", "length"]].values:
      if code not in input_dict:
        input_dict[code] = tf.keras.Input(shape=(int(length),), name=code, dtype=dtype)
    return input_dict

  def embedding_from_feature(self, features, is_training=True, emb_dict=None):
    if emb_dict is None:
      emb_dict = {}
    emb_reuse_count = 0
    for feature in features:
      fea_emb_name = feature.emb_name
      if not feature.emb_dynamic:  # 非id特征重命名成一个，方便后续进行合并
        fea_emb_name = "not_id_feature"
      else:
        print("DynamicEmbedding:", feature.name, feature.emb_name, is_training)

      if fea_emb_name and fea_emb_name not in emb_dict:
        if is_training:
          initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1. / math.sqrt(feature.emb_dim))
        else:
          initializer = tf.keras.initializers.Zeros()

        if feature.emb_dynamic:  # 现在所有的特征都走de，我们仍使用这个字段标识是否是id特征
          emb = DynamicEmbedding(
              embedding_size=feature.emb_dim,
              mini_batch_regularizer=l2(feature.emb_reg_l2),
              mask_value=feature.default_value,
              key_dtype=tf.int64,
              value_dtype=tf.float32,
              initializer=initializer,
              name='dynamic_' + fea_emb_name
          )
        else:
          if not FLAGS.use_horovod:
            emb = DynamicEmbedding(
                embedding_size=feature.emb_dim,
                mini_batch_regularizer=l2(feature.emb_reg_l2),
                mask_value=feature.default_value,
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                initializer=initializer,
                name="UnifiedDynamicEmbedding",
                # init_capacity=1000000 * 8  # 如果提示hash冲突，调整该参数
            )
          else:
            import horovod.tensorflow as hvd

            gpu_device = ["GPU:0"]
            mpi_size = hvd.size()
            mpi_rank = hvd.rank()
            emb = de.keras.layers.HvdAllToAllEmbedding(
                mpi_size=mpi_size,
                embedding_size=feature.emb_dim,
                key_dtype=tf.int64,
                value_dtype=tf.float32,
                initializer=initializer,
                devices=gpu_device,
                name='DenseUnifiedEmbeddingLayer',
                kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))
            )

        emb_dict[fea_emb_name] = emb
      else:
        emb_reuse_count += 1

    logging.info(f"embedding reuse count: {emb_reuse_count}")
    return emb_dict

  def dense_from_columns(self, features, emb_dict):  # 用于非id特征
    redis_inputs = list()
    for feature in features:
      if feature.boundaries:
        redis_inputs.append(Bucketize(feature.boundaries, name=f"bucket_{feature.code}")(self.input_dict[feature.code]))
      else:
        redis_inputs.append(self.input_dict[feature.code])

    id_tensors = list()
    fea_lens = list()
    for i, input_tensor in enumerate(redis_inputs):
      input_tensor = input_tensor if input_tensor.dtype == tf.int64 else tf.cast(input_tensor, tf.int64)
      id_tensor_prefix_code = tf.constant(int(features[i].code) << 47, dtype=tf.int64)  # 这里用的code
      id_tensor = tf.bitwise.bitwise_xor(input_tensor, id_tensor_prefix_code)  # 前半部分是特征code，后半部分是值，全部合到一起去查询
      id_tensors.append(id_tensor)
      fea_lens.append(features[i].len)

    split_dims_final = list()
    is_sequence_feature = list()  # 标记当前这一段是否是序列
    tmp_sum = 0
    for fea_len in fea_lens:
      if fea_len == 1:
        tmp_sum += 1
      elif fea_len > 1:
        if tmp_sum > 0:  # 如果当前特征是序列，先把之前的非序列加入，再处理当前特征
          split_dims_final.append(tmp_sum)
          is_sequence_feature.append(False)
        split_dims_final.append(fea_len)
        is_sequence_feature.append(True)
        tmp_sum = 0
      else:
        raise ("fea_len must >= 1, which is {}".format(fea_len))
    if tmp_sum > 0:  # 后处理：非序列特征
      split_dims_final.append(tmp_sum)
      is_sequence_feature.append(False)

    id_tensors_concat = Concatenate(axis=1)(id_tensors)
    embedding_outs_concat = emb_dict['not_id_feature'](id_tensors_concat)
    embedding_outs = tf.split(
        embedding_outs_concat, num_or_size_splits=split_dims_final, axis=1, name=f"split_not_id_fea"
    )

    dense_dict = OrderedDict()
    counter_flag = 0
    for i, embedding in enumerate(embedding_outs):
      if is_sequence_feature[i]:
        #             logging.info(f"seq fea: {embedding.get_shape()}, {features[counter_flag].name}")
        embedding_vec = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
        seq_fea = features[counter_flag]
        counter_flag += 1
        dense_dict[seq_fea.name] = embedding_vec
      else:
        simple_fea_embeddings = tf.split(
            embedding, num_or_size_splits=[1] * split_dims_final[i], axis=1, name=f"split_simple_fea_embeddings_{i}"
        )
        for _, simple_fea_embedding in enumerate(simple_fea_embeddings):
          #                 logging.info(f"simple fea: {simple_fea_embedding.get_shape()}, {features[counter_flag].name}")
          simple_fea = features[counter_flag]
          counter_flag += 1
          dense_dict[simple_fea.name] = simple_fea_embedding

    return dense_dict

  def dense_from_columns_id(self, features, emb_dict):  # 用于id特征，因为是cpu处理，只对相同emb_name的特征做合并查询
    merge_dict = defaultdict(list)  # 按照emb_name合并
    feature_set = set()
    for feature in features:
      if feature.name in feature_set:  # 避免重复特征多次查询
        continue
      feature_set.add(feature.name)
      merge_dict[feature.emb_name].append(feature)

    dense_dict = OrderedDict()
    for emb_name, feas in merge_dict.items():
      denses = list()
      splits = list()
      masks = dict()
      for fea in feas:
        if fea.name in dense_dict:
          continue
        dense = self.input_dict[fea.code]

        if fea.len > 1 and fea.combiner:  # 注意，id序列如果有特殊处理，一定要把combiner设置为None
          masks[fea.code] = tf.greater_equal(dense, 0)

        if emb_name in emb_dict:
          denses.append(dense)
          splits.append(fea.len)
      if denses:
        dense_concat = tf.concat(denses, axis=1, name=f"concat_{emb_name}")
        dense_look = emb_dict[emb_name](dense_concat)
        dense_looks = tf.split(dense_look, num_or_size_splits=splits, axis=1, name=f"split_{emb_name}")

        for i, fea in enumerate(feas):
          fea_dense = dense_looks[i]
          if fea.len > 1 and fea.combiner:
            fea_dense = Pooling(combiner=fea.combiner,
                                name=f"{fea.combiner}_{fea.name}")(fea_dense, mask=masks[fea.code])
          dense_dict[fea.name] = fea_dense

    return dense_dict

  def build_features(self):
    # Inputs
    conti_features = self.conti_fea_dict()
    features = dict()
    for code, fname, dtype, length, def_valu in self.feature_map[["code", "name", "dtype", "length",
                                                                  "def_valu"]].values:
      if 'int' in dtype:
        try:
          def_valu = int(def_valu)
        except ValueError as e:
          logging.info(f'[ERROR] default value, {fname} {dtype} {def_valu}')
          def_valu = -3 if length > 1 else 0
      elif 'float' in dtype:
        try:
          def_valu = float(def_valu)
        except ValueError as e:
          logging.info(f'[ERROR] default value, {fname} {dtype} {def_valu}')
          def_valu = -3.0 if length > 1 else 0.0
      if def_valu == "_PAD_":
        print("def:", fname, def_valu)

      use_hash = False
      emb_size = 0
      vocab = None
      boundaries = None
      emb_name = fname
      emb_dynamic = False
      if fname in self.conf.hash_fea_bucket_size_dict:
        use_hash = True
        emb_size = self.conf.hash_fea_bucket_size_dict[fname]
      elif fname in self.conf.id_fea_bucket_size_dict:
        emb_size = self.conf.id_fea_bucket_size_dict[fname]
      elif fname in self.cate_fea_dict:  # 属性特征/属性序列特征，其tag一定要在指定的tag_vocab中，才能确保被调用到
        if self.conf.emb_reuse:
          emb_name = self.fea_tag_dict[fname]
        vocab = self.cate_fea_dict[fname]
        emb_size = len(vocab) + 1
      elif fname in conti_features:  # 分桶特征，不存在序列
        boundaries = conti_features[fname]
        emb_size = len(boundaries) + 1
      else:
        print(fname + " is error feature or is id_fea")

      if fname in self.id_fea_dict:  # 这里会把上面hash中重复的特征给覆盖掉 - id/id序列特征，其tag一定要有4个id的某一个，才能确保被调用到
        emb_name = self.id_fea_dict[fname]  # 这个名字会重复吧
        emb_dynamic = True
        use_hash = False
        emb_size = 0

      features[fname] = self.make_feature(
          name=fname,
          code=code,
          dtype=dtype,
          len=int(length),
          default_value=def_valu,
          use_hash=use_hash,
          vocab=vocab,
          boundaries=boundaries,
          emb_name=emb_name,
          emb_size=emb_size,
          emb_dynamic=emb_dynamic,
          emb_reg_l2=self.conf.emb_reg_l2
      )
    return features

  def conti_fea_dict(self):
    if self.conf.bussiness == 'boss':
      conti_fea = self.fea_bpercentile_dict
    elif self.conf.bussiness == 'geek':
      conti_fea = self.fea_gpercentile_dict
    else:
      logging.info(f'system exits with code -1! unknown bussiness: {self.conf.bussiness}')
      sys.exit(-1)
    if self.conf.extra_conti_fea_cut:
      conti_fea = {**conti_fea, **self.conf.extra_conti_fea_cut}
    return conti_fea

  # 读取geek nn的特征交叉信息
  def get_geek_nn_compo(self):
    nn_cnt = dict()
    nn_path = os.path.join(self.conf.conf_path, self.conf.network_compo_dir, 'geek')
    with open(nn_path, 'r') as fr:
      for line in fr.readlines():
        line = line.strip().split('\t')
        fields = line[0].split(',')
        fea_len = list(map(lambda x: int(x), line[1].split(',')))
        for field in fields:
          if field not in nn_cnt:
            nn_cnt[field] = fea_len
    # {'geek_base': [32]}
    return nn_cnt

  # 读取job nn的特征交叉信息
  def get_job_nn_compo(self):
    nn_cnt = dict()
    nn_path = os.path.join(self.conf.conf_path, self.conf.network_compo_dir, 'job')
    with open(nn_path, 'r') as fr:
      for line in fr.readlines():
        line = line.strip().split('\t')
        fields = line[0].split(',')
        fea_len = list(map(lambda x: int(x), line[1].split(',')))
        for field in fields:
          if field not in nn_cnt:
            nn_cnt[field] = fea_len
    # {'job_base': [32]}
    return nn_cnt

  def cosin(self, input_query, input_doc):
    query_norm = K.sqrt(K.sum(K.square(input_query), axis=-1))
    doc_norm = K.sqrt(K.sum(K.square(input_doc), axis=-1))
    query_doc = K.sum(input_query * input_doc, axis=-1)
    cosin = query_doc / (query_norm * doc_norm + 1e-8)
    return cosin
