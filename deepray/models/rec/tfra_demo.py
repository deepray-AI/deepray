import tensorflow as tf

from tensorflow.keras.layers import (Layer, Input, Concatenate, Dense, Flatten, Lambda)
from tensorflow_recommenders_addons import dynamic_embedding as de


class DeepLayer(Layer):

  def __init__(self, hidden_dim, layer_num, out_dim):
    self.layers = []
    self.hidden_dim = hidden_dim
    self.layer_num = layer_num
    self.out_dim = out_dim
    for i in range(layer_num):
      self.layers.append(Dense(hidden_dim, "relu"))
    self.layers.append(Dense(out_dim, "sigmoid"))
    super(DeepLayer, self).__init__()

  def call(self, inputs):
    output = inputs
    for layer in self.layers:
      output = layer(output)
    return output  # (batch, out_dim)

  def get_config(self):
    config = super().get_config()
    config.update({
        "hidden_dim": self.hidden_dim,
        "layer_num": self.layer_num,
        "out_dim": self.out_dim,
    })
    return config


# 构建model
def build_keras_model(is_training, mpi_size, mpi_rank):
  # 初始化参数
  embedding_size = 8

  if is_training:
    initializer = tf.keras.initializers.VarianceScaling()
  else:
    initializer = tf.keras.initializers.Zeros()
  gpu_device = ["GPU:0"]
  cpu_device = ["CPU:0"]

  dense_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
      mpi_size=mpi_size,
      embedding_size=embedding_size,
      key_dtype=tf.int32,
      value_dtype=tf.float32,
      initializer=initializer,
      devices=gpu_device,
      name='DenseUnifiedEmbeddingLayer',
      kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))
  )

  sparse_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
      mpi_size=mpi_size,
      embedding_size=embedding_size,
      key_dtype=tf.int64,
      value_dtype=tf.float32,
      initializer=initializer,
      devices=cpu_device,
      name='SparseUnifiedEmbeddingLayer',
      kv_creator=de.CuckooHashTableCreator(saver=de.FileSystemSaver(proc_size=mpi_size, proc_rank=mpi_rank))
  )

  # 输入层
  dense_input_dict = {"movie_genres": {'code': 1111, 'dim': 1}, "user_gender": {'code': 2222, 'dim': 1}}
  sparse_input_dict = {"movie_id": {'code': 3333, 'dim': 1}, "user_id": {'code': 4444, 'dim': 1}}

  inputs = dict()
  embedding_outs = []

  # 定义 gpu embedding层
  # 主要思路是合并输入进行embedding查询，最大化利用gpu并行能力，并降低kernel launch time
  # 由于 gpu dynamic embedding的动态增机制，请务必设置os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"，以保证显存不会被tensorflow graph预读。
  ###################################################
  dense_input_tensors = list()
  dense_input_split_dims = list()
  for input_name in dense_input_dict.keys():
    dense_input_tensor = Input(shape=(1,), dtype=tf.int32, name=input_name)
    inputs[input_name] = dense_input_tensor

    input_tensor_prefix_code = int(dense_input_dict[input_name]["code"]) << 17
    # dense_input_tensor = tf.bitwise.bitwise_xor(dense_input_tensor, input_tensor_prefix_code)
    # xor可以用加法替代，方便后续TRT、openvino的优化
    dense_input_tensor = tf.add(dense_input_tensor, input_tensor_prefix_code)
    dense_input_tensors.append(dense_input_tensor)
    dense_input_split_dims.append(dense_input_dict[input_name]["dim"])

  tmp_sum = 0
  dense_input_split_dims_final = []
  dense_input_is_sequence_feature = []
  for dim in dense_input_split_dims:
    if dim == 1:
      tmp_sum = tmp_sum + 1
    elif dim > 1:
      if tmp_sum > 0:
        dense_input_split_dims_final.append(tmp_sum)
        dense_input_is_sequence_feature.append(False)
      dense_input_split_dims_final.append(dim)
      dense_input_is_sequence_feature.append(True)
      tmp_sum = 0
    else:
      raise ("dim must >= 1, which is {}".format(dim))
  if tmp_sum > 0:
    dense_input_split_dims_final.append(tmp_sum)
    dense_input_is_sequence_feature.append(False)

  dense_input_tensors_concat = Concatenate(axis=1)(dense_input_tensors)
  dense_embedding_out_concat = dense_embedding_layer(dense_input_tensors_concat)
  ###################################################
  # gpu embedding部分结束

  # 定义 cpu embedding层
  # id类特征维度空间大，显存不够用，放在主机内存
  ###################################################
  sparse_input_tensors = list()
  sparse_input_split_dims = list()
  for input_name in sparse_input_dict.keys():
    sparse_input_tensor = Input(shape=(1,), dtype=tf.int64, name=input_name)
    inputs[input_name] = sparse_input_tensor

    input_tensor_prefix_code = int(sparse_input_dict[input_name]["code"]) << 47
    # id_tensor = tf.bitwise.bitwise_xor(sparse_input_tensor, input_tensor_prefix_code)
    # xor可以用加法替代，方便后续TRT、openvino的优化
    sparse_input_tensor = tf.add(sparse_input_tensor, input_tensor_prefix_code)
    sparse_input_tensors.append(sparse_input_tensor)
    sparse_input_split_dims.append(sparse_input_dict[input_name]["dim"])

  tmp_sum = 0
  sparse_input_split_dims_final = []
  sparse_input_is_sequence_feature = []
  for dim in sparse_input_split_dims:
    if dim == 1:
      tmp_sum = tmp_sum + 1
    elif dim > 1:
      if tmp_sum > 0:
        sparse_input_split_dims_final.append(tmp_sum)
        sparse_input_is_sequence_feature.append(False)
      sparse_input_split_dims_final.append(dim)
      sparse_input_is_sequence_feature.append(True)
      tmp_sum = 0
    else:
      raise ("dim must >= 1, which is {}".format(dim))
  if tmp_sum > 0:
    sparse_input_split_dims_final.append(tmp_sum)
    sparse_input_is_sequence_feature.append(False)

  sparse_input_tensors_concat = Concatenate(axis=1)(sparse_input_tensors)
  sparse_embedding_out_concat = sparse_embedding_layer(sparse_input_tensors_concat)
  ###################################################
  # cpu embedding部分结束

  # 接下来是特别处理向量特征
  # split_dims和is_sequence_feature用来辨识向量特征
  ###################################################
  embedding_out = list()
  embedding_out.extend(
      tf.split(dense_embedding_out_concat, dense_input_split_dims_final, axis=1)
  )  # (feature_combin_num, (batch, dim, emb_size))
  embedding_out.extend(
      tf.split(sparse_embedding_out_concat, sparse_input_split_dims_final, axis=1)
  )  # (feature_combin_num, (batch, dim, emb_size))
  assert ((len(dense_input_is_sequence_feature) + len(sparse_input_is_sequence_feature)) == len(embedding_out))
  is_sequence_feature = dense_input_is_sequence_feature + sparse_input_is_sequence_feature
  for i, embedding in enumerate(embedding_out):
    if is_sequence_feature[i] == True:
      # 处理向量特征获得的embedding
      embedding_vec = tf.math.reduce_mean(
          embedding, axis=1, keepdims=True
      )  # (feature_combin_num, (batch, x, emb_size))
    else:
      embedding_vec = embedding
    embedding_outs.append(embedding_vec)

  ###################################################
  ###################################################
  # embedding层 部分结束
  ###################################################
  ###################################################

  # 算法后续部分
  embeddings_concat = Flatten()(Concatenate(axis=1)(embedding_outs))

  outs = DeepLayer(256, 1, 1)(embeddings_concat)
  outs = Lambda(lambda x: x, name="user_rating")(outs)

  model = tf.keras.Model(inputs=inputs, outputs=outs)
  return model
