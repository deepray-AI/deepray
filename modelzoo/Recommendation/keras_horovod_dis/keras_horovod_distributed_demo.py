import os

import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Layer, Input, Concatenate, Dense, Flatten, Lambda)
from tensorflow_recommenders_addons import dynamic_embedding as de

from deepray.datasets.movielens import Movielens100kRating

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 很重要！

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[hvd.local_rank()],
                                         True)


# 示例一下自定义callback,可选。
class log_callback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs=None):
    keys = list(logs.keys())
    print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

  def on_epoch_end(self, epoch, logs=None):
    keys = list(logs.keys())
    print("End epoch {} of training; got log keys: {}".format(epoch, keys))


# ckpt callback要重写，不然无法保存rank0以外的embedding


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
    kv_creator=de.CuckooHashTableCreator(
      saver=de.FileSystemSaver(proc_size=mpi_size,
                               proc_rank=mpi_rank)
    ))

  sparse_embedding_layer = de.keras.layers.HvdAllToAllEmbedding(
    mpi_size=mpi_size,
    embedding_size=embedding_size,
    key_dtype=tf.int64,
    value_dtype=tf.float32,
    initializer=initializer,
    devices=cpu_device,
    name='SparseUnifiedEmbeddingLayer',
    kv_creator=de.CuckooHashTableCreator(
      saver=de.FileSystemSaver(proc_size=mpi_size,
                               proc_rank=mpi_rank)
    ))

  # 输入层
  dense_input_dict = {
    "movie_genres": {
      'code': 1111,
      'dim': 1
    },
    "user_gender": {
      'code': 2222,
      'dim': 1
    }
  }
  sparse_input_dict = {
    "movie_id": {
      'code': 3333,
      'dim': 1
    },
    "user_id": {
      'code': 4444,
      'dim': 1
    }
  }

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
  sparse_embedding_out_concat = sparse_embedding_layer(
    sparse_input_tensors_concat)
  ###################################################
  # cpu embedding部分结束

  # 接下来是特别处理向量特征
  # split_dims和is_sequence_feature用来辨识向量特征
  ###################################################
  embedding_out = list()
  embedding_out.extend(
    tf.split(dense_embedding_out_concat, dense_input_split_dims_final,
             axis=1))  # (feature_combin_num, (batch, dim, emb_size))
  embedding_out.extend(
    tf.split(sparse_embedding_out_concat,
             sparse_input_split_dims_final,
             axis=1))  # (feature_combin_num, (batch, dim, emb_size))
  assert ((len(dense_input_is_sequence_feature) +
           len(sparse_input_is_sequence_feature)) == len(embedding_out))
  is_sequence_feature = dense_input_is_sequence_feature + sparse_input_is_sequence_feature
  for i, embedding in enumerate(embedding_out):
    if is_sequence_feature[i] == True:
      # 处理向量特征获得的embedding
      embedding_vec = tf.math.reduce_mean(
        embedding, axis=1,
        keepdims=True)  # (feature_combin_num, (batch, x, emb_size))
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

  optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4 * mpi_size,
                                       amsgrad=False)
  optimizer = de.DynamicEmbeddingOptimizer(optimizer, hvd_synchronous=True)

  # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
  # uses hvd.DistributedOptimizer() to compute gradients.
  model.compile(optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=tf.keras.metrics.AUC(num_thresholds=1000,
                                             summation_method='minoring'),
                experimental_run_tf_function=False)

  return model


def train(model, model_dir, savedmodel_dir):
  data_pipe = Movielens100kRating()

  # 因为设定了TF可见GPU，所以只有GPU0

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  # ModelCheckpoint需要特殊修改才能保存hvd其他rank的kv
  ckpt_callback = de.keras.callbacks.DEHvdModelCheckpoint(
    savedmodel_dir + "/weights_epoch{epoch:03d}_loss{loss:.4f}",
    save_freq=2,
    verbose=1,
    options=options)
  # hvd callback用于广播rank0的初始化器产生的值
  hvd_opt_init_callback = de.keras.callbacks.DEHvdBroadcastGlobalVariablesCallback(root_rank=0)
  callbacks_list = [hvd_opt_init_callback, ckpt_callback]

  if hvd.rank() == 0:
    callbacks_list.extend([log_callback(), tensorboard_callback])

  dataset = data_pipe("movielens/100k-ratings", 4096, is_training=True)
  model.fit(dataset,
            callbacks=callbacks_list,
            steps_per_epoch=10 // hvd.size(),
            epochs=100,
            verbose=1 if hvd.rank() == 0 else 0)
  model.evaluate(dataset)


def find_latest_savedmodel(dir):
  '''查找目录下最新的文件'''
  file_lists = os.listdir(dir)
  file_lists.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn)
  if os.path.exists(dir + "/" + fn + "/variables") else 0)
  file = ''
  if len(file_lists) > 0:
    print('最新的模型文件为： ' + file_lists[-1])
    file = os.path.join(dir, file_lists[-1])
    print('完整路径：', file)
  return file


def export_to_savedmodel(model, savedmodel_dir):
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  if not os.path.exists(savedmodel_dir):
    os.mkdir(savedmodel_dir)

  # 都存一起会导致文件冲突
  if hvd.rank() == 0:
    tf.keras.models.save_model(model,
                               savedmodel_dir,
                               overwrite=True,
                               include_optimizer=True,
                               save_traces=True,
                               options=options)
  else:
    de_dir = os.path.join(savedmodel_dir, "variables", "TFRADynamicEmbedding")
    for layer in model.layers:
      if hasattr(layer, "params"):
        # 保存embedding参数
        layer.params.save_to_file_system(dirpath=de_dir)
        # 保存优化器参数
        opt_de_vars = layer.optimizer_vars.as_list() if hasattr(
          layer.optimizer_vars, "as_list") else layer.optimizer_vars
        for opt_de_var in opt_de_vars:
          opt_de_var.save_to_file_system(dirpath=de_dir)


def export_for_serving(model, export_dir):
  if not os.path.exists(export_dir):
    os.mkdir(export_dir)

  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  # 记得删除优化器参数

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

  if hvd.rank() == 0:
    tf.keras.models.save_model(
      model,
      export_dir,
      overwrite=True,
      include_optimizer=False,
      options=options,
      signatures={
        'serving_default':
          serve.get_concrete_function(*arg_specs, **kwarg_specs)
      },
    )
  else:
    de_dir = os.path.join(export_dir, "variables", "TFRADynamicEmbedding")
    for layer in model.layers:
      if hasattr(layer, "params"):
        layer.params.save_to_file_system(dirpath=de_dir,
                                         mpi_size=hvd.size(),
                                         mpi_rank=hvd.rank())

  if hvd.rank() == 0:
    # 修改计算图变成单机版
    from tensorflow.python.saved_model import save as tf_save
    K.clear_session()
    de.enable_inference_mode()
    export_model = build_keras_model(is_training=False, mpi_size=1, mpi_rank=0)
    tf_save.save_and_return_nodes(obj=export_model,
                                  export_dir=export_dir,
                                  options=options,
                                  experimental_skip_checkpoint=True)


def export_for_arsenal_serving(export_model, export_dir):
  options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])

  # 记得删除优化器参数

  # 有效减少模型保存体积，但需要所有的自定义layer都实现get_config方法
  # 为了获取signature的特殊手段
  def save_spec():
    # tf 2.6 以上版本
    if hasattr(export_model, 'save_spec'):
      return export_model.save_spec()
    else:
      arg_specs = list()
      kwarg_specs = dict()
      for i in export_model.inputs:
        arg_specs.append(i.type_spec)
      return [arg_specs], kwarg_specs

  @tf.function
  def serve(*args, **kwargs):
    return export_model(*args, **kwargs)

  arg_specs, kwarg_specs = save_spec()

  if hvd.rank() == 0:
    tf.keras.models.save_model(
      export_model,
      export_dir,
      overwrite=True,
      include_optimizer=False,
      save_traces=False,
      options=options,
      signatures={
        'serving_default':
          serve.get_concrete_function(*arg_specs, **kwarg_specs)
      },
    )
  else:
    de_dir = os.path.join(export_dir, "variables", "TFRADynamicEmbedding")
    for layer in export_model.layers:
      if hasattr(layer, "params"):
        layer.params.save_to_file_system(dirpath=de_dir,
                                         mpi_size=hvd.size(),
                                         mpi_rank=hvd.rank())


def main():
  model_dir = "./model_dir"
  savedmodel_dir = model_dir + "/tf2_savedmodel"
  ckpt_dir = model_dir + "/tf1_ckpt"
  export_dir = "./export_dir"

  model = build_keras_model(is_training=True,
                            mpi_size=hvd.size(),
                            mpi_rank=hvd.rank())

  if os.path.exists(savedmodel_dir):
    savedmodel_file = find_latest_savedmodel(savedmodel_dir)
    if savedmodel_file != '':
      # Keras默认不构建优化器参数
      model.optimizer._create_all_weights(model.trainable_variables)
      model.load_weights(savedmodel_file)

  # 内含 ckpt_callback，对于tf2，checkpoint的存储格式为savemodel
  train(model, model_dir, savedmodel_dir)

  # save tf 2 ckpt
  export_to_savedmodel(model, savedmodel_dir + "/my_savedmodel")
  # 手动改图
  export_for_serving(model, export_dir)
  # 导出模型上线的时候需要改图
  # 在可以用脚本改图（具体代码见同目录下的脚本，导出模型后，在arsenal上线前自动执行）
  # export_for_arsenal_serving(model, export_dir)
  # 脚本改图暂不支持分布式
  # os.system('python ./MPI_TFRA_DE_all_in_one.py --model_dir={}'.format(export_dir))
  # os.system('python ./TFRA_DE_model_to_serve.py --model_dir={}'.format(export_dir))


if __name__ == "__main__":
  main()
