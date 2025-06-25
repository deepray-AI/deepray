import sys
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

import deepray as dp
from . import kv_variable_ops
from .group_embedding_types import DistStrategy, get_group_lookup_strategy

gen_group_embedding_ops = tf.load_op_library(resource_loader.get_path_to_datafile("../_group_embedding_ops.so"))

__all__ = ["group_embedding_lookup", "group_embedding_lookup_sparse"]


# for GPU EV group_lookup_dense
def group_embedding_var_lookup_dense(params, dense_values, dimensions, ev_init_value=None):
  if ev_init_value is not None:
    default_value = ev_init_value
    is_use_default_value_tensor = True
  else:
    default_value = ops.convert_to_tensor(1.0)
    is_use_default_value_tensor = False
  return gen_group_embedding_ops.group_embedding_var_lookup_dense(
    params, dense_values, default_value, dimensions, is_use_default_value_tensor
  )


# for GPU EV group_lookup
def group_embedding_var_lookup(
  params,
  sp_values,
  sp_indices,
  sp_weights,
  combiners,
  batch_size,
  dimensions,
  ignore_weights,
  is_sequence=False,
  ev_init_value=None,
):
  if ev_init_value is not None:
    default_value = ev_init_value
    is_use_default_value_tensor = True
  else:
    default_value = ops.convert_to_tensor(1.0)
    is_use_default_value_tensor = False
  if ignore_weights:
    sp_weight = ops.convert_to_tensor(1.0)
    sp_weights = [sp_weight for _ in range(len(sp_values))]
  return gen_group_embedding_ops.group_embedding_var_lookup(
    params,
    sp_values,
    sp_indices,
    sp_weights,
    batch_size,
    default_value,
    combiners,
    dimensions,
    ignore_weights=ignore_weights,
    is_use_default_value_tensor=is_use_default_value_tensor,
    is_sequence=is_sequence,
  )


def group_embedding_lookup(params, ids, partition_strategy="mod", name=None):
  """
  This interface is designed for fused multiple embedding lookup.
  Args:
    params: list, tuple
            a list or tuple of trainable *Variable* or *EmbeddingVariable*.
    ids: list, tuple
            a list or tuple of tf.SparseTensor or tf.Tensor.
            btw RaggedTensor is preferred.
    name: The operations name
  Returns
  -------
  emb_vec: list
          a list of tf.Tensor(the results of lookup).
  """

  if params is None:
    raise ValueError("params must be specified")
  if not isinstance(params, list):
    params = [params]
  for index, param in enumerate(params):
    if isinstance(param, dp.layers.embedding_variable.EmbeddingVariable):
      params[index] = param.embedding_variable

  if len(params) != len(ids):
    raise ValueError("len of params must be equal to len of ids")

  ## Currently not doing unique
  strategy = get_group_lookup_strategy()

  if strategy == DistStrategy.LOCALIZED:
    emb_vec = [None for _ in range(len(params))]

    ev_group_id_map = {}
    tf_group_id_map = {}
    ev_group_id = 0
    tf_group_id = 0
    is_ev_list = [False for _ in range(len(params))]
    params_idx_map = {}

    for index, param in enumerate(params):
      params_idx_map[param.ref()] = index

      if isinstance(param, kv_variable_ops.EmbeddingVariable):
        is_ev_list[index] = True
        dim = param.shape[0]
        if dim not in ev_group_id_map:
          ev_group_id_map[dim] = ev_group_id
          ev_group_id += 1
      else:  # tensorflow variable
        dim = param.shape[1]
        if dim not in tf_group_id_map:
          tf_group_id_map[dim] = tf_group_id
          tf_group_id += 1

    if ev_group_id > 0:
      ev_ids = [[] for _ in range(ev_group_id)]
      ev_handlers = [[] for _ in range(ev_group_id)]
      ev_dimensions = [0 for _ in range(ev_group_id)]
      output_index_list = [[] for _ in range(ev_group_id)]

      for index, ev_flag in enumerate(is_ev_list):
        if not ev_flag:
          continue
        param = params[index]
        dim = param.shape[0]
        group_id = ev_group_id_map[dim]
        ev_id = ids[index]

        ev_dimensions[group_id] = dim
        resource_variable_ops.variable_accessed(param)
        ev_handlers[group_id].append(param.handle)
        ev_ids[group_id].append(array_ops.reshape(ev_id, [-1]))
        output_index_list[group_id].append(params_idx_map[param.ref()])

      for group_id in range(ev_group_id):
        dim = ev_dimensions[group_id]
        output_index = output_index_list[group_id]
        with ops.name_scope(name, "localized_group_embedding_lookup_ev_dim{}".format(dim), params + ids) as name_scope:
          outputs = group_embedding_var_lookup_dense(ev_handlers[group_id], ev_ids[group_id], dim)[0]
          for idx, output in zip(output_index, outputs):
            emb_vec[idx] = output

    if tf_group_id > 0:
      tf_ids = [[] for _ in range(tf_group_id)]
      tf_handlers = [[] for _ in range(tf_group_id)]
      tf_dimensions = [0 for _ in range(tf_group_id)]
      output_index_list = [[] for _ in range(tf_group_id)]

      for index, ev_flag in enumerate(is_ev_list):
        if ev_flag:
          continue
        param = params[index]
        dim = param.shape[1]
        group_id = tf_group_id_map[dim]
        tf_id = ids[index]

        tf_dimensions[group_id] = dim
        tf_handlers[group_id].append(param)
        tf_ids[group_id].append(array_ops.reshape(tf_id, [-1]))
        output_index_list[group_id].append(params_idx_map[param.ref()])

      for group_id in range(tf_group_id):
        dim = tf_dimensions[group_id]
        output_index = output_index_list[group_id]
        with ops.name_scope(
          name, "localized_group_embedding_lookup_variable_dim{}".format(dim), params + ids
        ) as name_scope:
          outputs = gen_group_embedding_ops.group_variable_lookup_dense(tf_handlers[group_id], tf_ids[group_id], dim)[0]
          for idx, output in zip(output_index, outputs):
            emb_vec[idx] = output

  else:
    raise ValueError("Unrecognized strategy, expected collective, given{}".format(strategy))

  return emb_vec


def group_embedding_lookup_sparse(
  params,
  sp_ids,
  combiners,
  sp_weights=None,
  partition_strategy="mod",
  is_sequence=False,
  params_num_per_group=sys.maxsize,
  name=None,
):
  """
  This interface is designed for fused multiple embedding lookup.
  Args:
    params: list, tuple
            a list or tuple of trainable *Variable* or *EmbeddingVariable*.
    sp_ids: list, tuple
            a list or tuple of tf.SparseTensor or tf.RaggedTensor.
            btw RaggedTensor is preferred.
    combiners: list, tuple
            a list or tuple of string to specify the combiner of each embedding lookup,
            supported args is *sum* or *mean*
    sp_weights: list, tuple
             a list or tuple of tf.SparseTensor used for embedding lookup.
    is_sequence: bool
              return list of `Tensor` of shape `[batch_size, D]` when is False
              return list of `Tensor` of shape `[batch_size, T, D]` when is True
    params_num_per_group: int
              The number of params in GroupEmbedding op.Function will schedule len(params) // params_num_per_group + 1
              GroupEmbedding Op. Default setting would launch one Op containing all params which is suitable for GPU scenarios
              to maximize the GPU utilization.On the contrast, you could set value to 1 when Op
              is placed on CPU so as to maximize inter parallelism.
    name: The operations name
  Returns
  -------
  emb_vec: list
          a list of tf.Tensor(the results of lookup).
  """

  if combiners is None:
    logging.warn('The default value of combiner will change from "mean" to "sqrtn" after 2016/11/01.')
    combiners = ["mean"] * len(params)
  if not isinstance(combiners, list):
    combiners = [combiners]
  for combiner in combiners:
    if combiner not in ("mean", "sum"):
      raise ValueError("combiners must be one of 'mean', 'sum'")

  if params is None:
    raise ValueError("params must be specified")
  if not isinstance(params, list):
    params = [params]

  # Currently do not support PartitionedVariable.
  for index, param in enumerate(params):
    if isinstance(param, variables.PartitionedVariable):
      tmp_param = list(param)
      if len(tmp_param) != 1:
        raise TypeError("PartitionedVariable not support in 'group_embedding_lookup_sparse'. ")
      params[index] = tmp_param[0]
    elif isinstance(param, dp.layers.embedding_variable.EmbeddingVariable):
      params[index] = param.embedding_variable

  ignore_weights = sp_weights is None

  if len(combiners) != len(sp_ids):
    raise ValueError("len of combiners must be equal to len of sp_ids")
  if len(combiners) != len(params):
    raise ValueError("len of combiners must be equal to len of params")
  if not ignore_weights:
    if len(combiners) != len(sp_weights):
      raise ValueError("len of combiners must be equal to len of sp_weights")

  strategy = get_group_lookup_strategy()
  if strategy == DistStrategy.SOK:
    import horovod.tensorflow as hvd

    should_shard = False
    if len(params) > hvd.size():
      should_shard = True
      global_size = hvd.size()
    if should_shard:
      for index, param in enumerate(params):
        param.target_gpu = index % global_size
    else:
      for index, param in enumerate(params):
        param.target_gpu = -1

    try:
      from sparse_operation_kit import experiment as sok
    except:
      raise ImportError("sparse_operation_kit is not found while group_embedding strategy is given `collective`")
    with ops.name_scope(name, "group_embedding_lookup", params + sp_ids) as name_scope:
      emb_vec = sok.lookup_sparse(params, sp_ids, combiners=combiners)
  elif strategy == DistStrategy.HB:
    emb_vec = []
    with ops.name_scope(name, "group_embedding_lookup", params + sp_ids) as name_scope:
      for idx, embedding in enumerate(params):
        if not ignore_weights:
          sp_weight = sp_weights[idx]
        else:
          sp_weight = None
        emb_vec.append(embedding_lookup_sparse(embedding, sp_ids[idx], sp_weight, combiner=combiners[idx]))

  elif strategy == DistStrategy.LOCALIZED:
    emb_vec = [None for _ in range(len(params))]

    ev_group_id_map = {}
    tf_group_id_map = {}
    ev_group_id = 0
    tf_group_id = 0
    is_ev_list = [False for _ in range(len(params))]
    params_idx_map = defaultdict(list)  # queue

    for index, param in enumerate(params):
      params_idx_map[param.ref()].append(index)
      sp_id = sp_ids[index]
      if not isinstance(sp_id, sparse_tensor.SparseTensor):
        try:  # assume RaggedTensor
          sp_id = sp_id.to_sparse()
          sp_ids[index] = sp_id
        except:
          raise ValueError("sp_id is neither SparseTensor nor RaggedTensor!")

      if not ignore_weights:
        sp_weight = sp_weights[index]
        if sp_weight is not None:
          if not isinstance(sp_weight, sparse_tensor.SparseTensor):
            raise TypeError("sp_weights must be either None or SparseTensor")
          sp_id.values.get_shape().assert_is_compatible_with(sp_weight.values.get_shape())
          sp_id.indices.get_shape().assert_is_compatible_with(sp_weight.indices.get_shape())
          sp_id.dense_shape.get_shape().assert_is_compatible_with(sp_weight.dense_shape.get_shape())

      if isinstance(param, kv_variable_ops.EmbeddingVariable):
        is_ev_list[index] = True
        dim = param.shape[0]
        if dim not in ev_group_id_map:
          ev_group_id_map[dim] = ev_group_id
          ev_group_id += 1
      else:
        # tensorflow variable
        dim = param.shape[1]
        if dim not in tf_group_id_map:
          tf_group_id_map[dim] = tf_group_id
          tf_group_id += 1

    if ev_group_id > 0:
      ev_sp_values = [[] for _ in range(ev_group_id)]
      ev_sp_indices = [[] for _ in range(ev_group_id)]
      ev_sp_weights = [[] for _ in range(ev_group_id)]
      ev_dense_shapes = [[] for _ in range(ev_group_id)]
      ev_handlers = [[] for _ in range(ev_group_id)]
      ev_dimensions = [0 for _ in range(ev_group_id)]
      ev_combiners = ["mean" for _ in range(ev_group_id)]
      output_index_list = [[] for _ in range(ev_group_id)]

      for index, ev_flag in enumerate(is_ev_list):
        if not ev_flag:
          continue
        param = params[index]
        dim = param.shape[0]
        group_id = ev_group_id_map[dim]
        sp_id = sp_ids[index]
        combiner = combiners[index]

        ev_combiners[group_id] = combiner
        ev_dimensions[group_id] = dim
        resource_variable_ops.variable_accessed(param)
        ev_handlers[group_id].append(param.handle)
        ev_sp_values[group_id].append(sp_id.values)
        ev_sp_indices[group_id].append(sp_id.indices)
        ev_dense_shapes[group_id].append(sp_id.dense_shape)
        output_index_list[group_id].append(params_idx_map[param.ref()].pop(0))

        if not ignore_weights:
          sp_weight = sp_weights[index]
          ev_sp_weights[group_id].append(sp_weight.values)

      for group_id in range(ev_group_id):
        dim = ev_dimensions[group_id]
        output_index = output_index_list[group_id]

        (num_sub_group, num_remainder) = divmod(len(ev_handlers[group_id]), params_num_per_group)
        for j in range(num_sub_group):
          sub_ev_sp_weight = (
            [None for _ in range(params_num_per_group)]
            if ignore_weights
            else (ev_sp_weights[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group]
          )
          with ops.name_scope(
            name, "localized_group_embedding_lookup_ev_dim{}_{}".format(dim, j), params + sp_ids
          ) as name_scope:
            outputs = group_embedding_var_lookup(
              (ev_handlers[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              (ev_sp_values[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              (ev_sp_indices[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              sub_ev_sp_weight,
              ev_combiners[group_id],
              (ev_dense_shapes[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              dim,
              ignore_weights,
              is_sequence,
            )[0]

            for idx, output in zip(output_index[j * params_num_per_group : (j + 1) * params_num_per_group], outputs):
              emb_vec[idx] = output

        if num_remainder > 0:
          sub_ev_sp_weight = (
            [None for _ in range(num_remainder)] if ignore_weights else (ev_sp_weights[group_id])[-num_remainder:]
          )
          with ops.name_scope(
            name, "localized_group_embedding_lookup_ev_dim{}".format(dim), params + sp_ids
          ) as name_scope:
            outputs = group_embedding_var_lookup(
              (ev_handlers[group_id])[-num_remainder:],
              (ev_sp_values[group_id])[-num_remainder:],
              (ev_sp_indices[group_id])[-num_remainder:],
              sub_ev_sp_weight,
              ev_combiners[group_id],
              (ev_dense_shapes[group_id])[-num_remainder:],
              dim,
              ignore_weights,
              is_sequence,
            )[0]

            for idx, output in zip(output_index[-num_remainder:], outputs):
              emb_vec[idx] = output

    if tf_group_id > 0:
      tf_sp_values = [[] for _ in range(tf_group_id)]
      tf_sp_indices = [[] for _ in range(tf_group_id)]
      tf_sp_weights = [[] for _ in range(tf_group_id)]
      tf_dense_shape = [[] for _ in range(tf_group_id)]
      tf_handlers = [[] for _ in range(tf_group_id)]
      tf_dimensions = [0 for _ in range(tf_group_id)]
      tf_combiners = ["mean" for _ in range(tf_group_id)]
      output_index_list = [[] for _ in range(tf_group_id)]

      for index, ev_flag in enumerate(is_ev_list):
        if ev_flag:
          continue
        param = params[index]
        dim = param.shape[1]
        group_id = tf_group_id_map[dim]
        sp_id = sp_ids[index]
        combiner = combiners[index]

        tf_combiners[group_id] = combiner
        tf_dimensions[group_id] = dim
        tf_handlers[group_id].append(param)
        tf_sp_values[group_id].append(sp_id.values)
        tf_sp_indices[group_id].append(sp_id.indices)
        tf_dense_shape[group_id].append(sp_id.dense_shape)
        output_index_list[group_id].append(params_idx_map[param].pop(0))

        if not ignore_weights:
          sp_weight = sp_weights[index]
          tf_sp_weights[group_id].append(sp_weight.values)

      for group_id in range(tf_group_id):
        dim = tf_dimensions[group_id]
        output_index = output_index_list[group_id]

        (num_sub_group, num_remainder) = divmod(len(tf_handlers[group_id]), params_num_per_group)
        for j in range(num_sub_group):
          sub_tf_sp_weight = (
            [None for _ in range(params_num_per_group)]
            if ignore_weights
            else (tf_sp_weights[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group]
          )
          with ops.name_scope(
            name, "localized_group_embedding_lookup_variable_dim{}_{}".format(dim, j), params + sp_ids
          ) as name_scope:
            outputs = gen_group_embedding_ops.group_variable_lookup(
              (tf_handlers[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              (tf_sp_values[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              (tf_sp_indices[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              sub_tf_sp_weight,
              tf_combiners[group_id],
              (tf_dense_shape[group_id])[j * params_num_per_group : (j + 1) * params_num_per_group],
              dim,
              ignore_weights,
              is_sequence,
            )[0]

            for idx, output in zip(output_index[j * params_num_per_group : (j + 1) * params_num_per_group], outputs):
              emb_vec[idx] = output

        if num_remainder > 0:
          sub_tf_sp_weight = (
            [None for _ in range(num_remainder)] if ignore_weights else (tf_sp_weights[group_id])[-num_remainder:]
          )
          with ops.name_scope(
            name, "localized_group_embedding_lookup_variable_dim{}".format(dim), params + sp_ids
          ) as name_scope:
            outputs = gen_group_embedding_ops.group_variable_lookup(
              (tf_handlers[group_id])[-num_remainder:],
              (tf_sp_values[group_id])[-num_remainder:],
              (tf_sp_indices[group_id])[-num_remainder:],
              sub_tf_sp_weight,
              tf_combiners[group_id],
              (tf_dense_shape[group_id])[-num_remainder:],
              dim,
              ignore_weights,
              is_sequence,
            )[0]

            for idx, output in zip(output_index[-num_remainder:], outputs):
              emb_vec[idx] = output
  elif strategy == DistStrategy.UNKNOWN:
    raise ValueError("Unrecognized strategy, expected collective, given{}".format(strategy))

  return emb_vec


@ops.RegisterGradient("GroupEmbeddingVarLookupDense")
def _GroupGatherDenseGrad(op, *top_grads):
  ev_num = op.get_attr("num_lookups")
  grads = []
  for i in range(ev_num):
    handle = op.inputs[i]
    indice = op.inputs[ev_num + i]
    params_shape = resource_variable_ops.variable_shape(handle)
    grad = top_grads[i]
    grads.append(indexed_slices.IndexedSlices(grad, indice, params_shape))
  return grads + [None for _ in range(ev_num + 1)]


@ops.RegisterGradient("GroupEmbeddingVarLookup")
def _GroupGatherGrad(op, *grads):
  ev_num = op.get_attr("num_lookups")
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  return_grads = []
  params = op.inputs[:ev_num]
  sp_indices = op.inputs[ev_num * 2 : ev_num * 3]
  unique_values = op.outputs[ev_num : 2 * ev_num]
  batch_nums = op.outputs[3 * ev_num : 4 * ev_num]
  with ops.colocate_with(params[0]):
    nnz_grads = gen_group_embedding_ops.group_embedding_variable_lookup_grad(
      grads[:ev_num], params, unique_values, sp_indices, batch_nums, dimension, combiner
    )
  for i in range(ev_num):
    handle = params[i]
    params_shape = resource_variable_ops.variable_shape(handle)
    indice = unique_values[i]
    grad = nnz_grads[i]
    return_grads.append(indexed_slices.IndexedSlices(grad, indice, params_shape))
  return return_grads + [None for _ in range(ev_num * 4 + 1)]
