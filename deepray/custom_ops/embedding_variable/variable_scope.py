import collections as collections_lib
import copy
import functools
import traceback

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.variable_scope import AUTO_REUSE
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

from deepray.utils import logging_util
from . import variables as ev_variables
from .python import kv_variable_ops

logger = logging_util.get_logger()


@tf_export(v1=["VariableScope"])
class VariableScope(object):
  """Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  """

  def __init__(
    self,
    reuse,
    name="",
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    name_scope="",
    dtype=dtypes.float32,
    use_resource=None,
    constraint=None,
  ):
    """Creates a new VariableScope with the given properties."""
    self._name = name
    self._initializer = initializer
    self._regularizer = regularizer
    self._reuse = reuse
    self._caching_device = caching_device
    self._partitioner = partitioner
    self._custom_getter = custom_getter
    self._name_scope = name_scope
    self._dtype = dtype
    self._use_resource = use_resource
    self._constraint = constraint
    if context.executing_eagerly():
      if self._caching_device is not None:
        raise NotImplementedError("Caching devices is not yet supported when eager execution is enabled.")
      self._reuse = AUTO_REUSE
      self._use_resource = True

  @property
  def name(self):
    return self._name

  @property
  def original_name_scope(self):
    return self._name_scope

  @property
  def reuse(self):
    return self._reuse

  @property
  def initializer(self):
    return self._initializer

  @property
  def dtype(self):
    return self._dtype

  @property
  def use_resource(self):
    return self._use_resource

  @property
  def regularizer(self):
    return self._regularizer

  @property
  def caching_device(self):
    return self._caching_device

  @property
  def partitioner(self):
    return self._partitioner

  @property
  def custom_getter(self):
    return self._custom_getter

  @property
  def constraint(self):
    return self._constraint

  def reuse_variables(self):
    """Reuse variables in this scope."""
    self._reuse = True

  def set_initializer(self, initializer):
    """Set initializer for this scope."""
    self._initializer = initializer

  def set_dtype(self, dtype):
    """Set data type for this scope."""
    self._dtype = dtype

  def set_use_resource(self, use_resource):
    """Sets whether to use ResourceVariables for this scope."""
    if context.executing_eagerly() and not use_resource:
      raise ValueError("When eager execution is enabled, use_resource cannot be set to false.")
    self._use_resource = use_resource

  def set_regularizer(self, regularizer):
    """Set regularizer for this scope."""
    self._regularizer = regularizer

  def set_caching_device(self, caching_device):
    """Set caching_device for this scope."""
    if context.executing_eagerly():
      raise NotImplementedError("Caching devices are not yet supported when eager execution is enabled.")
    self._caching_device = caching_device

  def set_partitioner(self, partitioner):
    """Set partitioner for this scope."""
    self._partitioner = partitioner

  def set_custom_getter(self, custom_getter):
    """Set custom getter for this scope."""
    self._custom_getter = custom_getter

  def get_collection(self, name):
    """Get this scope's variables."""
    scope = self._name + "/" if self._name else ""
    return ops.get_collection(name, scope)

  def trainable_variables(self):
    """Get this scope's trainable variables."""
    return self.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

  def global_variables(self):
    """Get this scope's global variables."""
    return self.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def local_variables(self):
    """Get this scope's local variables."""
    return self.get_collection(ops.GraphKeys.LOCAL_VARIABLES)

  def get_variable(
    self,
    var_store,
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
  ):
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if context.executing_eagerly():
      reuse = False
      use_resource = True
    else:
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if dtype is not None and initializer is not None and not callable(initializer):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      return var_store.get_variable(
        full_name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
        use_resource=use_resource,
        custom_getter=custom_getter,
        constraint=constraint,
      )

  def get_embedding_variable(
    self,
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    invalid_key=None,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if not context.executing_eagerly():
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource
    else:
      reuse = AUTO_REUSE
      use_resource = True

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if dtype is not None and initializer is not None and not callable(initializer):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      if invalid_key is None:
        invalid_key = -1
      return _VariableStore().get_variable(
        full_name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
        use_resource=use_resource,
        custom_getter=custom_getter,
        constraint=constraint,
        invalid_key=invalid_key,
        evconfig=evconfig,
        ht_partition_num=ht_partition_num,
      )

  def get_dynamic_dimension_embedding_variable(
    self,
    var_store,
    name,
    shape=None,
    embedding_block_num=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    invalid_key=None,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Gets an existing variable with this name or create a new one."""
    if regularizer is None:
      regularizer = self._regularizer
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if custom_getter is None:
      custom_getter = self._custom_getter
    if not context.executing_eagerly():
      if reuse is None:
        reuse = self._reuse
      if use_resource is None:
        use_resource = self._use_resource
    else:
      reuse = AUTO_REUSE
      use_resource = True

    full_name = self.name + "/" + name if self.name else name
    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # Check that `initializer` dtype and `dtype` are consistent before
      # replacing them with defaults.
      if dtype is not None and initializer is not None and not callable(initializer):
        init_dtype = ops.convert_to_tensor(initializer).dtype.base_dtype
        if init_dtype != dtype:
          raise ValueError("Initializer type '%s' and explicit dtype '%s' don't match." % (init_dtype, dtype))
      if initializer is None:
        initializer = self._initializer
      if constraint is None:
        constraint = self._constraint
      if dtype is None:
        dtype = self._dtype
      if invalid_key is None:
        invalid_key = -1
      return var_store.get_variable(
        full_name,
        shape=shape,
        embedding_block_num=embedding_block_num,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
        use_resource=use_resource,
        custom_getter=custom_getter,
        constraint=constraint,
        invalid_key=invalid_key,
        evconfig=evconfig,
        ht_partition_num=ht_partition_num,
      )

  def _get_partitioned_variable(
    self,
    var_store,
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    constraint=None,
  ):
    """Gets an existing variable with this name or create a new one."""
    if initializer is None:
      initializer = self._initializer
    if regularizer is None:
      regularizer = self._regularizer
    if constraint is None:
      constraint = self._constraint
    if caching_device is None:
      caching_device = self._caching_device
    if partitioner is None:
      partitioner = self._partitioner
    if dtype is None:
      dtype = self._dtype
    if use_resource is None:
      use_resource = self._use_resource

    if self._custom_getter is not None:
      raise ValueError(
        "Private access to _get_partitioned_variable is not allowed when "
        "a custom getter is set.  Current custom getter: %s.  "
        "It is likely that you're using create_partitioned_variables.  "
        "If so, consider instead using get_variable with a non-empty "
        "partitioner parameter instead." % self._custom_getter
      )

    if partitioner is None:
      raise ValueError("No partitioner was specified")

    # This allows the variable scope name to be used as the variable name if
    # this function is invoked with an empty name arg, for backward
    # compatibility with create_partitioned_variables().
    full_name_list = []
    if self.name:
      full_name_list.append(self.name)
    if name:
      full_name_list.append(name)
    full_name = "/".join(full_name_list)

    # Variable names only depend on variable_scope (full_name here),
    # not name_scope, so we reset it below for the time of variable creation.
    with ops.name_scope(None):
      # pylint: disable=protected-access
      return var_store._get_partitioned_variable(
        full_name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=self.reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
        use_resource=use_resource,
        constraint=constraint,
      )
      # pylint: enable=protected-access


class _VariableStore(object):
  """Variable store that carries a number of named Variables.

  New variable names and new variables can be created; all stored
  variables are initialized with the initializer passed to __init__.

  Attributes:
    vars: a dictionary with string names (same as passed in GetVar) as keys and
      the corresponding TensorFlow Variables as values.
  """

  def __init__(self):
    """Create a variable store."""
    self._vars = {}  # A dictionary of the stored TensorFlow variables.
    self._partitioned_vars = {}  # A dict of the stored PartitionedVariables.
    self._store_eager_variables = False

  def get_variable(
    self,
    name,
    shape=None,
    embedding_block_num=None,
    dtype=dtypes.float32,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    invalid_key=None,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Gets an existing variable with these parameters or create a new one.

    If a variable with the given name is already stored, we return the stored
    variable. Otherwise, we create a new one.

    Set `reuse` to `True` when you only want to reuse existing Variables.
    Set `reuse` to `False` when you only want to create new Variables.
    Set `reuse` to None (the default) or tf.compat.v1.AUTO_REUSE when you want
    variables to be created if they don't exist or returned if they do.

    If initializer is `None` (the default), the default initializer passed in
    the constructor is used. If that one is `None` too, we use a new
    `glorot_uniform_initializer`. If initializer is a Tensor, we use
    it as a value and derive the shape from the initializer.

    If a partitioner is provided, a `PartitionedVariable` is returned.
    Accessing this object as a `Tensor` returns the shards concatenated along
    the partition axis.

    Some useful partitioners are available.  See, e.g.,
    `variable_axis_size_partitioner` and `min_max_variable_partitioner`.

    Args:
      name: The name of the new or existing variable.
      shape: Shape of the new or existing variable.
      dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
      initializer: Initializer for the variable.
      regularizer: A (Tensor -> Tensor or None) function; the result of applying
        it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
      reuse: a Boolean, None, or tf.AUTO_REUSE. Controls reuse or creation of
        variables. When eager execution is enabled  this argument is always
        forced to be False.
      trainable: If `True` also add the variable to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`). `trainable`
        defaults to `True`, unless `synchronization` is set to `ON_READ`, in
        which case it defaults to `False`.
      collections: List of graph collections keys to add the `Variable` to.
        Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see `tf.Variable`).
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the `Variable` reside, to
        deduplicate copying through `Switch` and other conditional statements.
      partitioner: Optional callable that accepts a fully defined `TensorShape`
        and dtype of the `Variable` to be created, and returns a list of
        partitions for each axis (currently only one axis can be partitioned).
      validate_shape: If False, allows the variable to be initialized with a
        value of unknown shape. If True, the default, the shape of initial_value
        must be known.
      use_resource: If False, creates a regular Variable. If True, creates
        instead an experimental ResourceVariable which has well-defined
        semantics. Defaults to False (will later change to True). When eager
        execution is enabled this argument is always forced to be true.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. The signature
        of `custom_getter` should match that of this method,
        but the most future-proof version will allow for changes: `def
          custom_getter(getter, *args, **kwargs)`.  Direct access to
        all `get_variable` parameters is also allowed: `def
          custom_getter(getter, name, *args, **kwargs)`.  A simple identity
        custom getter that simply creates variables with modified names is:
          ```python
        def custom_getter(getter, name, *args, **kwargs): return getter(name +
          '_suffix', *args, **kwargs) ```
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.

    Returns:
      The created or existing `Variable` (or `PartitionedVariable`, if a
      partitioner was used).

    Raises:
      ValueError: when creating a new variable and shape is not declared,
        when reusing a variable and specifying a conflicting shape,
        or when violating reuse during variable creation.
      RuntimeError: when eager execution is enabled and not called from an
        EagerVariableStore.
    """
    if custom_getter is not None and not callable(custom_getter):
      raise ValueError("Passed a custom_getter which is not callable: %s" % custom_getter)

    # If a *_ref type is passed in an error would be triggered further down the
    # stack. We prevent this using base_dtype to get a non-ref version of the
    # type, before doing anything else. When _ref types are removed in favor of
    # resources, this line can be removed.
    try:
      dtype = dtype.base_dtype
    except AttributeError:
      # .base_dtype not existing means that we will try and use the raw dtype
      # which was passed in - this might be a NumPy type which is valid.
      pass

    # This is the main logic of get_variable.  However, custom_getter
    # may override this logic.  So we save it as a callable and pass
    # it to custom_getter.
    # Note: the parameters of _true_getter, and their documentation, match
    # *exactly* item-for-item with the docstring of this method.
    def _true_getter(  # pylint: disable=missing-docstring
      name,
      shape=None,
      embedding_block_num=None,
      dtype=dtypes.float32,
      initializer=None,
      regularizer=None,
      reuse=None,
      trainable=None,
      collections=None,
      caching_device=None,
      partitioner=None,
      validate_shape=True,
      use_resource=None,
      constraint=None,
      invalid_key=None,
      evconfig=ev_variables.EmbeddingVariableConfig(),
      ht_partition_num=1000,
    ):
      is_scalar = shape is not None and isinstance(shape, collections_lib.abc.Sequence) and not shape
      # Partitioned variable case
      if partitioner is not None and not is_scalar:
        if not callable(partitioner):
          raise ValueError("Partitioner must be callable, but received: %s" % partitioner)
        with ops.name_scope(None):
          return self._get_partitioned_variable(
            name=name,
            shape=shape,
            embedding_block_num=embedding_block_num,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            reuse=reuse,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=partitioner,
            validate_shape=validate_shape,
            use_resource=use_resource,
            constraint=constraint,
            invalid_key=invalid_key,
            evconfig=evconfig,
            ht_partition_num=ht_partition_num,
          )

      # Special case for partitioned variable to allow reuse without having to
      # specify partitioner.
      if reuse is True and partitioner is None and name in self._partitioned_vars:
        return self._get_partitioned_variable(
          name=name,
          shape=shape,
          embedding_block_num=embedding_block_num,
          dtype=dtype,
          initializer=initializer,
          regularizer=regularizer,
          reuse=reuse,
          trainable=trainable,
          collections=collections,
          caching_device=caching_device,
          partitioner=None,
          validate_shape=validate_shape,
          use_resource=use_resource,
          constraint=constraint,
          invalid_key=invalid_key,
          evconfig=evconfig,
          ht_partition_num=ht_partition_num,
        )

      # Single variable case
      if "%s/part_0" % name in self._vars:
        raise ValueError(
          "No partitioner was provided, but a partitioned version of the "
          "variable was found: %s/part_0. Perhaps a variable of the same "
          "name was already created with partitioning?" % name
        )

      return self._get_single_variable(
        name=name,
        shape=shape,
        embedding_block_num=embedding_block_num,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        validate_shape=validate_shape,
        use_resource=use_resource,
        constraint=constraint,
        invalid_key=invalid_key,
        evconfig=evconfig,
        ht_partition_num=ht_partition_num,
      )

    if custom_getter is not None:
      # Handle backwards compatibility with getter arguments that were added
      # to the API after users started writing custom getters.
      custom_getter_kwargs = {
        "getter": _true_getter,
        "name": name,
        "shape": shape,
        "embedding_block_num": embedding_block_num,
        "dtype": dtype,
        "initializer": initializer,
        "regularizer": regularizer,
        "reuse": reuse,
        "trainable": trainable,
        "collections": collections,
        "caching_device": caching_device,
        "partitioner": partitioner,
        "validate_shape": validate_shape,
        "use_resource": use_resource,
        "invalid_key": invalid_key,
        "evconfig": evconfig,
        "ht_partition_num": ht_partition_num,
      }
      # `fn_args` and `has_kwargs` can handle functions, `functools.partial`,
      # `lambda`.
      if "constraint" in function_utils.fn_args(custom_getter) or function_utils.has_kwargs(custom_getter):
        custom_getter_kwargs["constraint"] = constraint
      return custom_getter(**custom_getter_kwargs)
    else:
      return _true_getter(
        name,
        shape=shape,
        embedding_block_num=embedding_block_num,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        reuse=reuse,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
        use_resource=use_resource,
        constraint=constraint,
        invalid_key=invalid_key,
        evconfig=evconfig,
        ht_partition_num=ht_partition_num,
      )

  def _get_single_variable(
    self,
    name,
    shape=None,
    embedding_block_num=None,
    dtype=dtypes.float32,
    initializer=None,
    regularizer=None,
    partition_info=None,
    reuse=None,
    trainable=None,
    collections=None,
    caching_device=None,
    validate_shape=True,
    use_resource=None,
    constraint=None,
    invalid_key=None,
    evconfig=ev_variables.EmbeddingVariableConfig(),
    ht_partition_num=1000,
  ):
    """Get or create a single Variable (e.g.

    a shard or entire variable).

    See the documentation of get_variable above (ignore partitioning components)
    for details.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.
      initializer: see get_variable.
      regularizer: see get_variable.
      partition_info: _PartitionInfo object.
      reuse: see get_variable.
      trainable: see get_variable.
      collections: see get_variable.
      caching_device: see get_variable.
      validate_shape: see get_variable.
      use_resource: see get_variable.
      constraint: see get_variable.

    Returns:
      A Variable.  See documentation of get_variable above.

    Raises:
      ValueError: See documentation of get_variable above.
    """
    # Set to true if initializer is a constant.
    initializing_from_value = False
    if initializer is not None and not callable(initializer):
      initializing_from_value = True
    if shape is not None and initializing_from_value:
      raise ValueError("If initializer is a constant, do not specify shape.")

    dtype = dtypes.as_dtype(dtype)
    shape = tensor_shape.as_shape(shape)

    if name in self._vars:
      # Here we handle the case when returning an existing variable.
      if reuse is False:
        var = self._vars[name]
        err_msg = (
          "Variable %s already exists, disallowed."
          " Did you mean to set reuse=True or "
          "reuse=tf.AUTO_REUSE in VarScope?" % name
        )
        # ResourceVariables don't have an op associated with so no traceback
        if isinstance(var, resource_variable_ops.ResourceVariable):
          raise ValueError(err_msg)
        tb = var.op.traceback[::-1]
        # Throw away internal tf entries and only take a few lines. In some
        # cases the traceback can be longer (e.g. if someone uses factory
        # functions to create variables) so we take more than needed in the
        # default case.
        tb = [x for x in tb if "tensorflow/python" not in x[0]][:5]
        raise ValueError("%s Originally defined at:\n\n%s" % (err_msg, "".join(traceback.format_list(tb))))
      found_var = self._vars[name]
      from tensorflow.python.ops.hash_table import hash_table

      if isinstance(found_var, (hash_table.HashTable, hash_table.DistributedHashTable)):
        raise ValueError(
          "Trying to reuse variable %s, but an existing variable is a"
          " HashTable or DistributedHashTable, can not reuse it." % (name)
        )
      if not shape.is_compatible_with(found_var.get_shape()):
        raise ValueError(
          "Trying to share variable %s, but specified shape %s"
          " and found shape %s." % (name, shape, found_var.get_shape())
        )
      if not dtype.is_compatible_with(found_var.dtype):
        dtype_str = dtype.name
        found_type_str = found_var.dtype.name
        raise ValueError(
          "Trying to share variable %s, but specified dtype %s and found dtype %s." % (name, dtype_str, found_type_str)
        )
      return found_var

    # Create the tensor to initialize the variable with default value.
    if initializer is None:
      initializer, initializing_from_value = self._get_default_initializer(name=name, shape=shape, dtype=dtype)
    # Enter an init scope when creating the initializer.
    with ops.init_scope():
      if initializing_from_value:
        init_val = initializer
        variable_dtype = None
      else:
        # Instantiate initializer if provided initializer is a type object.
        if tf_inspect.isclass(initializer):
          initializer = initializer()
        if shape is not None and shape.is_fully_defined():
          if use_resource and invalid_key is not None:
            s = [1 if isinstance(initializer, init_ops.Constant) else evconfig.default_value_dim] + shape.as_list()
            evconfig.default_value_dim = 1 if isinstance(initializer, init_ops.Constant) else evconfig.default_value_dim
          else:
            s = shape.as_list()
          init_val = functools.partial(initializer, shape=s, dtype=dtype)
          variable_dtype = dtype.base_dtype
        elif len(tf_inspect.getargspec(initializer).args) == len(tf_inspect.getargspec(initializer).defaults or []):
          init_val = initializer
          variable_dtype = None
        else:
          raise ValueError(
            "The initializer passed is not valid. It should "
            "be a callable with no arguments and the "
            "shape should not be provided or an instance of "
            "`tf.keras.initializers.*' and `shape` should be "
            "fully defined."
          )

    v = default_variable_creator(
      initial_value=init_val,
      name=name,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      embedding_block_num=embedding_block_num,
      dtype=variable_dtype,
      validate_shape=validate_shape,
      constraint=constraint,
      invalid_key=invalid_key,
      evconfig=evconfig,
      initializer=initializer,
      ht_partition_num=ht_partition_num,
    )
    if not context.executing_eagerly() or self._store_eager_variables:
      # In eager mode we do not want to keep default references to Variable
      # objects as this will prevent their memory from being released.
      self._vars[name] = v
    logging.vlog(1, "Created variable %s with shape %s and init %s", v.name, format(shape), initializer)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with ops.colocate_with(v):
        with ops.name_scope(name + "/Regularizer/"):
          with ops.init_scope():
            loss = regularizer(v)
        if loss is not None:
          if context.executing_eagerly():
            v_name = "v_%s" % type(v)
            loss_name = "loss_%s" % type(loss)
          else:
            v_name = v.name
            loss_name = loss.name
          logging.vlog(
            1, "Applied regularizer to %s and added the result %s to REGULARIZATION_LOSSES.", v_name, loss_name
          )
          ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v

  # Initialize variable when no initializer provided
  def _get_default_initializer(self, name, shape=None, dtype=dtypes.float32):
    """Provide a default initializer and a corresponding value.

    Args:
      name: see get_variable.
      shape: see get_variable.
      dtype: see get_variable.

    Returns:
      initializer and initializing_from_value. See get_variable above.

    Raises:
      ValueError: When giving unsupported dtype.
    """
    del shape
    # If dtype is DT_FLOAT, provide a uniform unit scaling initializer
    if dtype.is_floating:
      initializer = init_ops.glorot_uniform_initializer()
      initializing_from_value = False
    # If dtype is DT_INT/DT_UINT, provide a default value `zero`
    # If dtype is DT_BOOL, provide a default value `FALSE`
    elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool or dtype == dtypes.string:
      initializer = init_ops.zeros_initializer()
      initializing_from_value = False
    # NOTES:Do we need to support for handling DT_STRING and DT_COMPLEX here?
    else:
      raise ValueError("An initializer for variable %s of %s is required" % (name, dtype.base_dtype))

    return initializer, initializing_from_value


# @tf_export(v1=["get_embedding_variable"])
def get_embedding_variable_internal(
  name,
  embedding_dim,
  key_dtype=dtypes.int64,
  value_dtype=None,
  initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None,
  constraint=None,
  steps_to_live=None,
  init_data_source=None,
  ev_option=ev_variables.EmbeddingVariableOption(),
):
  if key_dtype == dtypes.int64:
    invalid_key = 9223372036854775807
  elif key_dtype == dtypes.int32:
    invalid_key = -1
  elif key_dtype == dtypes.string:
    invalid_key = ""
  else:
    raise ValueError("Not support key_dtype: %s, only support int64/int32/string" % key_dtype)
  l2_weight_threshold = -1.0
  if initializer is None and ev_option.init.initializer is None:
    initializer = init_ops.truncated_normal_initializer()
  elif ev_option.init.initializer is not None:
    if initializer is not None:
      logger.warning("Use initializer in InitializerOption.")
    initializer = ev_option.init.initializer
  if ev_option.evict is not None:
    if isinstance(ev_option.evict, ev_variables.GlobalStepEvict):
      if steps_to_live is not None:
        logger.warning("Warning: steps_to_live is double set, the steps_to_live in EvcitConfig is valid")
      steps_to_live = ev_option.evict.steps_to_live
    elif isinstance(ev_option.evict, ev_variables.L2WeightEvict):
      l2_weight_threshold = ev_option.evict.l2_weight_threshold
  else:
    l2_weight_threshold = -1.0
  if steps_to_live is not None and l2_weight_threshold > 0:
    raise ValueError("step_to_live and l2_weight_threshold can't be enabled at same time.")
  return VariableScope(reuse=False).get_embedding_variable(
    name,
    shape=embedding_dim,
    dtype=value_dtype,
    initializer=initializer,
    regularizer=regularizer,
    trainable=trainable,
    collections=collections,
    caching_device=caching_device,
    partitioner=partitioner,
    validate_shape=validate_shape,
    use_resource=True,
    custom_getter=custom_getter,
    constraint=constraint,
    invalid_key=invalid_key,
    evconfig=ev_variables.EmbeddingVariableConfig(
      steps_to_live=steps_to_live,
      init_data_source=init_data_source,
      ht_type=ev_option.ht_type,
      l2_weight_threshold=l2_weight_threshold,
      filter_strategy=ev_option.filter_strategy,
      storage_type=ev_option.storage_option.storage_type,
      storage_path=ev_option.storage_option.storage_path,
      storage_size=ev_option.storage_option.storage_size,
      storage_cache_strategy=ev_option.storage_option.cache_strategy,
      layout=ev_option.storage_option.layout,
      default_value_dim=ev_option.init.default_value_dim,
      default_value_no_permission=ev_option.init.default_value_no_permission,
    ),
    ht_partition_num=ev_option.ht_partition_num,
  )


# @tf_export(v1=["get_embedding_variable_v2"])
def get_embedding_variable_v2_internal(
  name,
  embedding_dim,
  key_dtype=dtypes.int64,
  value_dtype=None,
  initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None,
  constraint=None,
  evconfig=ev_variables.EmbeddingVariableConfig(),
  ht_partition_num=1000,
):
  if key_dtype == dtypes.int64:
    invalid_key = 9223372036854775807
  elif key_dtype == dtypes.int32:
    invalid_key = -1
  elif key_dtype == dtypes.string:
    invalid_key = ""
  else:
    raise ValueError("Not support key_dtype: %s, only support int64/int32/string" % key_dtype)
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer()
  return VariableScope(reuse=False).get_embedding_variable(
    name,
    shape=embedding_dim,
    dtype=value_dtype,
    initializer=initializer,
    regularizer=regularizer,
    trainable=trainable,
    collections=collections,
    caching_device=caching_device,
    partitioner=partitioner,
    validate_shape=validate_shape,
    use_resource=True,
    custom_getter=custom_getter,
    constraint=constraint,
    invalid_key=invalid_key,
    evconfig=evconfig,
    ht_partition_num=ht_partition_num,
  )


@tf_export(v1=["get_embedding_variable"])
def get_embedding_variable(
  name,
  embedding_dim,
  key_dtype=dtypes.int64,
  value_dtype=None,
  initializer=None,
  regularizer=None,
  trainable=True,
  collections=None,
  caching_device=None,
  partitioner=None,
  validate_shape=True,
  custom_getter=None,
  constraint=None,
  steps_to_live=None,
  init_data_source=None,
  ev_option=ev_variables.EmbeddingVariableOption(),
):
  if key_dtype == dtypes.int64:
    invalid_key = 9223372036854775807
  elif key_dtype == dtypes.int32:
    invalid_key = -1
  elif key_dtype == dtypes.string:
    invalid_key = ""
  else:
    raise ValueError("Not support key_dtype: %s, only support int64/int32/string" % key_dtype)
  l2_weight_threshold = -1.0
  if initializer is None and ev_option.init.initializer is None:
    initializer = init_ops.truncated_normal_initializer()
  elif ev_option.init.initializer is not None:
    if initializer is not None:
      print("use initializer give in InitializerOption.")
    initializer = ev_option.init.initializer
  if steps_to_live is not None:
    logger.warning("steps_to_live is deprecated, use tf.GlobaStepEvcit(steps_to_live)")
  if ev_option.evict is not None:
    if isinstance(ev_option.evict, ev_variables.GlobalStepEvict):
      if steps_to_live is not None:
        logger.warning("Warning: steps_to_live is double set, the steps_to_live in GlobalStepEvict is valid")
      steps_to_live = ev_option.evict.steps_to_live
    elif isinstance(ev_option.evict, ev_variables.L2WeightEvict):
      l2_weight_threshold = ev_option.evict.l2_weight_threshold
  else:
    l2_weight_threshold = -1.0
  if steps_to_live is not None and l2_weight_threshold > 0:
    raise ValueError("step_to_live and l2_weight_threshold can't be enabled at same time.")
  return VariableScope(reuse=False).get_embedding_variable(
    name,
    shape=embedding_dim,
    dtype=value_dtype,
    initializer=initializer,
    regularizer=regularizer,
    trainable=trainable,
    collections=collections,
    caching_device=caching_device,
    partitioner=partitioner,
    validate_shape=validate_shape,
    use_resource=True,
    custom_getter=custom_getter,
    constraint=constraint,
    invalid_key=invalid_key,
    evconfig=ev_variables.EmbeddingVariableConfig(
      steps_to_live=steps_to_live,
      init_data_source=init_data_source,
      ht_type=ev_option.ht_type,
      l2_weight_threshold=l2_weight_threshold,
      filter_strategy=ev_option.filter_strategy,
      storage_type=ev_option.storage_option.storage_type,
      storage_path=ev_option.storage_option.storage_path,
      storage_size=ev_option.storage_option.storage_size,
      storage_cache_strategy=ev_option.storage_option.cache_strategy,
      layout=ev_option.storage_option.layout,
      default_value_dim=ev_option.init.default_value_dim,
      default_value_no_permission=ev_option.init.default_value_no_permission,
    ),
    ht_partition_num=ev_option.ht_partition_num,
  )


def default_variable_creator(
  initial_value=None,
  trainable=None,
  collections=None,
  validate_shape=True,
  caching_device=None,
  name=None,
  variable_def=None,
  dtype=None,
  embedding_block_num=None,
  import_scope=None,
  constraint=None,
  invalid_key=None,
  evconfig=ev_variables.EmbeddingVariableConfig(),
  initializer=None,
  ht_partition_num=1000,
):
  if invalid_key is not None:
    emb_blocknum = embedding_block_num
    if emb_blocknum is None:
      ev = kv_variable_ops.EmbeddingVariable(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name,
        dtype=dtype,
        constraint=constraint,
        variable_def=variable_def,
        import_scope=import_scope,
        invalid_key=invalid_key,
        evconfig=evconfig,
        # initializer=initializer,
        ht_partition_num=ht_partition_num,
      )
      if evconfig.init_data_source is not None:
        ev.set_init_data_source_initializer(evconfig.init_data_source)
      return ev
    else:
      evconfig.block_num = emb_blocknum
      evlist = []
      block_evconfig = copy.copy(evconfig)
      block_evconfig.handle_name = name
      block_evconfig.emb_index = 0
      primary_ev = kv_variable_ops.EmbeddingVariable(
        initial_value=initial_value,
        trainable=trainable,
        collections=collections,
        validate_shape=validate_shape,
        caching_device=caching_device,
        name=name + "/block0",
        dtype=dtype,
        constraint=constraint,
        variable_def=variable_def,
        import_scope=import_scope,
        invalid_key=invalid_key,
        evconfig=block_evconfig,
        initializer=initializer,
        ht_partition_num=ht_partition_num,
      )
      if evconfig.init_data_source is not None:
        primary_ev.set_init_data_source_initializer(evconfig.init_data_source)
      evlist.append(primary_ev)
      block_evconfig.primary = primary_ev
      with ops.colocate_with(primary_ev):
        block_evconfig.handle_name = primary_ev._block_handle_name
        for i in range(emb_blocknum - 1):
          slave_evconfig = copy.copy(block_evconfig)
          slave_evconfig.emb_index = i + 1
          slave_evconfig._slot_num = primary_ev._slot_num
          slave_ev = kv_variable_ops.EmbeddingVariable(
            initial_value=initial_value,
            trainable=trainable,
            collections=collections,
            validate_shape=validate_shape,
            caching_device=caching_device,
            name=name + "/block" + str(i + 1),
            dtype=dtype,
            constraint=constraint,
            variable_def=variable_def,
            import_scope=import_scope,
            invalid_key=invalid_key,
            evconfig=slave_evconfig,
            initializer=initializer,
            ht_partition_num=ht_partition_num,
          )
          if evconfig.init_data_source is not None:
            slave_ev._set_init_data_source_initializer(evconfig.init_data_source)
          evlist.append(slave_ev)
        dyn_ev = kv_variable_ops.DynamicEmbeddingVariable(name, evlist)
        return dyn_ev
