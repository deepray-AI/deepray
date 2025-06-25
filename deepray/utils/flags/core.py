# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Public interface for flag definition.

See _example.py for detailed instructions on defining flags.
"""

import sys

from six.moves import shlex_quote

from absl import app as absl_app
from absl import flags

from deepray.utils.flags import _base
from deepray.utils.flags import _benchmark
from deepray.utils.flags import _conventions
from deepray.utils.flags import _data
from deepray.utils.flags import _device
from deepray.utils.flags import _distribution
from deepray.utils.flags import _misc
from deepray.utils.flags import _performance


def set_defaults(**kwargs):
  for key, value in kwargs.items():
    flags.FLAGS.set_default(name=key, value=value)


def parse_flags(argv=None):
  """Reset flags and reparse. Currently only used in testing."""
  flags.FLAGS.unparse_flags()
  absl_app.parse_flags_with_usage(argv or sys.argv)


def register_key_flags_in_core(f):
  """Defines a function in core.py, and registers its key flags.

  absl uses the location of a flags.declare_key_flag() to determine the context
  in which a flag is key. By making all declares in core, this allows model
  main functions to call flags.adopt_module_key_flags() on core and correctly
  chain key flags.

  Args:
    f:  The function to be wrapped

  Returns:
    The "core-defined" version of the input function.
  """

  def core_fn(*args, **kwargs):
    key_flags = f(*args, **kwargs)
    [flags.declare_key_flag(fl) for fl in key_flags]  # pylint: disable=expression-not-assigned

  return core_fn


define_base = register_key_flags_in_core(_base.define_base)
define_data = register_key_flags_in_core(_data.define_data_download_flags)
# We have define_base_eager for compatibility, since it used to be a separate
# function from define_base.
define_base_eager = define_base
define_log_steps = register_key_flags_in_core(_benchmark.define_log_steps)
define_benchmark = register_key_flags_in_core(_benchmark.define_benchmark)
define_device = register_key_flags_in_core(_device.define_device)
define_image = register_key_flags_in_core(_misc.define_image)
define_performance = register_key_flags_in_core(_performance.define_performance)
define_distribution = register_key_flags_in_core(_distribution.define_distribution)

help_wrap = _conventions.help_wrap

get_num_gpus = _base.get_num_gpus
get_tf_dtype = _performance.get_tf_dtype
get_loss_scale = _performance.get_loss_scale
DTYPE_MAP = _performance.DTYPE_MAP
require_cloud_storage = _device.require_cloud_storage


def _get_nondefault_flags_as_dict():
  """Returns the nondefault flags as a dict from flag name to value."""
  nondefault_flags = {}
  for flag_name in flags.FLAGS:
    flag_value = getattr(flags.FLAGS, flag_name)
    if (flag_name != flags.FLAGS[flag_name].short_name and flag_value != flags.FLAGS[flag_name].default):
      nondefault_flags[flag_name] = flag_value
  return nondefault_flags


def get_nondefault_flags_as_str():
  """Returns flags as a string that can be passed as command line arguments.

  E.g., returns: "--batch_size=256 --use_synthetic_data" for the following code
  block:

  ```
  flags.FLAGS.batch_size = 256
  flags.FLAGS.use_synthetic_data = True
  print(get_nondefault_flags_as_str())
  ```

  Only flags with nondefault values are returned, as passing default flags as
  command line arguments has no effect.

  Returns:
    A string with the flags, that can be passed as command line arguments to a
    program to use the flags.
  """
  nondefault_flags = _get_nondefault_flags_as_dict()
  flag_strings = []
  for name, value in sorted(nondefault_flags.items()):
    if isinstance(value, bool):
      flag_str = '--{}'.format(name) if value else '--no{}'.format(name)
    elif isinstance(value, list):
      flag_str = '--{}={}'.format(name, ','.join(value))
    else:
      flag_str = '--{}={}'.format(name, value)
    flag_strings.append(flag_str)
  return ' '.join(shlex_quote(flag_str) for flag_str in flag_strings)


def parse_flags(flags_obj):
  """Convenience function to turn flags into params."""
  num_gpus = get_num_gpus(flags_obj)

  batch_size = flags_obj.batch_size
  eval_batch_size = flags_obj.eval_batch_size or flags_obj.batch_size

  return {
      "epochs": flags_obj.epochs,
      "batches_per_step": 1,
      "use_seed": flags_obj.random_seed is not None,
      "batch_size": batch_size,
      "eval_batch_size": eval_batch_size,
      "learning_rate": flags_obj.learning_rate,
      "mf_dim": flags_obj.num_factors,
      "model_layers": [int(layer) for layer in flags_obj.layers],
      "mf_regularization": flags_obj.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in flags_obj.mlp_regularization],
      "num_neg": flags_obj.num_neg,
      "distribution_strategy": flags_obj.distribution_strategy,
      "num_gpus": num_gpus,
      "use_tpu": flags_obj.tpu is not None,
      "tpu": flags_obj.tpu,
      "tpu_zone": flags_obj.tpu_zone,
      "tpu_gcp_project": flags_obj.tpu_gcp_project,
      "beta1": flags_obj.beta1,
      "beta2": flags_obj.beta2,
      "epsilon": flags_obj.epsilon,
      "match_mlperf": flags_obj.ml_perf,
      # "epochs_between_evals": flags_obj.epochs_between_evals,
      "use_custom_training_loop": flags_obj.use_custom_training_loop,
      "hr_threshold": flags_obj.hr_threshold,
      "stream_files": flags_obj.tpu is not None,
      "train_dataset_path": flags_obj.train_dataset_path,
      "eval_dataset_path": flags_obj.eval_dataset_path,
  }
