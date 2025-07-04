import os
import platform
import warnings
import traceback

try:
  _DEEPRAY_PY_OPS = bool(int(os.environ["DEEPRAY_PY_OPS"]))
except KeyError:
  if platform.system() == "Linux":
    _DEEPRAY_PY_OPS = False
  else:
    _DEEPRAY_PY_OPS = True

_FALLBACK_WARNING_TEMPLATE = """{}

The {} C++/CUDA custom op could not be loaded.
For this reason, Deepray will fallback to an implementation written
in Python with public TensorFlow ops. There worst you might experience with
this is a moderate slowdown on GPU. There can be multiple
reason for this loading error, one of them may be an ABI incompatibility between
the TensorFlow installed on your system and the TensorFlow used to compile
Deepray' custom ops. The stacktrace generated when loading the
shared object file was displayed above.

If you want this warning to disappear, either make sure the TensorFlow installed
is compatible with this version of Deepray, or tell Deepray to
prefer using Python implementations and not custom C++/CUDA ones. You can do that
by setting the enviornment variable `DEEPRAY_PY_OPS=1`:
```bash
DEEPRAY_PY_OPS=1 python my_script.py
```
or run `dp.options.disable_custom_kernel()` in your code, after your imports:
```python
import deepray as dp
import ...
import ...

dp.options.disable_custom_kernel()
```
"""


def warn_fallback(op_name):
  warning_msg = _FALLBACK_WARNING_TEMPLATE.format(traceback.format_exc(), op_name)
  warnings.warn(warning_msg, RuntimeWarning)
  disable_custom_kernel()


def enable_custom_kernel():
  """Prefer custom C++/CUDA kernel to pure python operations.

  Enable using custom C++/CUDA kernel instead of pure python operations.
  It has the same effect as setting environment variable `DEEPRAY_PY_OPS=0`.
  """
  global _DEEPRAY_PY_OPS
  _DEEPRAY_PY_OPS = False


def disable_custom_kernel():
  """Prefer pure python operations to custom C++/CUDA kernel.

  Disable using custom C++/CUDA kernel instead of pure python operations.
  It has the same effect as setting environment variable `DEEPRAY_PY_OPS=1`.
  """
  global _DEEPRAY_PY_OPS
  _DEEPRAY_PY_OPS = True


def is_custom_kernel_disabled():
  """Return whether custom C++/CUDA kernel is disabled."""
  return _DEEPRAY_PY_OPS
