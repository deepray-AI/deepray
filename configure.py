# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""configure script to get build parameters from user."""

import argparse
import errno
import glob
import logging
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
from typing import Optional

import tensorflow as tf
from packaging.version import Version

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top

_DEFAULT_CUDA_VERSION = '11'
_DEFAULT_CUDNN_VERSION = '2'
_DEFAULT_TENSORRT_VERSION = '6'

_DEFAULT_PROMPT_ASK_ATTEMPTS = 10

_DP_BAZELRC_FILENAME = '.dp_configure.bazelrc'
_DP_WORKSPACE_ROOT = ''
_DP_BAZELRC = ''
_DP_CURRENT_BAZEL_VERSION = None


class UserInputError(Exception):
  pass


def is_windows():
  return platform.system() == 'Windows'


def is_linux():
  return platform.system() == 'Linux'


def is_raspi_arm():
  return os.uname()[4] == "armv7l" or os.uname()[4] == "aarch64"


def is_macos():
  return platform.system() == 'Darwin'


def is_ppc64le():
  return platform.machine() == 'ppc64le'


def is_s390x():
  return platform.machine() == 's390x'


def is_cygwin():
  return platform.system().startswith('CYGWIN_NT')


def get_tf_header_dir():
  import tensorflow as tf

  tf_header_dir = tf.sysconfig.get_compile_flags()[0][2:]
  if is_windows():
    tf_header_dir = tf_header_dir.replace("\\", "/")
  return tf_header_dir


def get_cpp_version():
  cpp_version = "c++14"
  if Version(tf.__version__) >= Version("2.10"):
    cpp_version = "c++17"
  return cpp_version


def get_tf_shared_lib_dir():
  import tensorflow as tf

  # OS Specific parsing
  if is_windows():
    tf_shared_lib_dir = tf.sysconfig.get_compile_flags()[0][2:-7] + "python"
    return tf_shared_lib_dir.replace("\\", "/")
  elif is_raspi_arm():
    return tf.sysconfig.get_compile_flags()[0][2:-7] + "python"
  else:
    return tf.sysconfig.get_link_flags()[0][2:]


# Converts the linkflag namespec to the full shared library name
def get_shared_lib_name():
  import tensorflow as tf

  namespec = tf.sysconfig.get_link_flags()
  if is_macos():
    # MacOS
    return "lib" + namespec[1][2:] + ".dylib"
  elif is_windows():
    # Windows
    return "_pywrap_tensorflow_internal.lib"
  elif is_raspi_arm():
    # The below command for linux would return an empty list
    return "_pywrap_tensorflow_internal.so"
  else:
    # Linux
    return namespec[1][3:]


def get_tf_version_integer():
  """
  Get Tensorflow version as a 4 digits string.

  For example:
    1.15.2 get 1152
    2.4.1 get 2041
    2.6.3 get 2063
    2.8.3 get 2083

  The 4-digits-string will be passed to C macro to discriminate different
  Tensorflow versions. 

  We assume that major version has 1 digit, minor version has 2 digits. And
  patch version has 1 digit.
  """
  try:
    version = tf.__version__
  except AttributeError:
    raise ImportError(
        '\nPlease install a TensorFlow on your compiling machine, '
        'The compiler needs to know the version of Tensorflow '
        'and get TF c++ headers according to the installed TensorFlow. '
        '\nNote: Only TensorFlow 2.8.3, 2.6.3, 2.4.1, 1.15.2 are supported.'
    )
  try:
    major, minor, patch = version.split('.')
    assert len(major) == 1, "Tensorflow major version must be length of 1. Version: {}".format(version)
    assert len(minor) <= 2, "Tensorflow minor version must be less or equal to 2. Version: {}".format(version)
    assert len(patch) == 1, "Tensorflow patch version must be length of 1. Version: {}".format(version)
  except:
    raise ValueError('got wrong tf.__version__: {}'.format(version))
  tf_version_num = str(int(major) * 1000 + int(minor) * 10 + int(patch))
  if len(tf_version_num) != 4:
    raise ValueError(
        'Tensorflow version flag must be length of 4 (major'
        ' version: 1, minor version: 2, patch_version: 1). But'
        ' get: {}'.format(tf_version_num)
    )
  return int(tf_version_num)


def get_input(question):
  try:
    try:
      answer = raw_input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def symlink_force(target, link_name):
  """Force symlink, equivalent of 'ln -sf'.

  Args:
    target: items to link to.
    link_name: name of the link.
  """
  try:
    os.symlink(target, link_name)
  except OSError as e:
    if e.errno == errno.EEXIST:
      os.remove(link_name)
      os.symlink(target, link_name)
    else:
      raise e


def write_to_bazelrc(line):
  with open(_DP_BAZELRC, 'a') as f:
    f.write(line + '\n')


def write_action_env(var_name, var):
  write_to_bazelrc('build --action_env {}="{}"'.format(var_name, str(var)))


def write_repo_env(var_name, var):
  write_to_bazelrc('build --repo_env {}="{}"'.format(var_name, str(var)))


def run_shell(cmd, allow_non_zero=False, stderr=None):
  if stderr is None:
    stderr = sys.stdout
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd, stderr=stderr)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd, stderr=stderr)
  return output.decode('UTF-8').strip()


def cygpath(path):
  """Convert path from posix to windows."""
  return os.path.abspath(path).replace('\\', '/')


def get_python_path(environ_cp, python_bin_path):
  """Get the python site package paths."""
  python_paths = []
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
  try:
    stderr = open(os.devnull, 'wb')
    library_paths = run_shell(
        [python_bin_path, '-c', 'import site; print("\\n".join(site.getsitepackages()))'], stderr=stderr
    ).split('\n')
  except subprocess.CalledProcessError:
    library_paths = [
        run_shell([python_bin_path, '-c', 'from distutils.sysconfig import get_python_lib;'
                   'print(get_python_lib())'])
    ]

  all_paths = set(python_paths + library_paths)
  # Sort set so order is deterministic
  all_paths = sorted(all_paths)

  paths = []
  for path in all_paths:
    if os.path.isdir(path):
      paths.append(path)
  return paths


def get_python_major_version(python_bin_path):
  """Get the python major version."""
  return run_shell([python_bin_path, '-c', 'import sys; print(sys.version[0])'])


def setup_python(environ_cp):
  """Setup python related env variables."""
  # Get PYTHON_BIN_PATH, default is the current running python.
  default_python_bin_path = sys.executable
  ask_python_bin_path = ('Please specify the location of python. [Default is '
                         '{}]: ').format(default_python_bin_path)
  while True:
    python_bin_path = get_from_env_or_user_or_default(
        environ_cp, 'PYTHON_BIN_PATH', ask_python_bin_path, default_python_bin_path
    )
    # Check if the path is valid
    if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
      break
    elif not os.path.exists(python_bin_path):
      print('Invalid python path: {} cannot be found.'.format(python_bin_path))
    else:
      print('{} is not executable.  Is it the python binary?'.format(python_bin_path))
    environ_cp['PYTHON_BIN_PATH'] = ''

  # Convert python path to Windows style before checking lib and version
  if is_windows() or is_cygwin():
    python_bin_path = cygpath(python_bin_path)

  # Get PYTHON_LIB_PATH
  python_lib_path = environ_cp.get('PYTHON_LIB_PATH')
  if not python_lib_path:
    python_lib_paths = get_python_path(environ_cp, python_bin_path)
    if environ_cp.get('USE_DEFAULT_PYTHON_LIB_PATH') == '1':
      python_lib_path = python_lib_paths[0]
    else:
      print('Found possible Python library paths:\n  %s' % '\n  '.join(python_lib_paths))
      default_python_lib_path = python_lib_paths[0]
      python_lib_path = get_input(
          'Please input the desired Python library path to use.  '
          'Default is [{}]\n'.format(python_lib_paths[0])
      )
      if not python_lib_path:
        python_lib_path = default_python_lib_path
    environ_cp['PYTHON_LIB_PATH'] = python_lib_path

  python_major_version = get_python_major_version(python_bin_path)
  if python_major_version == '2':
    write_to_bazelrc('build --host_force_python=PY2')
  logging.debug(f"Hermetic Python version: {sys.version_info.major}.{sys.version_info.minor}")
  write_repo_env("HERMETIC_PYTHON_VERSION", f"{sys.version_info.major}.{sys.version_info.minor}")

  # Convert python path to Windows style before writing into bazel.rc
  if is_windows() or is_cygwin():
    python_lib_path = cygpath(python_lib_path)

  # Set-up env variables used by python_configure.bzl
  write_action_env('PYTHON_BIN_PATH', python_bin_path)
  write_action_env('PYTHON_LIB_PATH', python_lib_path)
  write_to_bazelrc('build --python_path=\"{}"'.format(python_bin_path))
  environ_cp['PYTHON_BIN_PATH'] = python_bin_path

  # If choosen python_lib_path is from a path specified in the PYTHONPATH
  # variable, need to tell bazel to include PYTHONPATH
  if environ_cp.get('PYTHONPATH'):
    python_paths = environ_cp.get('PYTHONPATH').split(':')
    if python_lib_path in python_paths:
      write_action_env('PYTHONPATH', environ_cp.get('PYTHONPATH'))

  # Write tools/python_bin_path.sh
  with open(os.path.join(_DP_WORKSPACE_ROOT, 'tools', 'python_bin_path.sh'), 'w') as f:
    f.write('export PYTHON_BIN_PATH="{}"'.format(python_bin_path))


def reset_tf_configure_bazelrc():
  """Reset file that contains customized config settings."""
  open(_DP_BAZELRC, 'w').close()


def cleanup_makefile():
  """Delete any leftover BUILD files from the Makefile build.

  These files could interfere with Bazel parsing.
  """
  makefile_download_dir = os.path.join(_DP_WORKSPACE_ROOT, 'tensorflow', 'contrib', 'makefile', 'downloads')
  if os.path.isdir(makefile_download_dir):
    for root, _, filenames in os.walk(makefile_download_dir):
      for f in filenames:
        if f.endswith('BUILD'):
          os.remove(os.path.join(root, f))


def get_var(environ_cp, var_name, query_item, enabled_by_default, question=None, yes_reply=None, no_reply=None):
  """Get boolean input from user.

  If var_name is not set in env, ask user to enable query_item or not. If the
  response is empty, use the default.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.

  Returns:
    boolean value of the variable.

  Raises:
    UserInputError: if an environment variable is set, but it cannot be
      interpreted as a boolean indicator, assume that the user has made a
      scripting error, and will continue to provide invalid input.
      Raise the error to avoid infinitely looping.
  """
  if not question:
    question = 'Do you wish to build Deepray with {} support?'.format(query_item)
  if not yes_reply:
    yes_reply = '{} support will be enabled for Deepray.'.format(query_item)
  if not no_reply:
    no_reply = 'No {}'.format(yes_reply)

  yes_reply += '\n'
  no_reply += '\n'

  if enabled_by_default:
    question += ' [Y/n]: '
  else:
    question += ' [y/N]: '

  var = environ_cp.get(var_name)
  if var is not None:
    var_content = var.strip().lower()
    true_strings = ('1', 't', 'true', 'y', 'yes')
    false_strings = ('0', 'f', 'false', 'n', 'no')
    if var_content in true_strings:
      var = True
    elif var_content in false_strings:
      var = False
    else:
      raise UserInputError(
          'Environment variable %s must be set as a boolean indicator.\n'
          'The following are accepted as TRUE : %s.\n'
          'The following are accepted as FALSE: %s.\n'
          'Current value is %s.' % (var_name, ', '.join(true_strings), ', '.join(false_strings), var)
      )

  while var is None:
    user_input_origin = get_input(question)
    user_input = user_input_origin.strip().lower()
    if user_input == 'y':
      print(yes_reply)
      var = True
    elif user_input == 'n':
      print(no_reply)
      var = False
    elif not user_input:
      if enabled_by_default:
        print(yes_reply)
        var = True
      else:
        print(no_reply)
        var = False
    else:
      print('Invalid selection: {}'.format(user_input_origin))
  return var


def set_action_env_var(
    environ_cp,
    var_name,
    query_item,
    enabled_by_default,
    question=None,
    yes_reply=None,
    no_reply=None,
    bazel_config_name=None
):
  """Set boolean action_env variable.

  Ask user if query_item will be enabled. Default is used if no input is given.
  Set environment variable and write to .bazelrc.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    query_item: string for feature related to the variable, e.g. "CUDA for
      Nvidia GPUs".
    enabled_by_default: boolean for default behavior.
    question: optional string for how to ask for user input.
    yes_reply: optional string for reply when feature is enabled.
    no_reply: optional string for reply when feature is disabled.
    bazel_config_name: adding config to .bazelrc instead of action_env.
  """
  var = int(get_var(environ_cp, var_name, query_item, enabled_by_default, question, yes_reply, no_reply))

  if not bazel_config_name:
    write_action_env(var_name, var)
  elif var:
    write_to_bazelrc('build --config=%s' % bazel_config_name)
  environ_cp[var_name] = str(var)


def convert_version_to_int(version):
  """Convert a version number to a integer that can be used to compare.

  Version strings of the form X.YZ and X.Y.Z-xxxxx are supported. The
  'xxxxx' part, for instance 'homebrew' on OS/X, is ignored.

  Args:
    version: a version to be converted

  Returns:
    An integer if converted successfully, otherwise return None.
  """
  version = version.split('-')[0]
  version_segments = version.split('.')
  # Treat "0.24" as "0.24.0"
  if len(version_segments) == 2:
    version_segments.append('0')
  for seg in version_segments:
    if not seg.isdigit():
      return None

  version_str = ''.join(['%03d' % int(seg) for seg in version_segments])
  return int(version_str)


def retrieve_bazel_version():
  """Retrieve installed bazel version (or bazelisk).

  Returns:
    The bazel version detected.
  """
  bazel_executable = which('bazel')
  if bazel_executable is None:
    bazel_executable = which('bazelisk')
    if bazel_executable is None:
      print('Cannot find bazel. Please install bazel/bazelisk.')
      sys.exit(1)

  stderr = open(os.devnull, 'wb')
  curr_version = run_shell([bazel_executable, '--version'], allow_non_zero=True, stderr=stderr)
  if curr_version.startswith('bazel '):
    curr_version = curr_version.split('bazel ')[1]

  curr_version_int = convert_version_to_int(curr_version)

  # Check if current bazel version can be detected properly.
  if not curr_version_int:
    print('WARNING: current bazel installation is not a release version.')
    return curr_version

  print('You have bazel %s installed.' % curr_version)
  return curr_version


def set_cc_opt_flags(environ_cp):
  """Set up architecture-dependent optimization flags.

  Also append CC optimization flags to bazel.rc..

  Args:
    environ_cp: copy of the os.environ.
  """
  if is_ppc64le():
    # gcc on ppc64le does not support -march, use mcpu instead
    default_cc_opt_flags = '-mcpu=native'
  elif is_windows():
    default_cc_opt_flags = '/arch:AVX'
  else:
    # On all other platforms, no longer use `-march=native` as this can result
    # in instructions that are too modern being generated. Users that want
    # maximum performance should compile TF in their environment and can pass
    # `-march=native` there.
    # See https://github.com/tensorflow/tensorflow/issues/45744 and duplicates
    default_cc_opt_flags = '-Wno-sign-compare'
  question = (
      'Please specify optimization flags to use during compilation when'
      ' bazel option "--config=opt" is specified [Default is %s]: '
  ) % default_cc_opt_flags
  cc_opt_flags = get_from_env_or_user_or_default(environ_cp, 'CC_OPT_FLAGS', question, default_cc_opt_flags)
  for opt in cc_opt_flags.split():
    write_to_bazelrc('build:opt --copt=%s' % opt)
    write_to_bazelrc('build:opt --host_copt=%s' % opt)


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var, var_default):
  """Get var_name either from env, or user or default.

  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.

  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.

  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(
    environ_cp,
    var_name,
    var_default,
    ask_for_var,
    check_success,
    error_msg,
    suppress_default_error=False,
    resolve_symlinks=False,
    n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS
):
  """Loop over user prompts for an ENV param until receiving a valid response.

  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.

  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) String with one and only one '%s'. Formatted with each
      invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    resolve_symlinks: (Bool) Translate symbolic links into the real filepath.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.

  Returns:
    [String] The value of var_name after querying for input.

  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query, default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg % val)
    environ_cp[var_name] = ''
  else:
    raise UserInputError(
        'Invalid %s setting was provided %d times in a row. '
        'Assuming to be a scripting mistake.' % (var_name, n_ask_attempts)
    )

  if resolve_symlinks:
    val = os.path.realpath(val)
  environ_cp[var_name] = val
  return val


def choose_compiler(environ_cp):
  question = 'Do you want to use Clang as the compiler?'
  yes_reply = 'Clang will be used to compile Deepray.'
  no_reply = 'GCC will be used to compile Deepray.'
  var = int(get_var(environ_cp, 'TF_NEED_CLANG', None, False, question, yes_reply, no_reply))
  return var


def set_gcc_host_compiler_path(environ_cp):
  """Set GCC_HOST_COMPILER_PATH."""
  default_gcc_host_compiler_path = which('gcc') or ''
  cuda_bin_symlink = '%s/bin/gcc' % environ_cp.get('CUDA_TOOLKIT_PATH')

  if os.path.islink(cuda_bin_symlink):
    # os.readlink is only available in linux
    default_gcc_host_compiler_path = os.path.realpath(cuda_bin_symlink)

  gcc_host_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='GCC_HOST_COMPILER_PATH',
      var_default=default_gcc_host_compiler_path,
      ask_for_var='Please specify which gcc should be used by nvcc as the host '
      'compiler.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg='Invalid gcc path. %s cannot be found.',
  )
  write_repo_env("CC", gcc_host_compiler_path)
  write_repo_env("BAZEL_COMPILER", gcc_host_compiler_path)
  return gcc_host_compiler_path


def get_gcc_major_version(gcc_path: str):
  gcc_version_proc = subprocess.run(
      [gcc_path, "-dumpversion"],
      check=True,
      capture_output=True,
      text=True,
  )
  major_version = int(gcc_version_proc.stdout)
  return major_version


def set_clang_compiler_path(environ_cp):
  """Set CLANG_COMPILER_PATH and environment variables.

  Loop over user prompts for clang path until receiving a valid response.
  Default is used if no input is given. Set CLANG_COMPILER_PATH and write
  environment variables CC and BAZEL_COMPILER to .bazelrc.

  Args:
    environ_cp: (Dict) copy of the os.environ.

  Returns:
    string value for clang_compiler_path.
  """
  # Default path if clang-18 is installed by using apt-get install
  # remove 16 logic upon release of 19
  default_clang_path = '/usr/lib/llvm-18/bin/clang'
  if not os.path.exists(default_clang_path):
    default_clang_path = '/usr/lib/llvm-17/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = '/usr/lib/llvm-16/bin/clang'
    if not os.path.exists(default_clang_path):
      default_clang_path = which('clang') or ''

  clang_compiler_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='CLANG_COMPILER_PATH',
      var_default=default_clang_path,
      ask_for_var='Please specify the path to clang executable.',
      check_success=os.path.exists,
      resolve_symlinks=True,
      error_msg=(
          'Invalid clang path. %s cannot be found. Note that TensorFlow now'
          ' requires clang to compile. You may override this behavior by'
          ' setting TF_NEED_CLANG=0'
      ),
  )

  write_repo_env('CC', clang_compiler_path)
  write_repo_env('BAZEL_COMPILER', clang_compiler_path)

  return clang_compiler_path


def retrieve_clang_version(clang_executable):
  """Retrieve installed clang version.

  Args:
    clang_executable: (String) path to clang executable

  Returns:
    The clang version detected.
  """
  stderr = open(os.devnull, 'wb')
  curr_version = run_shell([clang_executable, '--version'], allow_non_zero=True, stderr=stderr)

  curr_version_split = curr_version.lower().split('clang version ')
  if len(curr_version_split) > 1:
    curr_version = curr_version_split[1].split()[0]

  curr_version_int = convert_version_to_int(curr_version)
  # Check if current clang version can be detected properly.
  if not curr_version_int:
    print('WARNING: current clang installation is not a release version.\n')
    return None

  print('You have Clang %s installed.\n' % curr_version)
  return curr_version


# Disable clang extension that rejects type definitions within offsetof.
# This was added in clang-16 by https://reviews.llvm.org/D133574.
# Still required for clang-17.
# Can be removed once upb is updated, since a type definition is used within
# offset of in the current version of ubp. See
# https://github.com/protocolbuffers/upb/blob/9effcbcb27f0a665f9f345030188c0b291e32482/upb/upb.c#L183.
def disable_clang_offsetof_extension(clang_version):
  clang_major_version = int(clang_version.split('.')[0])
  if clang_major_version in (16, 17):
    write_to_bazelrc('build --copt=-Wno-gnu-offsetof-extensions')
  if clang_major_version >= 16:
    # Enable clang settings that are needed for the build to work with newer
    # versions of Clang.
    write_to_bazelrc("build --config=clang")
  if clang_major_version < 19:
    # Prevent XNNPACK from using `-mavxvnniint8` (only available in clang 16+/gcc 13+).
    write_to_bazelrc("build --define=xnn_enable_avxvnniint8=false")


def set_tf_cuda_paths(environ_cp):
  """Set TF_CUDA_PATHS."""
  ask_cuda_paths = (
      'Please specify the comma-separated list of base paths to look for CUDA '
      'libraries and headers. [Leave empty to use the default]: '
  )
  tf_cuda_paths = get_from_env_or_user_or_default(environ_cp, 'TF_CUDA_PATHS', ask_cuda_paths, '')
  if tf_cuda_paths:
    environ_cp['TF_CUDA_PATHS'] = tf_cuda_paths


def set_tf_cuda_version(environ_cp):
  """Set TF_CUDA_VERSION."""
  ask_cuda_version = (
      'Please specify the CUDA SDK version you want to use. '
      '[Leave empty to default to CUDA %s]: '
  ) % _DEFAULT_CUDA_VERSION
  tf_cuda_version = get_from_env_or_user_or_default(
      environ_cp, 'TF_CUDA_VERSION', ask_cuda_version, _DEFAULT_CUDA_VERSION
  )
  environ_cp['TF_CUDA_VERSION'] = tf_cuda_version


def set_tf_cudnn_version(environ_cp):
  """Set TF_CUDNN_VERSION."""
  ask_cudnn_version = (
      'Please specify the cuDNN version you want to use. '
      '[Leave empty to default to cuDNN %s]: '
  ) % _DEFAULT_CUDNN_VERSION
  tf_cudnn_version = get_from_env_or_user_or_default(
      environ_cp, 'TF_CUDNN_VERSION', ask_cudnn_version, _DEFAULT_CUDNN_VERSION
  )
  environ_cp['TF_CUDNN_VERSION'] = tf_cudnn_version


def set_tf_tensorrt_version(environ_cp):
  """Set TF_TENSORRT_VERSION."""
  if not (is_linux() or is_windows()):
    raise ValueError('Currently TensorRT is only supported on Linux platform.')

  if not int(environ_cp.get('TF_NEED_TENSORRT', False)):
    return

  ask_tensorrt_version = (
      'Please specify the TensorRT version you want to use. '
      '[Leave empty to default to TensorRT %s]: '
  ) % _DEFAULT_TENSORRT_VERSION
  tf_tensorrt_version = get_from_env_or_user_or_default(
      environ_cp, 'TF_TENSORRT_VERSION', ask_tensorrt_version, _DEFAULT_TENSORRT_VERSION
  )
  environ_cp['TF_TENSORRT_VERSION'] = tf_tensorrt_version


def set_tf_nccl_version(environ_cp):
  """Set TF_NCCL_VERSION."""
  if not is_linux():
    raise ValueError('Currently NCCL is only supported on Linux platform.')

  if 'TF_NCCL_VERSION' in environ_cp:
    return

  ask_nccl_version = (
      'Please specify the locally installed NCCL version you want to use. '
      '[Leave empty to use http://github.com/nvidia/nccl]: '
  )
  tf_nccl_version = get_from_env_or_user_or_default(environ_cp, 'TF_NCCL_VERSION', ask_nccl_version, '')
  environ_cp['TF_NCCL_VERSION'] = tf_nccl_version


def _find_executable(executable: str) -> Optional[str]:
  logging.info("Trying to find path to %s...", executable)
  # Resolving the symlink is necessary for finding system headers.
  if unresolved_path := shutil.which(executable):
    return str(pathlib.Path(unresolved_path).resolve())
  return None


def _find_executable_or_die(executable_name: str, executable_path: Optional[str] = None) -> str:
  """Finds executable and resolves symlinks or raises RuntimeError.

  Resolving symlinks is sometimes necessary for finding system headers.

  Args:
    executable_name: The name of the executable that we want to find.
    executable_path: If not None, the path to the executable.

  Returns:
    The path to the executable we are looking for, after symlinks are resolved.
  Raises:
    RuntimeError: if path to the executable cannot be found.
  """
  if executable_path:
    return str(pathlib.Path(executable_path).resolve(strict=True))
  resolved_path_to_exe = _find_executable(executable_name)
  if resolved_path_to_exe is None:
    raise RuntimeError(
        f"Could not find executable `{executable_name}`! "
        "Please change your $PATH or pass the path directly like"
        f"`--{executable_name}_path=path/to/executable."
    )
  logging.info("Found path to %s at %s", executable_name, resolved_path_to_exe)

  return resolved_path_to_exe


def _get_cuda_compute_capabilities_or_die() -> list[str]:
  """Finds compute capabilities via nvidia-smi or rasies exception.

  Returns:
    list of unique, sorted strings representing compute capabilities:
  Raises:
    RuntimeError: if path to nvidia-smi couldn't be found.
    subprocess.CalledProcessError: if nvidia-smi process failed.
  """
  try:
    nvidia_smi = _find_executable_or_die("nvidia-smi")
    nvidia_smi_proc = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    # Command above returns a newline separated list of compute capabilities
    # with possible repeats. So we should unique them and sort the final result.
    capabilities = sorted(set(nvidia_smi_proc.stdout.strip().split("\n")))
    logging.info("Found CUDA compute capabilities: %s", capabilities)
    return ','.join(capabilities)
  except (RuntimeError, subprocess.CalledProcessError) as e:
    logging.info(
        "Could not find nvidia-smi, or nvidia-smi command failed. Please pass"
        " capabilities directly using --cuda_compute_capabilities."
    )
    raise e


def set_hermetic_cuda_compute_capabilities(environ_cp):
  """Set HERMETIC_CUDA_COMPUTE_CAPABILITIES."""
  while True:
    default_cuda_compute_capabilities = _get_cuda_compute_capabilities_or_die()

    ask_cuda_compute_capabilities = (
        'Please specify a list of comma-separated CUDA compute capabilities '
        'you want to build with.\nYou can find the compute capability of your '
        'device at: https://developer.nvidia.com/cuda-gpus. Each capability '
        'can be specified as "x.y" or "compute_xy" to include both virtual and'
        ' binary GPU code, or as "sm_xy" to only include the binary '
        'code.\nPlease note that each additional compute capability '
        'significantly increases your build time and binary size, and that '
        'Deepray only supports compute capabilities >= 3.5 [Default is: '
        '%s]: ' % default_cuda_compute_capabilities
    )
    hermetic_cuda_compute_capabilities = get_from_env_or_user_or_default(
        environ_cp,
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES',
        ask_cuda_compute_capabilities,
        default_cuda_compute_capabilities,
    )
    # Check whether all capabilities from the input is valid
    all_valid = True
    # Remove all whitespace characters before splitting the string
    # that users may insert by accident, as this will result in error
    hermetic_cuda_compute_capabilities = ''.join(hermetic_cuda_compute_capabilities.split())
    for compute_capability in hermetic_cuda_compute_capabilities.split(','):
      m = re.match('[0-9]+.[0-9]+', compute_capability)
      if not m:
        # We now support sm_35,sm_50,sm_60,compute_70.
        sm_compute_match = re.match('(sm|compute)_?([0-9]+[0-9]+)', compute_capability)
        if not sm_compute_match:
          print('Invalid compute capability: %s' % compute_capability)
          all_valid = False
        else:
          ver = int(sm_compute_match.group(2))
          if ver < 30:
            print(
                'ERROR: TensorFlow only supports small CUDA compute'
                ' capabilities of sm_30 and higher. Please re-specify the list'
                ' of compute capabilities excluding version %s.' % ver
            )
            all_valid = False
          if ver < 35:
            print(
                'WARNING: XLA does not support CUDA compute capabilities '
                'lower than sm_35. Disable XLA when running on older GPUs.'
            )
      else:
        ver = float(m.group(0))
        if ver < 3.0:
          print(
              'ERROR: TensorFlow only supports CUDA compute capabilities 3.0 '
              'and higher. Please re-specify the list of compute '
              'capabilities excluding version %s.' % ver
          )
          all_valid = False
        if ver < 3.5:
          print(
              'WARNING: XLA does not support CUDA compute capabilities '
              'lower than 3.5. Disable XLA when running on older GPUs.'
          )

    if all_valid:
      break

    # Reset and Retry
    environ_cp['HERMETIC_CUDA_COMPUTE_CAPABILITIES'] = ''

  # Set HERMETIC_CUDA_COMPUTE_CAPABILITIES
  environ_cp['HERMETIC_CUDA_COMPUTE_CAPABILITIES'] = (hermetic_cuda_compute_capabilities)
  write_to_bazelrc(
      'build:{} --repo_env {}="{}"'.format(
          'cuda', 'HERMETIC_CUDA_COMPUTE_CAPABILITIES', str(hermetic_cuda_compute_capabilities)
      )
  )


def set_other_cuda_vars(environ_cp):
  """Set other CUDA related variables."""
  # If CUDA is enabled, always use GPU during build and test.
  if environ_cp.get('TF_NEED_CLANG') == '1':
    write_to_bazelrc('build --config=cuda_clang')
    write_action_env('CLANG_CUDA_COMPILER_PATH', environ_cp.get('CLANG_COMPILER_PATH'))
  else:
    write_to_bazelrc('build --config=cuda')
    write_to_bazelrc('build --config=cuda_nvcc')


def system_specific_test_config(environ_cp):
  """Add default build and test flags required for TF tests to bazelrc."""
  write_to_bazelrc('test --test_size_filters=small,medium')

  # Each instance of --test_tag_filters or --build_tag_filters overrides all
  # previous instances, so we need to build up a complete list and write a
  # single list of filters for the .bazelrc file.

  # Filters to use with both --test_tag_filters and --build_tag_filters
  test_and_build_filters = ['-benchmark-test', '-no_oss', '-oss_excluded']
  # Additional filters for --test_tag_filters beyond those in
  # test_and_build_filters
  test_only_filters = ['-oss_serial']
  if is_windows():
    test_and_build_filters += ['-no_windows', '-windows_excluded']
    if (environ_cp.get('TF_NEED_CUDA', None) == '1') or (environ_cp.get('TF_NEED_ROCM', None) == '1'):
      test_and_build_filters += ['-no_windows_gpu', '-no_gpu']
    else:
      test_and_build_filters.append('-gpu')
  elif is_macos():
    test_and_build_filters += ['-gpu', '-nomac', '-no_mac', '-mac_excluded']
  elif is_linux():
    if (environ_cp.get('TF_NEED_CUDA', None) == '1') or (environ_cp.get('TF_NEED_ROCM', None) == '1'):
      test_and_build_filters.append('-no_gpu')
      write_to_bazelrc('test --test_env=LD_LIBRARY_PATH')
    else:
      test_and_build_filters.append('-gpu')


def set_system_libs_flag(environ_cp):
  """Set system libs flags."""
  syslibs = environ_cp.get('TF_SYSTEM_LIBS', '')

  if is_s390x() and 'boringssl' not in syslibs:
    syslibs = 'boringssl' + (', ' + syslibs if syslibs else '')

  if syslibs:
    if ',' in syslibs:
      syslibs = ','.join(sorted(syslibs.split(',')))
    else:
      syslibs = ','.join(sorted(syslibs.split()))
    write_action_env('TF_SYSTEM_LIBS', syslibs)

  for varname in ('PREFIX', 'LIBDIR', 'INCLUDEDIR', 'PROTOBUF_INCLUDE_PATH'):
    if varname in environ_cp:
      write_to_bazelrc('build --define=%s=%s' % (varname, environ_cp[varname]))


def set_windows_build_flags(environ_cp):
  """Set Windows specific build options."""

  # First available in VS 16.4. Speeds up Windows compile times by a lot. See
  # https://groups.google.com/a/tensorflow.org/d/topic/build/SsW98Eo7l3o/discussion
  # pylint: disable=line-too-long
  write_to_bazelrc('build --copt=/d2ReducedOptimizeHugeFunctions --host_copt=/d2ReducedOptimizeHugeFunctions')

  if get_var(
      environ_cp, 'TF_OVERRIDE_EIGEN_STRONG_INLINE', 'Eigen strong inline', True,
      ('Would you like to override eigen strong inline for some C++ '
       'compilation to reduce the compilation time?'), 'Eigen strong inline overridden.',
      'Not overriding eigen strong inline, '
      'some compilations could take more than 20 mins.'
  ):
    # Due to a known MSVC compiler issue
    # https://github.com/tensorflow/tensorflow/issues/10521
    # Overriding eigen strong inline speeds up the compiling of
    # conv_grad_ops_3d.cc and conv_ops_3d.cc by 20 minutes,
    # but this also hurts the performance. Let users decide what they want.
    write_to_bazelrc('build --define=override_eigen_strong_inline=true')


def config_info_line(name, help_text):
  """Helper function to print formatted help text for Bazel config options."""
  print('\t--config=%-12s\t# %s' % (name, help_text))


def validate_cuda_config(environ_cp):
  """Run find_cuda_config.py and return cuda_toolkit_path, or None."""

  def maybe_encode_env(env):
    """Encodes unicode in env to str on Windows python 2.x."""
    if not is_windows() or sys.version_info[0] != 2:
      return env
    for k, v in env.items():
      if isinstance(k, unicode):
        k = k.encode('ascii')
      if isinstance(v, unicode):
        v = v.encode('ascii')
      env[k] = v
    return env

  cuda_libraries = ['cuda', 'cudnn']
  if is_linux():
    if int(environ_cp.get('TF_NEED_TENSORRT', False)):
      cuda_libraries.append('tensorrt')
    if environ_cp.get('TF_NCCL_VERSION', None):
      cuda_libraries.append('nccl')
  if is_windows():
    if int(environ_cp.get('TF_NEED_TENSORRT', False)):
      cuda_libraries.append('tensorrt')
      print('WARNING: TensorRT support on Windows is experimental\n')

  paths = glob.glob('**/third_party/gpus/find_cuda_config.py', recursive=True)
  if not paths:
    raise FileNotFoundError("Can't find 'find_cuda_config.py' script inside working directory")
  proc = subprocess.Popen(
      [environ_cp['PYTHON_BIN_PATH'], paths[0]] + cuda_libraries,
      stdout=subprocess.PIPE,
      env=maybe_encode_env(environ_cp)
  )

  if proc.wait():
    # Errors from find_cuda_config.py were sent to stderr.
    print('Asking for detailed CUDA configuration...\n')
    return False

  config = dict(tuple(line.decode('ascii').rstrip().split(': ')) for line in proc.stdout)

  print('Found CUDA %s in:' % config['cuda_version'])
  print('    %s' % config['cuda_library_dir'])
  print('    %s' % config['cuda_include_dir'])

  print('Found cuDNN %s in:' % config['cudnn_version'])
  print('    %s' % config['cudnn_library_dir'])
  print('    %s' % config['cudnn_include_dir'])

  if 'tensorrt_version' in config:
    print('Found TensorRT %s in:' % config['tensorrt_version'])
    print('    %s' % config['tensorrt_library_dir'])
    print('    %s' % config['tensorrt_include_dir'])

  if config.get('nccl_version', None):
    print('Found NCCL %s in:' % config['nccl_version'])
    print('    %s' % config['nccl_library_dir'])
    print('    %s' % config['nccl_include_dir'])

  print('\n')

  environ_cp['CUDA_TOOLKIT_PATH'] = config['cuda_toolkit_path']
  return True


def get_gcc_compiler(environ_cp):
  gcc_env = environ_cp.get('CXX') or environ_cp.get('CC') or which('gcc')
  if gcc_env is not None:
    gcc_version = run_shell([gcc_env, '--version']).split()
    if gcc_version[0] in ('gcc', 'g++'):
      return gcc_env
  return None


def main():
  print("Configuring Deepray to be built from source...")

  global _DP_WORKSPACE_ROOT
  global _DP_BAZELRC
  global _DP_CURRENT_BAZEL_VERSION

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--workspace',
      type=str,
      default=os.path.abspath(os.path.dirname(__file__)),
      help='The absolute path to your active Bazel workspace.'
  )
  args = parser.parse_args()

  _DP_WORKSPACE_ROOT = args.workspace
  _DP_BAZELRC = os.path.join(_DP_WORKSPACE_ROOT, _DP_BAZELRC_FILENAME)

  # Make a copy of os.environ to be clear when functions and getting and setting
  # environment variables.
  environ_cp = dict(os.environ)

  try:
    current_bazel_version = retrieve_bazel_version()
  except subprocess.CalledProcessError as e:
    print('Error retrieving bazel version: ', e.output.decode('UTF-8').strip())
    raise e

  _DP_CURRENT_BAZEL_VERSION = convert_version_to_int(current_bazel_version)

  reset_tf_configure_bazelrc()

  cleanup_makefile()
  setup_python(environ_cp)

  write_action_env("TF_HEADER_DIR", get_tf_header_dir())
  write_action_env("TF_SHARED_LIBRARY_DIR", get_tf_shared_lib_dir())
  write_action_env("TF_SHARED_LIBRARY_NAME", get_shared_lib_name())
  write_action_env(
      "TF_SHARED_CC_LIBRARY_NAME",
      get_shared_lib_name().replace("libtensorflow_framework", "libtensorflow_cc")
  )
  write_action_env("TF_CXX11_ABI_FLAG", tf.sysconfig.CXX11_ABI_FLAG)

  # This should be replaced with a call to tf.sysconfig if it's added
  write_action_env("TF_CPLUSPLUS_VER", get_cpp_version())

  tf_version_integer = get_tf_version_integer()
  # This is used to trace the difference between Tensorflow versions.
  write_action_env("TF_VERSION_INTEGER", tf_version_integer)
  write_to_bazelrc('')

  # Ask whether we should use clang for the CPU build.
  if is_linux():
    environ_cp['TF_NEED_CLANG'] = str(choose_compiler(environ_cp))
    if environ_cp.get('TF_NEED_CLANG') == '1':
      clang_compiler_path = set_clang_compiler_path(environ_cp)
      clang_version = retrieve_clang_version(clang_compiler_path)
      disable_clang_offsetof_extension(clang_version)
    else:
      gcc_path = set_gcc_host_compiler_path(environ_cp)
      gcc_major_version = get_gcc_major_version(gcc_path)
      if gcc_major_version < 13:
        # Prevent XNNPACK from using `-mavxvnniint8` (only available in clang 16+/gcc 13+).
        write_to_bazelrc('build --define=xnn_enable_avxvnniint8=false')

  if is_windows():
    print(
        '\nWARNING: Cannot build with CUDA support on Windows.\n'
        'Starting in TF 2.11, CUDA build is not supported for Windows. '
        'For using TensorFlow GPU on Windows, you will need to build/install '
        'TensorFlow in WSL2.\n'
    )
    environ_cp['TF_NEED_CUDA'] = '0'
  else:
    environ_cp['TF_NEED_CUDA'] = str(int(get_var(environ_cp, 'TF_NEED_CUDA', 'CUDA', True)))
  if environ_cp.get('TF_NEED_CUDA') == '1' and 'TF_CUDA_CONFIG_REPO' not in environ_cp:

    # set_action_env_var(environ_cp, 'TF_NEED_TENSORRT', 'TensorRT', False, bazel_config_name='tensorrt')

    environ_save = dict(environ_cp)
    for _ in range(_DEFAULT_PROMPT_ASK_ATTEMPTS):

      if validate_cuda_config(environ_cp):
        cuda_env_names = [
            'TF_CUDA_VERSION',
            'TF_CUBLAS_VERSION',
            'TF_CUDNN_VERSION',
            'TF_TENSORRT_VERSION',
            'TF_NCCL_VERSION',
            'TF_CUDA_PATHS',
            # Items below are for backwards compatibility when not using
            # TF_CUDA_PATHS.
            'CUDA_TOOLKIT_PATH',
            'CUDNN_INSTALL_PATH',
            'NCCL_INSTALL_PATH',
            'NCCL_HDR_PATH',
            'TENSORRT_INSTALL_PATH'
        ]
        # Note: set_action_env_var above already writes to bazelrc.
        for name in cuda_env_names:
          if name in environ_cp:
            write_action_env(name, environ_cp[name])
        break

      # Restore settings changed below if CUDA config could not be validated.
      environ_cp = dict(environ_save)

      set_tf_cuda_version(environ_cp)
      set_tf_cudnn_version(environ_cp)
      if is_windows():
        set_tf_tensorrt_version(environ_cp)
      if is_linux():
        set_tf_tensorrt_version(environ_cp)
        set_tf_nccl_version(environ_cp)

      set_tf_cuda_paths(environ_cp)

    else:
      raise UserInputError(
          'Invalid CUDA setting were provided %d '
          'times in a row. Assuming to be a scripting mistake.' % _DEFAULT_PROMPT_ASK_ATTEMPTS
      )

    set_hermetic_cuda_compute_capabilities(environ_cp)
    if 'LD_LIBRARY_PATH' in environ_cp and environ_cp.get('LD_LIBRARY_PATH') != '1':
      write_action_env('LD_LIBRARY_PATH', environ_cp.get('LD_LIBRARY_PATH'))

    set_other_cuda_vars(environ_cp)
  else:
    if environ_cp.get('TF_NEED_CLANG') == '1':
      write_action_env('CLANG_COMPILER_PATH', clang_compiler_path)

  # ROCm / CUDA are mutually exclusive.
  # At most 1 GPU platform can be configured.
  gpu_platform_count = 0
  if environ_cp.get('TF_NEED_ROCM') == '1':
    gpu_platform_count += 1
  if environ_cp.get('TF_NEED_CUDA') == '1':
    gpu_platform_count += 1
  if gpu_platform_count >= 2:
    raise UserInputError('CUDA / ROCm are mututally exclusive. '
                         'At most 1 GPU platform can be configured.')

  set_cc_opt_flags(environ_cp)
  set_system_libs_flag(environ_cp)
  if is_windows():
    set_windows_build_flags(environ_cp)

  system_specific_test_config(environ_cp)

  print(
      'Preconfigured Bazel build configs. You can use any of the below by '
      'adding "--config=<>" to your build command. See .bazelrc for more '
      'details.'
  )
  config_info_line('mkl', 'Build with MKL support.')
  config_info_line('mkl_aarch64', 'Build with oneDNN and Compute Library for the Arm Architecture (ACL).')
  config_info_line('monolithic', 'Config for mostly static monolithic build.')
  config_info_line('numa', 'Build with NUMA support.')
  config_info_line('dynamic_kernels', '(Experimental) Build kernels into separate shared objects.')

  print('Preconfigured Bazel build configs to DISABLE default on features:')
  config_info_line('nogcp', 'Disable GCP support.')
  config_info_line('nonccl', 'Disable NVIDIA NCCL support.')

  print()
  if environ_cp.get("TF_NEED_CUDA", "0") == "1":
    print("> Building GPU & CPU ops")
  else:
    print("> Building only CPU ops")

  print("Build configurations successfully written to", _DP_BAZELRC, ":\n")
  print(pathlib.Path(_DP_BAZELRC).read_text())


if __name__ == '__main__':
  main()
