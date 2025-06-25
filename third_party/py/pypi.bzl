# Copyright 2024 The JAX SC Authors.
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
"""Utilities for working with pypi dependencies."""

load("@pypi//:requirements.bzl", "requirement")

# Use a map for python packages whose names don't precisely correspond to the
# import names.  For example, the 'absl' python is 'absl_py'.  These are the
# exceptions.  Most packages do have a direct correspondence (e.g. jax, numpy).
_PYPI_PACKAGE_MAP = {
    "absl": "absl_py",
    "google/protobuf": "protobuf",
    "tree": "dm-tree",
}

def pypi_requirement(dep):
    """Determines the pypi package dependency for a target.

    Args:
        dep: dependency target

    Returns:
        pypi requirement.
    """
    package_name = dep

    # Remove target in root package.
    target_sep = package_name.find(":")
    if target_sep >= 0:
        package_name = package_name[:target_sep]

    # Check map if there is a direct dependency replacement.
    package_name = _PYPI_PACKAGE_MAP.get(package_name, package_name)

    # Remove any subpackage names.
    path_sep = package_name.find("/")
    if path_sep >= 0:
        package_name = package_name[:path_sep]

    # Replace any known package name substitutions.
    package_name = _PYPI_PACKAGE_MAP.get(package_name, package_name)

    return requirement(package_name)
