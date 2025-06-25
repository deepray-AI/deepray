load("@bazel_skylib//rules:build_test.bzl", "build_test")

# Copyright 2024 The Deepray Authors.
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
load("@rules_license//rules:license.bzl", "license")
load("//third_party/py:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//deepray:__subpackages__"],
)

license(
    name = "license",
    package_name = "deepray",
)

exports_files([
    "LICENSE",
    "setup.py",
    "MANIFEST.in",
    "README.md",
    "requirements.txt",
])

###############################################################################
# PIP Package
###############################################################################
sh_binary(
    name = "build_pip_pkg",
    srcs = ["//build_deps:build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "requirements.txt",
        "setup.py",
        "//deepray",
    ],
)
