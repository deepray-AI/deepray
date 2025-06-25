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
load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")

licenses(["notice"])

package(
    default_visibility = ["//deepray:__subpackages__"],
)

exports_files(["build_pip_pkg.sh"])

compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
        "--resolver=backtracking",
        "-i https://pypi.tuna.tsinghua.edu.cn/simple",
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)
