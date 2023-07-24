#!/usr/bin/env bash
# Copyright 2023 The Deepray Authors. All Rights Reserved.
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
set -x -e

# Downloads bazelisk to ${output_dir} as `bazel`.
date

output_dir=${1:-"/usr/local/bin"}

mkdir -p "${output_dir}"
wget --progress=dot:mega -O ${output_dir}/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-$([ $(uname -m) = "aarch64" ] && echo "arm64" || echo "amd64")

chmod u+x "${output_dir}/bazel"

if [[ ! ":$PATH:" =~ :${output_dir}/?: ]]; then
    PATH="${output_dir}:$PATH"
fi

which bazel
date
