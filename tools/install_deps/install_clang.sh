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

CLANG_VERSION=${1:-"18"}

apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    wget software-properties-common gnupg

wget https://apt.llvm.org/llvm.sh \
    --progress=dot:mega -O /tmp/llvm-install.sh &&
    chmod u+x /tmp/llvm-install.sh &&
    /tmp/llvm-install.sh ${CLANG_VERSION}
ln -s /usr/bin/clang-17 /usr/bin/clang && ln -s /usr/bin/clang++-17 /usr/bin/clang++
clang --version
