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

CMAKE_VERSION=${1:-"3.31.0"}

wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    --progress=dot:mega -O /tmp/cmake-install.sh &&
    chmod u+x /tmp/cmake-install.sh &&
    mkdir /usr/bin/cmake &&
    /tmp/cmake-install.sh --skip-license --prefix=/usr &&
    rm /tmp/cmake-install.sh &&
    cmake --version
