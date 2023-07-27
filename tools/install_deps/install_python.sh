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

PY_VERSION=${1:-"3.8"}

apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    python${PY_VERSION} \
    python${PY_VERSION}-dev \
    python${PY_VERSION}-distutils &&
    apt-get clean && rm -rf /var/lib/apt/lists/*

ln -s /usr/bin/python${PY_VERSION} /usr/bin/python

curl -O https://bootstrap.pypa.io/get-pip.py &&
    python get-pip.py &&
    rm get-pip.py

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple