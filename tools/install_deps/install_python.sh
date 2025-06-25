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

PY_VERSION=${1:-"3.10"}

if [ "$PY_VERSION" = "3.10" ]; then
    echo "3.10 selected!"
    apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        python${PY_VERSION} \
        python${PY_VERSION}-dev \
        python${PY_VERSION}-distutils &&
        apt-get clean && rm -rf /var/lib/apt/lists/*

    ln -s /usr/bin/python${PY_VERSION} /usr/bin/python

    curl -O https://bootstrap.pypa.io/get-pip.py &&
        python get-pip.py &&
        rm get-pip.py

elif [ "$PY_VERSION" = "3.9" ]; then
    echo "Not supported yet."
elif [ "$PY_VERSION" = "3.8" ]; then
    echo "3.8 selected!"
    apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        wget zlib1g-dev libffi-dev libbz2-dev libssl-dev libcurl4-openssl-dev libcurl3-dev
    cd /tmp/
    wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz --progress=dot:mega &&
        tar -xzvf Python-3.8.10.tgz &&
        cd Python-3.8.10 &&
        ./configure --prefix=/usr/local/python3 --enable-optimizations &&
        make -j$(nproc) &&
        make install
    rm -f /usr/bin/python &&
        rm -f /usr/bin/python3 &&
        rm -f /usr/bin/pip &&
        ln -s /usr/local/python3/bin/python3.8 /usr/bin/python &&
        ln -s /usr/local/python3/bin/python3.8 /usr/bin/python3 &&
        ln -s /usr/local/python3/bin/pip3 /usr/bin/pip
    pip install pip setuptools -U
else
    echo "No python version selected."
fi

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -V
pip -V
