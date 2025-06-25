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

CONDA_DIR=/opt/conda

apt-get update &&
    apt-get install --no-install-recommends --yes \
        wget

if [ "$PY_VERSION" = "3.10" ]; then
    echo "3.10 selected!"
    wget --no-hsts --no-check-certificate --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh
elif [ "$PY_VERSION" = "3.9" ]; then
    echo "Not supported yet."
    wget --no-hsts --no-check-certificate --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/4.10.0-0/Mambaforge-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh
elif [ "$PY_VERSION" = "3.8" ]; then
    echo "3.8 selected!"
    wget --no-hsts --no-check-certificate --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/4.10.0-0/Mambaforge-$(uname)-$(uname -m).sh -O /tmp/miniforge.sh
else
    echo "No python version selected."
fi

/bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} &&
    rm /tmp/miniforge.sh

export PATH=${CONDA_DIR}/bin:${PATH}

conda clean --tarballs --index-cache --packages --yes &&
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete &&
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete &&
    conda clean --force-pkgs-dirs --all --yes

echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >>/etc/skel/.bashrc &&
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >>~/.bashrc

conda init

# https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found
conda install -c conda-forge libstdcxx-ng -y

rm -f /bin/python3
ln -s /opt/conda/bin/python /bin/python3
