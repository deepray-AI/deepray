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

OPENMPI_VERSION=${1:-"5.0.7"}

apt-get update &&
    apt-get install --no-install-recommends --yes \
        wget build-essential

# Install Open MPI
mkdir /tmp/openmpi &&
    cd /tmp/openmpi
wget --no-check-certificate --progress=dot:mega -O openmpi-${OPENMPI_VERSION}.tar.gz https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OPENMPI_VERSION}.tar.gz
tar -zxf openmpi-${OPENMPI_VERSION}.tar.gz
cd openmpi-${OPENMPI_VERSION}
./configure --enable-orterun-prefix-by-default --prefix=/opt/openmpi
make -j $(nproc)
make install
ldconfig

# Configure OpenMPI
cat >bashrc.txt <<'EOF'
export OPENMPI_HOME=/opt/openmpi
export PATH="${OPENMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_HOME}/lib:${LD_LIBRARY_PATH}"
EOF
cat bashrc.txt >>/root/.bashrc

export OPENMPI_HOME=/opt/openmpi
export PATH="${OPENMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_HOME}/lib:${LD_LIBRARY_PATH}"

mpirun --version &&
    rm -rf /tmp/openmpi
