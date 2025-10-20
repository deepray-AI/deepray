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
set -xeuo pipefail

OPENMPI_VERSION=${1:-"5.0.8"}
UCX_VERSION=${2:-"1.19.0"}
UCC_VERSION=${3:-"1.5.1"}
export BUILD_DIR=/tmp
export INSTALL_DIR=/opt
export UCX_DIR=${INSTALL_DIR}/ucx
export UCC_DIR=${INSTALL_DIR}/ucc
export OMPI_DIR=${INSTALL_DIR}/openmpi

apt-get update &&
    apt-get install --no-install-recommends --yes \
        wget build-essential autoconf automake libtool git

# Install UCX
mkdir /tmp/ucx && cd /tmp/ucx
wget --no-check-certificate --progress=dot:mega -O ucx-${UCX_VERSION}.tar.gz https://github.com/openucx/ucx/archive/refs/tags/v${UCX_VERSION}.tar.gz
tar -zxf ucx-${UCX_VERSION}.tar.gz
cd ucx-${UCX_VERSION}
./autogen.sh
./configure --prefix=${UCX_DIR}
make -j $(nproc)
make install

# Install UCC
mkdir /tmp/ucc && cd /tmp/ucc
wget --no-check-certificate --progress=dot:mega -O ucc-${UCC_VERSION}.tar.gz https://github.com/openucx/ucc/archive/refs/tags/v${UCC_VERSION}.tar.gz
tar -zxf ucc-${UCC_VERSION}.tar.gz
cd ucc-${UCC_VERSION}
./autogen.sh
./configure --prefix=${UCC_DIR} --with-ucx=${UCX_DIR}
make -j $(nproc)
make install

# Install OpenMPI
mkdir /tmp/openmpi && cd /tmp/openmpi
wget --no-check-certificate --progress=dot:mega -O openmpi-${OPENMPI_VERSION}.tar.gz https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OPENMPI_VERSION}.tar.gz
tar -zxf openmpi-${OPENMPI_VERSION}.tar.gz
cd openmpi-${OPENMPI_VERSION}
./configure --enable-orterun-prefix-by-default --prefix=${OMPI_DIR} --with-ucx=${UCX_DIR} --with-ucc=${UCC_DIR}
make -j $(nproc)
make install
ldconfig

# Configure OpenMPI
cat >bashrc.txt <<'EOF'
export OPENMPI_HOME=/opt/openmpi
export PATH="${OPENMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
EOF
cat bashrc.txt >>/root/.bashrc

export OPENMPI_HOME=/opt/openmpi
export PATH="${OPENMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${OPENMPI_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mpirun --version &&
    rm -rf ${BUILD_DIR}
