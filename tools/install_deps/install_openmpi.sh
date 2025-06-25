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

# Install Open MPI
mkdir /tmp/openmpi &&
    cd /tmp/openmpi
wget --progress=dot:mega -O openmpi-${OPENMPI_VERSION}.tar.gz https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OPENMPI_VERSION}.tar.gz &&
    tar -zxf openmpi-${OPENMPI_VERSION}.tar.gz &&
    cd openmpi-${OPENMPI_VERSION} &&
    ./configure --enable-orterun-prefix-by-default &&
    make -j $(nproc) &&
    make install &&
    ldconfig

# Install OpenSSH for MPI to communicate between containers
apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    openssh-client openssh-server &&
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config &&
    echo "    UserKnownHostsFile /dev/null" >>/etc/ssh/ssh_config &&
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

mpirun --version &&
    rm -rf /tmp/openmpi
