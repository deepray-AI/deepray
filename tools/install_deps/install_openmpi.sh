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

# Install Open MPI
wget --progress=dot:mega -O /tmp/openmpi-4.1.4-bin.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz &&
    cd /tmp && tar -zxf /tmp/openmpi-4.1.4-bin.tar.gz &&
    mkdir openmpi-4.1.4/build && cd openmpi-4.1.4/build && ../configure --prefix=/usr/local &&
    make -j all && make install && ldconfig &&
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
mkdir -p /var/run/sshd
cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking >/etc/ssh/ssh_config.new &&
    echo "    StrictHostKeyChecking no" >>/etc/ssh/ssh_config.new &&
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
