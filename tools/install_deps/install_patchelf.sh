#!/usr/bin/env bash
# Copyright 2025 The Deepray Authors. All Rights Reserved.
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

PATCHELF_VERSION=${1:-"0.18.0"}

TEMP_DIR=$(mktemp -d)
cd "${TEMP_DIR}" || exit 1

# auditwheel repair requires patchelf >= 0.14.
wget --no-hsts --no-check-certificate https://github.com/NixOS/patchelf/releases/download/${PATCHELF_VERSION}/patchelf-${PATCHELF_VERSION}-x86_64.tar.gz \
    --progress=dot:mega -O patchelf.tar.gz
tar -xvf patchelf.tar.gz

cp ./bin/patchelf /usr/local/bin/
chmod +x /usr/local/bin/patchelf

patchelf --version
rm -rf "${TEMP_DIR}"
