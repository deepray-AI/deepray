#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
wget --no-hsts --no-check-certificate --progress=dot:mega -O /usr/local/bin/clang-format-9 https://github.com/DoozyX/clang-format-lint-action/raw/master/clang-format/clang-format9
chmod +x /usr/local/bin/clang-format-9
ln -s /usr/local/bin/clang-format-9 /usr/local/bin/clang-format
clang-format --version
