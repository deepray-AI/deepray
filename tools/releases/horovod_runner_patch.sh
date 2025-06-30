#!/bin/bash
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

set -e -x

SITE_PKG_LOCATION=$(python -c "import site; print(site.getsitepackages()[0])")
sed -i "s/\(command = \[executable, '-m', 'horovod.runner.run_task', str(driver_ip), str(run_func_server_port)\]\)/\1 + sys.argv[1:]/" ${SITE_PKG_LOCATION}/horovod/runner/launch.py
sed -i 's/sys.argv/sys.argv[:3]/g' ${SITE_PKG_LOCATION}/horovod/runner/run_task.py
