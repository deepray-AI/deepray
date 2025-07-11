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
TF_SHARED_LIBRARY_NAME=$(grep -r TF_SHARED_LIBRARY_NAME .dp_configure.bazelrc | awk -F= '{print$2}')
TF_SHARED_CC_LIBRARY_NAME=$(grep -r TF_SHARED_CC_LIBRARY_NAME .dp_configure.bazelrc | awk -F= '{print$2}')
POLICY_JSON="${SITE_PKG_LOCATION}/auditwheel/policy/manylinux-policy.json"
sed -i "s/libresolv.so.2\"/libresolv.so.2\", $TF_SHARED_LIBRARY_NAME, $TF_SHARED_CC_LIBRARY_NAME/g" $POLICY_JSON
