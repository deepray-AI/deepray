#!/usr/bin/env bash

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
set -eu
set -o pipefail

num_gpu=${1:-"4"}
profile=${2:-"false"}

if [ $num_gpu -gt 1 ]; then
    hvd_command="horovodrun -np $num_gpu "
else
    hvd_command=""
fi

if [ "$profile" = "true" ]; then
    nsys_command="--timeline-filename $RESULTS_DIR/timeline.json --timeline-mark-cycles"
    echo "profile activated"
else
    nsys_command=""
fi

set -x
$hvd_command $nsys_command python tensorflow2_synthetic_benchmark.py
set +x
