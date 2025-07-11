/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cstdlib>

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t err = cmd;                                          \
    if (err != cudaSuccess) {                                       \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#endif  // CHECK_H
