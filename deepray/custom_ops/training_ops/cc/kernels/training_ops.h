/* Copyright 2023 The Deepray Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DEEPRAY_KERNELS_TRAINING_OPS_H_
#define DEEPRAY_KERNELS_TRAINING_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace deepray {
namespace functor {

// Each training algorithm has a ApplyXYZ functor struct declared in
// this header file. They are specialized for different devices
// (CPUDevice in training_ops.cc or GPUDevice in training_ops_gpu.cc).

template <typename Device, typename T, typename Tindex>
struct SparseApplyAdam {
  Status operator()(const Device& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<T>::ConstScalar beta1_power,
                    typename TTypes<T>::ConstScalar beta2_power,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar beta1,
                    typename TTypes<T>::ConstScalar beta2,
                    typename TTypes<T>::ConstScalar epsilon,
                    typename TTypes<Tindex>::ConstVec indices,
                    const int64 inner_dim);
};

}  // end namespace functor
}  // end namespace deepray
}  // namespace tensorflow

#endif  // DEEPRAY_KERNELS_TRAINING_OPS_H_
