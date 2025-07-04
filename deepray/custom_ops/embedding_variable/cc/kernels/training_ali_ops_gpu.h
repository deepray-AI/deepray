/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_

#if GOOGLE_CUDA
#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, typename Tindex>
struct KvSparseApplyAdagrad {
  void operator()(int32 num_items, Allocator* alloc,
                  EmbeddingVar<Tindex, T>* var, EmbeddingVar<Tindex, T>* accum,
                  const Tindex* key_base, const T* grad, T lr, int64 gs,
                  const Device& device);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyFtrl {
  void operator()(int32 num_items, Allocator* alloc, EmbeddingVar<TKey, T>* var,
                  EmbeddingVar<TKey, T>* accum, EmbeddingVar<TKey, T>* linear,
                  const TKey* key_base, const T* grad, T lr, T l1, T l2,
                  T lr_power, bool has_l2_shrinkage, T l2_shrinkage,
                  const Device& device);
};

template <typename Device, typename T, typename Tindex, typename Tstep>
struct KvSparseApplyAdam {
  Status operator()(const Device& d, EmbeddingVar<Tindex, T>* var,
                    EmbeddingVar<Tindex, T>* m, EmbeddingVar<Tindex, T>* v,
                    typename TTypes<T>::ConstScalar beta1_power_scalar,
                    typename TTypes<T>::ConstScalar beta2_power_scalar,
                    typename TTypes<Tindex>::ConstVec indices_vec,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<T>::ConstScalar lr_scalar,
                    typename TTypes<T>::ConstScalar beta1_scalar,
                    typename TTypes<T>::ConstScalar beta2_scalar,
                    typename TTypes<T>::ConstScalar epsilon_scalar,
                    typename TTypes<Tstep>::ConstScalar global_step_scalar,
                    const int64 inner_dim, Allocator* alloc);
};

template <typename Device, typename T, typename Tindex, typename Tstep>
struct KvSparseApplyAdamAsync {
  Status operator()(const Device& d, EmbeddingVar<Tindex, T>* var,
                    EmbeddingVar<Tindex, T>* m, EmbeddingVar<Tindex, T>* v,
                    typename TTypes<T>::Scalar beta1_power_scalar,
                    typename TTypes<T>::Scalar beta2_power_scalar,
                    typename TTypes<Tindex>::ConstVec indices_vec,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<T>::ConstScalar lr_scalar,
                    typename TTypes<T>::ConstScalar beta1_scalar,
                    typename TTypes<T>::ConstScalar beta2_scalar,
                    typename TTypes<T>::ConstScalar epsilon_scalar,
                    typename TTypes<Tstep>::ConstScalar global_step_scalar,
                    bool apply_sparse_rmsprop, const int64 inner_dim,
                    Allocator* alloc);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdamAsyncSparseRmspropHbm {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, int64 task_size, const Device& device);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdamHbm {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T beta1_power, T beta2_power, int64 task_size,
                  const Device& device);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdagradHbm {
  void operator()(int block_size, int embedding_dim, T** dev_a, T** dev_v,
                  const T* grad_base, T lr_scalar, int64 task_size,
                  const Device& device);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdamAsyncHbm {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T* beta1_power_ptr, T* beta2_power_ptr,
                  int64 task_size, const Device& device);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdamWHbm {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T weight_decay, int64 task_size,
                  const Device& device);
};
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_
