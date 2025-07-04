/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "training_ali_ops_gpu.h"

#include "deepray/custom_ops/embedding_variable/cc/embedding/gpu_hash_table.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
__device__ T impl_sqrt(T x) {
  return sqrt(x);
}
template <typename T>
__device__ T impl_rsqrt(T x) {
  return rsqrt(x);
}
template <>
__device__ Eigen::half impl_sqrt(Eigen::half x) {
  return __float2half(sqrt(__half2float(x)));
}
template <>
__device__ Eigen::half impl_rsqrt(Eigen::half x) {
  return __float2half(rsqrt(__half2float(x)));
}

template <typename Tindex, typename Value>
__global__ void kv_sparse_apply_adagrad_kernel(
    const Tindex* key_base, int32* item_idxs, int64 dim, Value** d_banks,
    bool** d_flags, int32 var_slot_idx, int32 acc_slot_idx, int32 slot_num,
    int32 bank_size, Value lr, const Value* grad, Value* var_default_v,
    Value* acc_default_v, int32 var_default_v_num, int32 acc_default_v_num) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto acc_slot_offset = bank_idx * slot_num + acc_slot_idx;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool acc_stored = d_flags[acc_slot_offset][offset_in_bank];
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] =
          var_default_v[(*(key_base + item_idx) % var_default_v_num) * dim +
                        id];
    }
  }
  if (acc_default_v != nullptr && acc_stored == false) {
    d_flags[acc_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[acc_slot_offset][offset_in_bank * dim + id] =
          acc_default_v[(*(key_base + item_idx) % acc_default_v_num) * dim +
                        id];
    }
  }
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    Value g = grad[item_idx * dim + id];
    Value* acc = &d_banks[acc_slot_offset][tmp_offset];
    (*acc) += g * g;
    d_banks[var_slot_offset][tmp_offset] -= lr * g * rsqrtf(*acc);
  }
}

template <typename T, typename Tindex>
struct KvSparseApplyAdagrad<GPUDevice, T, Tindex> {
  void operator()(int32 num_items, Allocator* alloc,
                  EmbeddingVar<Tindex, T>* var, EmbeddingVar<Tindex, T>* accum,
                  const Tindex* key_base, const T* grad, T lr, int64 gs,
                  const GPUDevice& device) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc, num_items,
                                                       AllocationAttributes());
    var->LookupOrCreateKey(key_base, item_idxs, num_items, device, gs);
    auto const block_size = 256;
    auto const grid_size = num_items;
    GPUHashTable<Tindex, T>* hashtable = var->HashTable();
    TF_CHECK_OK(GpuLaunchKernel(
        kv_sparse_apply_adagrad_kernel<Tindex, T>, grid_size, block_size, 0,
        device.stream(), key_base, item_idxs, var->ValueLen(),
        hashtable->d_bank_ptrs, hashtable->d_existence_flag_ptrs, var->EmbIdx(),
        accum->EmbIdx(), var->SlotNum(), hashtable->initial_bank_size, lr, grad,
        var->GetDefaultValuePtr(), accum->GetDefaultValuePtr(),
        var->GetDefaultValueDim(), accum->GetDefaultValueDim()));
    TypedAllocator::Deallocate(alloc, item_idxs, num_items);
  }
};

template <typename TKey, typename T>
struct KvSparseApplyAdagradHbm<GPUDevice, TKey, T> {
  void operator()(int block_size, int embedding_dim, T** dev_a, T** dev_v,
                  const T* grad_base, T lr_scalar, int64 task_size,
                  const GPUDevice& device) {
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyAdagradGPU<T>,
        (task_size + block_size - 1) / block_size * embedding_dim, block_size,
        0, device.stream(), dev_a, dev_v, grad_base, lr_scalar, embedding_dim,
        task_size));
  }
};

template <typename TKey, typename T>
__global__ void KvSparseApplyAdamKernel(
    const TKey* key_base, int32* item_idxs, int64 dim, T** d_banks,
    bool** d_flags, int32 var_slot_idx, int32 v_slot_idx, int32 m_slot_idx,
    int32 slot_num, int32 bank_size, const T* beta1_scalar,
    const T* beta2_scalar, const T* beta1_power_scalar,
    const T* beta2_power_scalar, const T* epsilon_scalar, const T* lr_scalar,
    const T* grad, T* var_default_v, T* v_default_v, T* m_default_v,
    int32 var_default_v_num, int32 v_default_v_num, int32 m_default_v_num) {
  const T lr = *lr_scalar;
  const T beta1 = *beta1_scalar;
  const T beta2 = *beta2_scalar;
  const T beta1_power = *beta1_power_scalar;
  const T beta2_power = *beta2_power_scalar;
  const T epsilon = *epsilon_scalar;

  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto v_slot_offset = bank_idx * slot_num + v_slot_idx;
  auto m_slot_offset = bank_idx * slot_num + m_slot_idx;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool v_stored = d_flags[v_slot_offset][offset_in_bank];
  bool m_stored = d_flags[m_slot_offset][offset_in_bank];
  const T alpha = lr * sqrt(static_cast<T>(1) - beta2_power) /
                  (static_cast<T>(1) - beta1_power);
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] =
          var_default_v[(*(key_base + item_idx) % var_default_v_num) * dim +
                        id];
    }
  }
  if (v_default_v != nullptr && v_stored == false) {
    d_flags[v_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[v_slot_offset][offset_in_bank * dim + id] =
          v_default_v[(*(key_base + item_idx) % v_default_v_num) * dim + id];
    }
  }
  if (m_default_v != nullptr && m_stored == false) {
    d_flags[m_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[m_slot_offset][offset_in_bank * dim + id] =
          m_default_v[(*(key_base + item_idx) % m_default_v_num) * dim + id];
    }
  }
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    T grad_a = grad[item_idx * dim + id];
    T& var_a = d_banks[var_slot_offset][tmp_offset];
    T& v_a = d_banks[v_slot_offset][tmp_offset];
    T& m_a = d_banks[m_slot_offset][tmp_offset];

    m_a = m_a * beta1 + grad_a * (static_cast<T>(1) - beta1);
    v_a = v_a * beta2 + grad_a * grad_a * (static_cast<T>(1) - beta2);
    var_a -= (m_a * alpha) / (sqrt(v_a) + epsilon);
  }
}

template <typename T, typename Tindex, typename Tstep>
struct KvSparseApplyAdam<GPUDevice, T, Tindex, Tstep> {
  Status operator()(const GPUDevice& d, EmbeddingVar<Tindex, T>* var,
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
                    const int64 inner_dim, Allocator* alloc) {
    const int32 N = indices_vec.dimension(0);
    if (N <= 0) return OkStatus();

    if (inner_dim > 0) {
      const int64 global_step = global_step_scalar();
      int32* item_idxs =
          TypedAllocator::Allocate<int32>(alloc, N, AllocationAttributes());
      var->LookupOrCreateKey(indices_vec.data(), item_idxs, N, d, global_step);
      auto const block_size = 256;
      auto const grid_size = N;
      auto hashtable = var->HashTable();
      TF_CHECK_OK(GpuLaunchKernel(
          KvSparseApplyAdamKernel<Tindex, T>, grid_size, block_size, 0,
          d.stream(), indices_vec.data(), item_idxs, var->ValueLen(),
          hashtable->d_bank_ptrs, hashtable->d_existence_flag_ptrs,
          var->EmbIdx(), v->EmbIdx(), m->EmbIdx(), var->SlotNum(),
          hashtable->initial_bank_size, beta1_scalar.data(),
          beta2_scalar.data(), beta1_power_scalar.data(),
          beta2_power_scalar.data(), epsilon_scalar.data(), lr_scalar.data(),
          grad.data(), var->GetDefaultValuePtr(), v->GetDefaultValuePtr(),
          m->GetDefaultValuePtr(), var->GetDefaultValueDim(),
          v->GetDefaultValueDim(), m->GetDefaultValueDim()));
      TypedAllocator::Deallocate(alloc, item_idxs, N);
    }

    return OkStatus();
  }
};

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename TKey, typename Value>
__global__ void kv_sparse_apply_ftrl_kernel(
    const TKey* key_base, int32* item_idxs, int64 dim, Value** d_banks,
    bool** d_flags, int32 var_slot_idx, int32 acc_slot_idx,
    int32 linear_slot_idx, int32 slot_num, int32 bank_size, Value lr_scalar,
    const Value* grad, Value* var_default_v, Value* acc_default_v,
    Value* linear_default_v, int32 var_default_v_num, int32 acc_default_v_num,
    int32 linear_default_v_num, Value l1_scalar, Value l2_scalar,
    Value lr_power_scalar, bool has_l2_shrinkage, Value l2_shrinkage_scalar) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto acc_slot_offset = bank_idx * slot_num + acc_slot_idx;
  auto linear_slot_offset = bank_idx * slot_num + linear_slot_idx;
  extern __shared__ __align__(sizeof(Value)) unsigned char shared[];
  Value* new_acc = reinterpret_cast<Value*>(shared);
  __shared__ Value linear_sqr_sum;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool acc_stored = d_flags[acc_slot_offset][offset_in_bank];
  bool linear_stored = d_flags[linear_slot_offset][offset_in_bank];
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] =
          var_default_v[(*(key_base + item_idx) % var_default_v_num) * dim +
                        id];
    }
  }
  if (acc_default_v != nullptr && acc_stored == false) {
    d_flags[acc_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[acc_slot_offset][offset_in_bank * dim + id] =
          acc_default_v[(*(key_base + item_idx) % acc_default_v_num) * dim +
                        id];
    }
  }
  if (linear_default_v != nullptr && linear_stored == false) {
    d_flags[linear_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[linear_slot_offset][offset_in_bank * dim + id] =
          linear_default_v[(*(key_base + item_idx) % linear_default_v_num) *
                               dim +
                           id];
    }
  }
  Value linear_tmp = 0;
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    Value* var_p = &d_banks[var_slot_offset][tmp_offset];
    Value g = grad[item_idx * dim + id];
    Value gg;
    if (has_l2_shrinkage) {
      gg = g + 2 * l2_shrinkage_scalar * (*var_p);
    } else {
      gg = g;
    }
    Value* acc_p = &d_banks[acc_slot_offset][tmp_offset];
    new_acc[id] = *acc_p + gg * gg;
    Value* linear_p = &d_banks[linear_slot_offset][tmp_offset];
    if (lr_power_scalar == -0.5) {
      (*linear_p) +=
          gg - (sqrtf(new_acc[id]) - sqrtf(*acc_p)) / lr_scalar * (*var_p);
    } else {
      (*linear_p) += gg - (powf(new_acc[id], -lr_power_scalar) -
                           powf(*acc_p, -lr_power_scalar)) /
                              lr_scalar * (*var_p);
    }
    linear_tmp += (*linear_p) * (*linear_p);
  }
  linear_tmp = blockReduceSum<Value>(linear_tmp);
  if (threadIdx.x == 0) {
    linear_sqr_sum = linear_tmp;
  }
  __syncthreads();
  Value linear_norm = sqrtf(linear_sqr_sum);
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    Value* var_p = &d_banks[var_slot_offset][tmp_offset];
    Value* acc_p = &d_banks[acc_slot_offset][tmp_offset];
    Value* linear_p = &d_banks[linear_slot_offset][tmp_offset];
    Value g = grad[item_idx * dim + id];
    if (linear_norm > l1_scalar) {
      if (lr_power_scalar == -0.5) {
        auto eta_rec = sqrtf(new_acc[id]) / lr_scalar;
        auto coef = (l1_scalar - linear_norm) /
                    ((eta_rec + 2 * l2_scalar) * linear_norm);
        *var_p = coef * (*linear_p);
      } else {
        auto eta_rec = powf(new_acc[id], -lr_power_scalar) / lr_scalar;
        auto coef = (l1_scalar - linear_norm) /
                    ((eta_rec + 2 * l2_scalar) * linear_norm);
        *var_p = coef * (*linear_p);
      }
    } else {
      *var_p = 0;
    }
    (*acc_p) += g * g;
  }
}

template <typename TKey, typename T>
struct KvSparseApplyFtrl<GPUDevice, TKey, T> {
  void operator()(int32 num_items, Allocator* alloc, EmbeddingVar<TKey, T>* var,
                  EmbeddingVar<TKey, T>* accum, EmbeddingVar<TKey, T>* linear,
                  const TKey* key_base, const T* grad, T lr, T l1, T l2,
                  T lr_power, bool has_l2_shrinkage, T l2_shrinkage,
                  const GPUDevice& device) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc, num_items,
                                                       AllocationAttributes());
    var->LookupOrCreateKey(key_base, item_idxs, num_items, device);
    auto const block_size = 256;
    auto const grid_size = num_items;
    auto hashtable = var->HashTable();
    TF_CHECK_OK(GpuLaunchKernel(
        kv_sparse_apply_ftrl_kernel<TKey, T>, grid_size, block_size,
        (var->ValueLen()) * sizeof(T), device.stream(), key_base, item_idxs,
        var->ValueLen(), hashtable->d_bank_ptrs,
        hashtable->d_existence_flag_ptrs, var->EmbIdx(), accum->EmbIdx(),
        linear->EmbIdx(), var->SlotNum(), hashtable->initial_bank_size, lr,
        grad, var->GetDefaultValuePtr(), accum->GetDefaultValuePtr(),
        linear->GetDefaultValuePtr(), var->GetDefaultValueDim(),
        accum->GetDefaultValueDim(), linear->GetDefaultValueDim(), l1, l2,
        lr_power, has_l2_shrinkage, l2_shrinkage));
    TypedAllocator::Deallocate(alloc, item_idxs, num_items);
  }
};

template <typename TKey, typename T>
__global__ void KvSparseApplyAdamAsyncKernel(
    const TKey* key_base, int32* item_idxs, int64 dim, T** d_banks,
    bool** d_flags, int32 var_slot_idx, int32 v_slot_idx, int32 m_slot_idx,
    int32 slot_num, int32 bank_size, const T* beta1_scalar,
    const T* beta2_scalar, const T* beta1_power_scalar,
    const T* beta2_power_scalar, const T* epsilon_scalar, const T* lr_scalar,
    const T* grad, T* var_default_v, T* v_default_v, T* m_default_v,
    int32 var_default_v_num, int32 v_default_v_num, int32 m_default_v_num,
    bool apply_sparse_rmsprop) {
  const T lr = *lr_scalar;
  const T beta1 = *beta1_scalar;
  const T beta2 = *beta2_scalar;
  const T beta1_power = *beta1_power_scalar;
  const T beta2_power = *beta2_power_scalar;
  const T epsilon = *epsilon_scalar;

  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto v_slot_offset = bank_idx * slot_num + v_slot_idx;
  auto m_slot_offset = bank_idx * slot_num + m_slot_idx;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool v_stored = d_flags[v_slot_offset][offset_in_bank];
  bool m_stored = d_flags[m_slot_offset][offset_in_bank];
  const T alpha = lr * sqrt(static_cast<T>(1) - beta2_power) /
                  (static_cast<T>(1) - beta1_power);
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] =
          var_default_v[(*(key_base + item_idx) % var_default_v_num) * dim +
                        id];
    }
  }
  if (v_default_v != nullptr && v_stored == false) {
    d_flags[v_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[v_slot_offset][offset_in_bank * dim + id] =
          v_default_v[(*(key_base + item_idx) % v_default_v_num) * dim + id];
    }
  }
  if (m_default_v != nullptr && m_stored == false) {
    d_flags[m_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[m_slot_offset][offset_in_bank * dim + id] =
          m_default_v[(*(key_base + item_idx) % m_default_v_num) * dim + id];
    }
  }

  if (apply_sparse_rmsprop) {
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      auto tmp_offset = offset_in_bank * dim + id;
      T grad_a = grad[item_idx * dim + id];
      T& var_a = d_banks[var_slot_offset][tmp_offset];
      T& v_a = d_banks[v_slot_offset][tmp_offset];
      T& m_a = d_banks[m_slot_offset][tmp_offset];

      v_a = v_a * beta2 + grad_a * grad_a * (static_cast<T>(1) - beta2);
      m_a = m_a * beta1 + rsqrt(v_a + epsilon) * lr * grad_a;
      var_a -= m_a;
    }
  } else {
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      auto tmp_offset = offset_in_bank * dim + id;
      T grad_a = grad[item_idx * dim + id];
      T& var_a = d_banks[var_slot_offset][tmp_offset];
      T& v_a = d_banks[v_slot_offset][tmp_offset];
      T& m_a = d_banks[m_slot_offset][tmp_offset];

      m_a = m_a * beta1 + grad_a * (static_cast<T>(1) - beta1);
      v_a = v_a * beta2 + grad_a * grad_a * (static_cast<T>(1) - beta2);
      var_a -= (m_a * alpha) / (sqrt(v_a) + epsilon);
    }
  }
}

template <typename T, typename Tindex, typename Tstep>
struct KvSparseApplyAdamAsync<GPUDevice, T, Tindex, Tstep> {
  Status operator()(const GPUDevice& d, EmbeddingVar<Tindex, T>* var,
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
                    Allocator* alloc) {
    const int32 N = indices_vec.dimension(0);
    if (N <= 0) return OkStatus();

    if (inner_dim > 0) {
      const int64 global_step = global_step_scalar();
      int32* item_idxs =
          TypedAllocator::Allocate<int32>(alloc, N, AllocationAttributes());
      var->LookupOrCreateKey(indices_vec.data(), item_idxs, N, d, global_step);
      auto const block_size = 256;
      auto const grid_size = N;
      auto hashtable = var->HashTable();
      TF_CHECK_OK(GpuLaunchKernel(
          KvSparseApplyAdamAsyncKernel<Tindex, T>, grid_size, block_size, 0,
          d.stream(), indices_vec.data(), item_idxs, var->ValueLen(),
          hashtable->d_bank_ptrs, hashtable->d_existence_flag_ptrs,
          var->EmbIdx(), v->EmbIdx(), m->EmbIdx(), var->SlotNum(),
          hashtable->initial_bank_size, beta1_scalar.data(),
          beta2_scalar.data(), beta1_power_scalar.data(),
          beta2_power_scalar.data(), epsilon_scalar.data(), lr_scalar.data(),
          grad.data(), var->GetDefaultValuePtr(), v->GetDefaultValuePtr(),
          m->GetDefaultValuePtr(), var->GetDefaultValueDim(),
          v->GetDefaultValueDim(), m->GetDefaultValueDim(),
          apply_sparse_rmsprop));
      TypedAllocator::Deallocate(alloc, item_idxs, N);
    }

    if (!apply_sparse_rmsprop) {
      beta1_power_scalar.device(d) = beta1_power_scalar * beta1_scalar;
      beta2_power_scalar.device(d) = beta2_power_scalar * beta2_scalar;
    }

    return OkStatus();
  }
};

template <typename TKey, typename T>
struct KvSparseApplyAdamAsyncHbm<GPUDevice, TKey, T> {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T* beta1_power_ptr, T* beta2_power_ptr,
                  int64 task_size, const GPUDevice& device) {
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyAdamAsyncGPU<T>,
        (task_size + block_size - 1) / block_size * embedding_dim, block_size,
        0, device.stream(), dev_var, dev_m, dev_v, grad_base, lr, beta1, beta2,
        epsilon, beta1_power_ptr, beta2_power_ptr, embedding_dim, task_size));
  }
};

template <typename TKey, typename T>
struct KvSparseApplyAdamAsyncSparseRmspropHbm<GPUDevice, TKey, T> {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, int64 task_size, const GPUDevice& device) {
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyAdamAsyncSparseRmspropGPU<T>,
        (task_size + block_size - 1) / block_size * embedding_dim, block_size,
        0, device.stream(), dev_var, dev_m, dev_v, grad_base, lr, beta1, beta2,
        epsilon, embedding_dim, task_size));
  }
};

template <typename TKey, typename T>
struct KvSparseApplyAdamHbm<GPUDevice, TKey, T> {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T beta1_power, T beta2_power, int64 task_size,
                  const GPUDevice& device) {
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyAdamGPU<T>,
        (task_size + block_size - 1) / block_size * embedding_dim, block_size,
        0, device.stream(), dev_var, dev_m, dev_v, grad_base, lr, beta1, beta2,
        epsilon, beta1_power, beta2_power, embedding_dim, task_size));
  }
};

template <typename TKey, typename T>
struct KvSparseApplyAdamWHbm<GPUDevice, TKey, T> {
  void operator()(int block_size, int embedding_dim, T** dev_var, T** dev_m,
                  T** dev_v, const T* grad_base, T lr, T beta1, T beta2,
                  T epsilon, T weight_decay, int64 task_size,
                  const GPUDevice& device) {
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyAdamWGPU<T>,
        (task_size + block_size - 1) / block_size * embedding_dim, block_size,
        0, device.stream(), dev_var, dev_m, dev_v, grad_base, lr, beta1, beta2,
        epsilon, weight_decay, embedding_dim, task_size));
  }
};

}  // namespace functor

#define REGISTER_ALL_TYPE(type)                                          \
  template struct functor::KvSparseApplyAdagrad<GPUDevice, type, int32>; \
  template struct functor::KvSparseApplyAdagrad<GPUDevice, type, int64>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                       \
  template struct functor::KvSparseApplyFtrl<GPUDevice, int32, type>; \
  template struct functor::KvSparseApplyFtrl<GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                              \
  template struct functor::KvSparseApplyAdam<GPUDevice, type, int32, int32>; \
  template struct functor::KvSparseApplyAdam<GPUDevice, type, int32, int64>; \
  template struct functor::KvSparseApplyAdam<GPUDevice, type, int64, int32>; \
  template struct functor::KvSparseApplyAdam<GPUDevice, type, int64, int64>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                           \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, type, int32, \
                                                  int32>;                 \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, type, int32, \
                                                  int64>;                 \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, type, int64, \
                                                  int32>;                 \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, type, int64, \
                                                  int64>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                    \
  template struct functor::KvSparseApplyAdamAsyncSparseRmspropHbm< \
      GPUDevice, int32, type>;                                     \
  template struct functor::KvSparseApplyAdamAsyncSparseRmspropHbm< \
      GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                               \
  template struct functor::KvSparseApplyAdamAsyncHbm<GPUDevice, int32, type>; \
  template struct functor::KvSparseApplyAdamAsyncHbm<GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                          \
  template struct functor::KvSparseApplyAdamHbm<GPUDevice, int32, type>; \
  template struct functor::KvSparseApplyAdamHbm<GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                             \
  template struct functor::KvSparseApplyAdagradHbm<GPUDevice, int32, type>; \
  template struct functor::KvSparseApplyAdagradHbm<GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

#define REGISTER_ALL_TYPE(type)                                           \
  template struct functor::KvSparseApplyAdamWHbm<GPUDevice, int32, type>; \
  template struct functor::KvSparseApplyAdamWHbm<GPUDevice, int64, type>;
TF_CALL_float(REGISTER_ALL_TYPE);
TF_CALL_double(REGISTER_ALL_TYPE);
#undef REGISTER_ALL_TYPE

}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
