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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA
#include <algorithm>

#include "deepray/custom_ops/embedding_variable/cc/kernels/kv_variable_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/util/work_sharder.h"
#include "training_ali_op_helpers.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "training_ali_ops_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using SYCLDevice = Eigen::SyclDevice;

template <typename T, typename Tindex, bool indices_as_pointer, bool has_counts>
class KvSparseApplyAdagradOp : public OpKernel {
 public:
  explicit KvSparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1});

    EmbeddingVar<Tindex, T>* var = NULL;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);
    EmbeddingVar<Tindex, T>* accum = NULL;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum));
    core::ScopedUnref unref_accum(accum);

    const Tensor& lr = ctx->input(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& global_step = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    int64* indices_counts = nullptr;
    std::function<int64(int64*, int64)> get_count_fn = 0;
    if (has_counts) {
      const Tensor& counts_tensor = ctx->input(6);
      indices_counts = (int64*)counts_tensor.data();
      get_count_fn = [](int64* counts, int64 index) { return counts[index]; };
    } else {
      get_count_fn = [](int64* counts, int64 index) { return 1; };
    }

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_vec = indices.vec<Tindex>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        int64 gs = global_step.scalar<int64>()();
        auto do_work = [this, ctx, &indices_vec, var, accum, &grad_flat, &gs,
                        &lr_scalar, indices_counts,
                        get_count_fn](int64 start_i, int64 limit_i) {
          for (int64 i = start_i; i < limit_i; i++) {
            const Tindex index = indices_vec(i);
            void* value_ptr = nullptr;
            bool is_filter = false;
            int64 count = get_count_fn(indices_counts, i);
            OP_REQUIRES_OK(ctx,
                           var->LookupOrCreateKey(index, &value_ptr, &is_filter,
                                                  indices_as_pointer, count));
            var->UpdateVersion(value_ptr, gs);
            if (is_filter) {
              auto var_i = var->flat(value_ptr);
              auto a = accum->flat(value_ptr);
              auto g = grad_flat.template chip<0>(i);
              a += g.square();
              var_i -= g.constant(lr_scalar) * g * a.rsqrt();
            }
          }
        };
        const int64 cost = 1000;  // very unreliable estimate for cost per step.
        auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
        Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
              do_work);

        if (has_counts && !indices_as_pointer) {
          const Tensor& indices_counts = ctx->input(6);
          var->UpdateCache(indices, indices_counts);
        }
      }
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                         \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdagrad")                \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          KvSparseApplyAdagradOp<T, Tindices, false, false>); \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceSparseApplyAdagrad")           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          KvSparseApplyAdagradOp<T, Tindices, true, false>);  \
  REGISTER_KERNEL_BUILDER(Name("KvResourceSparseApplyAdagradWithCounts")      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          KvSparseApplyAdagradOp<T, Tindices, false, true>);  \
  REGISTER_KERNEL_BUILDER(Name("_OPT_KvResourceSparseApplyAdagradWithCounts") \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          KvSparseApplyAdagradOp<T, Tindices, true, true>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_float(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename Device, typename T, typename Tindex, bool indices_as_pointer,
          bool has_counts>
class KvSparseApplyAdagradGPUOp : public OpKernel {
 public:
  explicit KvSparseApplyAdagradGPUOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));

    int num_worker_threads =
        ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
    thread_copy_id_alloc_.reset(
        new IntraThreadCopyIdAllocator(num_worker_threads));
  }

  void ApplyGradients(EmbeddingVar<Tindex, T>* var,
                      EmbeddingVar<Tindex, T>* accum, T** var_ptr, T** acc_ptr,
                      T lr_scalar, const T* grad_base, const int64 task_size,
                      se::Stream* stream, EventMgr* event_mgr,
                      const Eigen::GpuDevice& gpu_device) {
    // Send pointers of embeddings to GPU
    T** dev_var_ptr = (T**)var->GetBuffer(task_size * 2);
    T** dev_acc_ptr = dev_var_ptr + task_size;
    CHECK(dev_var_ptr);
    CHECK(dev_acc_ptr);
    se::DeviceMemoryBase dst_ptr(dev_var_ptr, sizeof(T*) * task_size * 2);
    stream->ThenMemcpy(&dst_ptr, var_ptr, sizeof(T*) * task_size * 2);

    int block_size = 128;
    int embedding_dim = var->ValueLen();
    functor::KvSparseApplyAdagradHbm<GPUDevice, Tindex, T>()(
        block_size, embedding_dim, dev_acc_ptr, dev_var_ptr, grad_base,
        lr_scalar, task_size, gpu_device);
    SyncWithEventMgr(stream, event_mgr);
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1});

    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);
    EmbeddingVar<Tindex, T>* accum = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &accum));
    core::ScopedUnref unref_accum(accum);

    const Tensor& lr = ctx->input(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    const Tensor& global_step = ctx->input(5);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    TensorShape var_shape({var->ValueLen()});
    for (int d = 0; d < var_shape.dims(); d++) {
      OP_REQUIRES(ctx, var_shape.dim_size(d) == grad.dim_size(d + 1),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d + 1)));
      inner_dim *= grad.dim_size(d + 1);
    }
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      if (inner_dim > 0) {
        auto indices_flat = indices.flat<Tindex>();
        auto grad_flat = grad.flat_outer_dims<T>();
        int64 gs = global_step.scalar<int64>()();
        T lr_scalar = lr.scalar<T>()();
        if (var->IsSingleHbm()) {
          const Tindex* key_base = &indices_flat(0);
          const T* grad_base = &grad_flat(0);
          const Device& device = ctx->eigen_device<Device>();

          functor::KvSparseApplyAdagrad<Device, T, Tindex>()(
              N, ctx->get_allocator(AllocatorAttributes()), var, accum,
              key_base, grad_base, lr_scalar, gs, device);
        } else {
          Tensor indices_temp_host(indices.dtype(), indices.shape());
          const Tensor* indices_host_ptr = nullptr;
          // Copy ids from GPU to CPU for CPU Lookup.
          auto stream = ctx->op_device_context()->stream();
          auto event_mgr =
              ctx->device()->tensorflow_accelerator_device_info()->event_mgr;
          if (!indices_as_pointer) {
            indices_host_ptr = &indices_temp_host;
            se::DeviceMemoryBase gpu_src(const_cast<Tindex*>(&indices_flat(0)),
                                         N * sizeof(Tindex));
            stream->ThenMemcpy(indices_host_ptr->data(), gpu_src,
                               N * sizeof(Tindex));
            SyncWithEventMgr(stream, event_mgr);
          } else {
            indices_host_ptr = &indices;
          }

          int counts_index = has_counts ? 6 : -1;
          T** var_ptr = new T*[N * 2];
          T** acc_ptr = var_ptr + N;
          std::vector<std::pair<EmbeddingVar<Tindex, T>*, T**>> vars(2);
          vars[0] = std::pair<EmbeddingVar<Tindex, T>*, T**>(var, var_ptr);
          vars[1] = std::pair<EmbeddingVar<Tindex, T>*, T**>(accum, acc_ptr);
          GetEmbeddingPointers(ctx, vars, (Tindex*)indices_host_ptr->data(), gs,
                               indices_as_pointer, counts_index, N,
                               thread_copy_id_alloc_.get());

          ApplyGradients(var, accum, var_ptr, acc_ptr, lr_scalar, &grad_flat(0),
                         N, stream, event_mgr, ctx->eigen_device<GPUDevice>());

          if (has_counts && !indices_as_pointer) {
            const Tensor& counts_tensor = ctx->input(counts_index);
            var->UpdateCache(*indices_host_ptr, counts_tensor);
          }

          delete[] var_ptr;
        }
      }
    }
  }

 private:
  bool use_exclusive_lock_;
  std::unique_ptr<IntraThreadCopyIdAllocator> thread_copy_id_alloc_;
};

namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                          \
  template <>                                                                \
  void KvSparseApplyAdagrad<GPUDevice, T, Tindex>::operator()(               \
      int32 num_items, Allocator* alloc, EmbeddingVar<Tindex, T>* var,       \
      EmbeddingVar<Tindex, T>* accum, const Tindex* key_base, const T* grad, \
      T lr, int64 gs, const GPUDevice& device);                              \
  extern template struct KvSparseApplyAdagrad<GPUDevice, T, Tindex>;
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#define REGISTER_KERNELS(T, Tindices)                                   \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("KvResourceSparseApplyAdagrad")                              \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("lr")                                             \
          .HostMemory("global_step")                                    \
          .TypeConstraint<Tindices>("Tindices"),                        \
      KvSparseApplyAdagradGPUOp<GPUDevice, T, Tindices, false, false>); \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OPT_KvResourceSparseApplyAdagrad")                         \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("indices")                                        \
          .HostMemory("lr")                                             \
          .HostMemory("global_step")                                    \
          .TypeConstraint<Tindices>("Tindices"),                        \
      KvSparseApplyAdagradGPUOp<GPUDevice, T, Tindices, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("KvResourceSparseApplyAdagradWithCounts")                    \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("lr")                                             \
          .HostMemory("global_step")                                    \
          .HostMemory("indices_counts")                                 \
          .TypeConstraint<Tindices>("Tindices"),                        \
      KvSparseApplyAdagradGPUOp<GPUDevice, T, Tindices, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_OPT_KvResourceSparseApplyAdagradWithCounts")               \
          .Device(DEVICE_GPU)                                           \
          .TypeConstraint<T>("T")                                       \
          .HostMemory("indices")                                        \
          .HostMemory("lr")                                             \
          .HostMemory("global_step")                                    \
          .HostMemory("indices_counts")                                 \
          .TypeConstraint<Tindices>("Tindices"),                        \
      KvSparseApplyAdagradGPUOp<GPUDevice, T, Tindices, true, true>);
#define REGISTER_GPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_float(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
#endif  // End of GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
