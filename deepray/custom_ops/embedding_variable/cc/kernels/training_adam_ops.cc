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

#include "deepray/custom_ops/embedding_variable/cc/embedding/intra_thread_copy_id_allocator.h"
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

template <typename Device, typename T, typename Tindex, typename Tstep,
          bool indices_as_pointer, bool has_counts>
class KvSparseApplyAdamOp : public OpKernel {
 public:
  explicit KvSparseApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1, 2});
    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);
    const Tensor& global_step = ctx->input(11);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

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

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    int64* indices_counts = nullptr;
    std::function<int64(int64*, int64)> get_count_fn = 0;
    if (has_counts) {
      const Tensor& counts_tensor = ctx->input(12);
      indices_counts = (int64*)counts_tensor.data();
      get_count_fn = [](int64* counts, int64 index) { return counts[index]; };
    } else {
      get_count_fn = [](int64* counts, int64 index) { return 1; };
    }
    if (N > 0) {
      T beta1_power_scalar = beta1_power.scalar<T>()();
      T beta2_power_scalar = beta2_power.scalar<T>()();
      T lr_scalar = lr.scalar<T>()();
      T beta1_scalar = beta1.scalar<T>()();
      T beta2_scalar = beta2.scalar<T>()();
      T epsilon_scalar = epsilon.scalar<T>()();
      const T alpha =
          lr_scalar *
          Eigen::numext::sqrt(static_cast<T>(1) - beta2_power_scalar) /
          (static_cast<T>(1) - beta1_power_scalar);

      auto do_work = [this, ctx, inner_dim, &var, &m, &v, &grad, &indices,
                      &lr_scalar, &beta1_scalar, &beta1_power, &beta2_power,
                      &beta2_scalar, &epsilon_scalar, &alpha, &global_step,
                      get_count_fn,
                      indices_counts](int64 start_i, int64 limit_i) {
        if (inner_dim > 0) {
          auto grad_flat = grad.flat_outer_dims<T>();
          auto indices_vec = indices.vec<Tindex>();
          Tstep gs = global_step.scalar<Tstep>()();

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
              auto m_a = m->flat(value_ptr);
              auto v_a = v->flat(value_ptr);
              auto g = grad_flat.template chip<0>(i);
              auto var_i = var->flat(value_ptr);

              m_a = m_a * beta1_scalar + g * (static_cast<T>(1) - beta1_scalar);
              v_a = v_a * beta2_scalar +
                    g.square() * (static_cast<T>(1) - beta2_scalar);
              var_i -= (m_a * alpha) / (v_a.sqrt() + epsilon_scalar);
            }
          }
        }
      };

      const int64 cost = 1000;
      auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
      Shard(worker_threads.num_threads, worker_threads.workers, N, cost,
            do_work);
      if (has_counts && !indices_as_pointer) {
        const Tensor& indices_counts = ctx->input(12);
        var->UpdateCache(indices, indices_counts);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T, Tindices, Tstep)                          \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("KvResourceSparseApplyAdam")                                  \
          .Device(DEVICE_##D)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Tindices>("Tindices")                          \
          .TypeConstraint<Tstep>("Tstep"),                               \
      KvSparseApplyAdamOp<D##Device, T, Tindices, Tstep, false, false>); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OPT_KvResourceSparseApplyAdam")                             \
          .Device(DEVICE_##D)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Tindices>("Tindices")                          \
          .TypeConstraint<Tstep>("Tstep"),                               \
      KvSparseApplyAdamOp<D##Device, T, Tindices, Tstep, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("KvResourceSparseApplyAdamWithCounts")                        \
          .Device(DEVICE_##D)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Tindices>("Tindices")                          \
          .TypeConstraint<Tstep>("Tstep"),                               \
      KvSparseApplyAdamOp<D##Device, T, Tindices, Tstep, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_OPT_KvResourceSparseApplyAdamWithCounts")                   \
          .Device(DEVICE_##D)                                            \
          .TypeConstraint<T>("T")                                        \
          .TypeConstraint<Tindices>("Tindices")                          \
          .TypeConstraint<Tstep>("Tstep"),                               \
      KvSparseApplyAdamOp<D##Device, T, Tindices, Tstep, true, true>);

#define REGISTER_CPU_KERNELS(T)           \
  REGISTER_KERNELS(CPU, T, int32, int32); \
  REGISTER_KERNELS(CPU, T, int64, int32); \
  REGISTER_KERNELS(CPU, T, int32, int64); \
  REGISTER_KERNELS(CPU, T, int64, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
template <typename Device, typename T, typename Tindex, typename Tstep,
          bool indices_as_pointer, bool has_counts>
class KvSparseApplyAdamGPUOp : public OpKernel {
 public:
  explicit KvSparseApplyAdamGPUOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));

    int num_worker_threads =
        ctx->device()->tensorflow_cpu_worker_threads()->num_threads;
    thread_copy_id_alloc_.reset(
        new IntraThreadCopyIdAllocator(num_worker_threads));
  }

  void ApplyGradients(EmbeddingVar<Tindex, T>* var, EmbeddingVar<Tindex, T>* m,
                      EmbeddingVar<Tindex, T>* v, T** var_ptr, T** m_ptr,
                      T** v_ptr, T beta1, T beta2, T epsilon, T lr,
                      T beta1_power, T beta2_power, const T* grad_base,
                      const int64 task_size, se::Stream* stream,
                      EventMgr* event_mgr, const Eigen::GpuDevice& gpu_device) {
    // Send pointers of embeddings to GPU
    T** dev_var_ptr = (T**)var->GetBuffer(task_size * 3);
    T** dev_m_ptr = dev_var_ptr + task_size;
    T** dev_v_ptr = dev_m_ptr + task_size;
    CHECK(dev_var_ptr);
    CHECK(dev_m_ptr);
    CHECK(dev_v_ptr);

    se::DeviceMemoryBase dst_ptr(dev_var_ptr, sizeof(T*) * task_size * 3);
    stream->ThenMemcpy(&dst_ptr, var_ptr, sizeof(T*) * task_size * 3);

    int block_size = 128;
    int embedding_dim = var->ValueLen();
    functor::KvSparseApplyAdamHbm<GPUDevice, Tindex, T>()(
        block_size, embedding_dim, dev_var_ptr, dev_m_ptr, dev_v_ptr, grad_base,
        lr, beta1, beta2, epsilon, beta1_power, beta2_power, task_size,
        gpu_device);
    SyncWithEventMgr(stream, event_mgr);
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    auto locks = MaybeLockEmbeddingVariableInputMutexesInOrder<Tindex, T>(
        ctx, use_exclusive_lock_, {0, 1, 2});
    EmbeddingVar<Tindex, T>* var = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 0, &var));
    core::ScopedUnref unref_var(var);

    EmbeddingVar<Tindex, T>* m = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 1, &m));
    core::ScopedUnref unref_m(m);

    EmbeddingVar<Tindex, T>* v = nullptr;
    OP_REQUIRES_OK(ctx, GetInputEmbeddingVar(ctx, 2, &v));
    core::ScopedUnref unref_v(v);

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);
    const Tensor& indices = ctx->input(10);
    const Tensor& global_step = ctx->input(11);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

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

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const int64 N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      if (var->IsSingleHbm()) {
        const Device& device = ctx->eigen_device<Device>();
        OP_REQUIRES_OK(
            ctx, functor::KvSparseApplyAdam<Device, T, Tindex, Tstep>()(
                     device, var, m, v, beta1_power.scalar<T>(),
                     beta2_power.scalar<T>(), indices.vec<Tindex>(),
                     grad.flat_outer_dims<T>(), lr.scalar<T>(),
                     beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
                     global_step.scalar<Tstep>(), inner_dim,
                     ctx->get_allocator(AllocatorAttributes())));
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto grad_flat = grad.flat_outer_dims<T>();
        Tstep gs = global_step.scalar<int64>()();
        const T lr_scalar = lr.scalar<T>()();
        const T beta1_scalar = beta1.scalar<T>()();
        const T beta2_scalar = beta2.scalar<T>()();
        const T epsilon_scalar = epsilon.scalar<T>()();
        const T beta1_power_scalar = beta1_power.scalar<T>()();
        const T beta2_power_scalar = beta2_power.scalar<T>()();

        Tensor indices_temp_host(indices.dtype(), indices.shape());
        const Tensor* indices_host_ptr = nullptr;
        // Copy ids from GPU to CPU for CPU Lookup.
        auto stream = ctx->op_device_context()->stream();
        auto event_mgr =
            ctx->device()->tensorflow_accelerator_device_info()->event_mgr;
        if (!indices_as_pointer) {
          indices_host_ptr = &indices_temp_host;
          se::DeviceMemoryBase gpu_src(const_cast<Tindex*>(&indices_vec(0)),
                                       N * sizeof(Tindex));
          stream->ThenMemcpy(indices_host_ptr->data(), gpu_src,
                             N * sizeof(Tindex));
          SyncWithEventMgr(stream, event_mgr);
        } else {
          indices_host_ptr = &indices;
        }

        int counts_index = has_counts ? 12 : -1;
        T** var_ptr = new T*[N * 3];
        T** m_ptr = var_ptr + N;
        T** v_ptr = m_ptr + N;
        std::vector<std::pair<EmbeddingVar<Tindex, T>*, T**>> vars(3);
        vars[0] = std::pair<EmbeddingVar<Tindex, T>*, T**>(var, var_ptr);
        vars[1] = std::pair<EmbeddingVar<Tindex, T>*, T**>(m, m_ptr);
        vars[2] = std::pair<EmbeddingVar<Tindex, T>*, T**>(v, v_ptr);
        GetEmbeddingPointers(ctx, vars, (Tindex*)indices_host_ptr->data(), gs,
                             indices_as_pointer, counts_index, N,
                             thread_copy_id_alloc_.get());

        ApplyGradients(var, m, v, var_ptr, m_ptr, v_ptr, beta1_scalar,
                       beta2_scalar, epsilon_scalar, lr_scalar,
                       beta1_power_scalar, beta2_power_scalar, &grad_flat(0), N,
                       stream, event_mgr, ctx->eigen_device<GPUDevice>());

        if (has_counts && !indices_as_pointer) {
          const Tensor& counts_tensor = ctx->input(counts_index);
          var->UpdateCache(*indices_host_ptr, counts_tensor);
        }

        delete[] var_ptr;
      }
    }
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  std::unique_ptr<IntraThreadCopyIdAllocator> thread_copy_id_alloc_;
};

#define REGISTER_KERNELS(D, T, Tindices, Tstep)                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("KvResourceSparseApplyAdam")                                     \
          .Device(DEVICE_##D)                                               \
          .HostMemory("lr")                                                 \
          .HostMemory("beta1_power")                                        \
          .HostMemory("beta2_power")                                        \
          .HostMemory("beta1")                                              \
          .HostMemory("beta2")                                              \
          .HostMemory("epsilon")                                            \
          .HostMemory("global_step")                                        \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<Tindices>("Tindices")                             \
          .TypeConstraint<Tstep>("Tstep"),                                  \
      KvSparseApplyAdamGPUOp<D##Device, T, Tindices, Tstep, false, false>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OPT_KvResourceSparseApplyAdam")                                \
          .Device(DEVICE_##D)                                               \
          .HostMemory("indices")                                            \
          .HostMemory("lr")                                                 \
          .HostMemory("beta1_power")                                        \
          .HostMemory("beta2_power")                                        \
          .HostMemory("beta1")                                              \
          .HostMemory("beta2")                                              \
          .HostMemory("epsilon")                                            \
          .HostMemory("global_step")                                        \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<Tindices>("Tindices")                             \
          .TypeConstraint<Tstep>("Tstep"),                                  \
      KvSparseApplyAdamGPUOp<D##Device, T, Tindices, Tstep, true, false>);  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("KvResourceSparseApplyAdamWithCounts")                           \
          .Device(DEVICE_##D)                                               \
          .HostMemory("lr")                                                 \
          .HostMemory("beta1_power")                                        \
          .HostMemory("beta2_power")                                        \
          .HostMemory("beta1")                                              \
          .HostMemory("beta2")                                              \
          .HostMemory("epsilon")                                            \
          .HostMemory("global_step")                                        \
          .HostMemory("indices_counts")                                     \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<Tindices>("Tindices")                             \
          .TypeConstraint<Tstep>("Tstep"),                                  \
      KvSparseApplyAdamGPUOp<D##Device, T, Tindices, Tstep, false, true>);  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_OPT_KvResourceSparseApplyAdamWithCounts")                      \
          .Device(DEVICE_##D)                                               \
          .HostMemory("indices")                                            \
          .HostMemory("lr")                                                 \
          .HostMemory("beta1_power")                                        \
          .HostMemory("beta2_power")                                        \
          .HostMemory("beta1")                                              \
          .HostMemory("beta2")                                              \
          .HostMemory("epsilon")                                            \
          .HostMemory("global_step")                                        \
          .HostMemory("indices_counts")                                     \
          .TypeConstraint<T>("T")                                           \
          .TypeConstraint<Tindices>("Tindices")                             \
          .TypeConstraint<Tstep>("Tstep"),                                  \
      KvSparseApplyAdamGPUOp<D##Device, T, Tindices, Tstep, true, true>);
#define REGISTER_GPU_KERNELS(T)           \
  REGISTER_KERNELS(GPU, T, int32, int32); \
  REGISTER_KERNELS(GPU, T, int64, int32); \
  REGISTER_KERNELS(GPU, T, int32, int64); \
  REGISTER_KERNELS(GPU, T, int64, int64);

TF_CALL_float(REGISTER_GPU_KERNELS);
TF_CALL_double(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS
#undef REGISTER_KERNELS
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex, Tstep)                           \
  template <>                                                        \
  Status KvSparseApplyAdam<GPUDevice, T, Tindex, Tstep>::operator()( \
      const GPUDevice& d, EmbeddingVar<Tindex, T>* var,              \
      EmbeddingVar<Tindex, T>* m, EmbeddingVar<Tindex, T>* v,        \
      typename TTypes<T>::ConstScalar beta1_power_scalar,            \
      typename TTypes<T>::ConstScalar beta2_power_scalar,            \
      typename TTypes<Tindex>::ConstVec indices_vec,                 \
      typename TTypes<T>::ConstMatrix grad,                          \
      typename TTypes<T>::ConstScalar lr_scalar,                     \
      typename TTypes<T>::ConstScalar beta1_scalar,                  \
      typename TTypes<T>::ConstScalar beta2_scalar,                  \
      typename TTypes<T>::ConstScalar epsilon_scalar,                \
      typename TTypes<Tstep>::ConstScalar global_step_scalar,        \
      const int64 inner_dim, Allocator* alloc);                      \
  extern template struct KvSparseApplyAdam<GPUDevice, T, Tindex, Tstep>;

#define DECLARE_GPU_SPEC_TYPE(T)     \
  DECLARE_GPU_SPEC(T, int32, int32); \
  DECLARE_GPU_SPEC(T, int32, int64); \
  DECLARE_GPU_SPEC(T, int64, int32); \
  DECLARE_GPU_SPEC(T, int64, int64);

DECLARE_GPU_SPEC_TYPE(float);
DECLARE_GPU_SPEC_TYPE(double);

#undef DECLARE_GPU_SPEC_TYPE
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#endif  // End of GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
