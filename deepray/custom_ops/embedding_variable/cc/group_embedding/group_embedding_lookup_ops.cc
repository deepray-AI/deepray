/* Copyright 2022 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#include "deepray/custom_ops/embedding_variable/cc/embedding/cache.h"
#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "group_embedding_lookup_sparse_forward_base_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

#define USING_BASE_CLASS_MEMBER                           \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_num_lookup; \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_dimension;  \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_is_use_default_value_tensor;

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupDenseCpuOp
    : public GroupLookupBaseCpuOp<TKey, TValue> {
  USING_BASE_CLASS_MEMBER
 public:
  explicit GroupEmbeddingVariableLookupDenseCpuOp(OpKernelConstruction* c)
      : GroupLookupBaseCpuOp<TKey, TValue>(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &m_is_use_default_value_tensor));
  }

  void Compute(OpKernelContext* ctx) override {
    /*
      step 1: unique and assign unique output and index
      step 2: doing parallel unique value gather
    */
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    for (int i = 0; i < m_num_lookup; ++i) {
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      OP_REQUIRES_OK(
          ctx, LookupResource(ctx, HandleFromInput(ctx, i), &embedding_var));
      core::ScopedUnref unref_me(embedding_var);

      const Tensor& dense_values_tensor = ctx->input(m_num_lookup + i);
      auto dense_values = dense_values_tensor.flat<TKey>().data();
      int nnz = dense_values_tensor.NumElements();

      auto dense_values_tensor_shape = dense_values_tensor.shape();
      TensorShape emb_vectors_tensor_shape =
          TensorShape(dense_values_tensor_shape);
      emb_vectors_tensor_shape.AddDim(m_dimension);
      Tensor* gather_embedding_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &gather_embedding_tensor));
      auto gather_embedding = gather_embedding_tensor->flat<TValue>().data();

      OP_REQUIRES(
          ctx,
          !embedding_var->IsMultiLevel() || (embedding_var->IsMultiLevel() &&
                                             embedding_var->CacheSize() >= nnz),
          errors::InvalidArgument("MultiLevel EV's Cache size ",
                                  embedding_var->CacheSize(),
                                  " should large than IDs in batch ", nnz));

      EmbeddingVarContext<CPUDevice> ev_ctx(ctx);
      if (m_is_use_default_value_tensor) {
        embedding_var->GetEmbeddings(
            ev_ctx, dense_values, gather_embedding, nnz,
            reinterpret_cast<TValue*>(ctx->input(m_num_lookup * 4 + 1).data()));
      } else {
        embedding_var->GetEmbeddings(ev_ctx, dense_values, gather_embedding,
                                     nnz);
        embedding_var->UpdateCache(dense_values_tensor, true);
      }
    }
  }
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVarLookupDense")         \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVariableLookupDenseCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class GroupVariableLookupDenseCpuOp
    : public GroupLookupBaseCpuOp<TKey, TValue> {
  USING_BASE_CLASS_MEMBER
 public:
  explicit GroupVariableLookupDenseCpuOp(OpKernelConstruction* c)
      : GroupLookupBaseCpuOp<TKey, TValue>(c) {}

  void Compute(OpKernelContext* ctx) override {
    /*
      step 1: unique and assign unique output and index
      step 2: doing parallel unique value gather
    */
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    for (int i = 0; i < m_num_lookup; ++i) {
      const Tensor& emb_variable_tensor = ctx->input(i);
      auto embedding_variable = emb_variable_tensor.flat<TValue>().data();

      const Tensor& dense_values_tensor = ctx->input(m_num_lookup + i);

      int nnz = dense_values_tensor.NumElements();

      auto dense_values_tensor_shape = dense_values_tensor.shape();
      TensorShape emb_vectors_tensor_shape =
          TensorShape(dense_values_tensor_shape);
      emb_vectors_tensor_shape.AddDim(m_dimension);
      Tensor* gather_embedding_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, emb_vectors_tensor_shape,
                                               &gather_embedding_tensor));
      auto gather_embedding = gather_embedding_tensor->flat<TValue>().data();

      // Stage 1
      Tensor unique_idx_tensor;
      Tensor unique_tensor;
      Tensor unique_counter;

      UniqueWithoutAxis<TKey, int32>(
          ctx, dense_values_tensor, &unique_idx_tensor, &unique_tensor,
          &unique_counter, 0, this->partition_size_, this->serial_,
          this->unique_ratio_hint_, this->map_flag_);

      ctx->set_output(m_num_lookup + i, unique_tensor);
      ctx->set_output(2 * m_num_lookup + i, unique_idx_tensor);
      auto* unique = unique_tensor.flat<TKey>().data();
      auto* unique_idx = unique_idx_tensor.flat<int>().data();
      int slice_bytes = nnz * m_dimension * 1000;
      auto do_lookup = [this, ctx, embedding_variable, unique, unique_idx,
                        gather_embedding](int64 start, int64 end) {
        for (int k = start; k < end; ++k) {
          auto indices = unique_idx[k];
          TKey unique_id = unique[indices];
          memcpy(gather_embedding + k * m_dimension,
                 embedding_variable + unique_id * m_dimension,
                 sizeof(float) * m_dimension);
        }
      };
      Shard(worker_threads->num_threads, worker_threads->workers, nnz,
            slice_bytes, do_lookup);
    }
  }
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookupDense")          \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          GroupVariableLookupDenseCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

#undef USING_BASE_CLASS_MEMBER
}  // namespace tensorflow