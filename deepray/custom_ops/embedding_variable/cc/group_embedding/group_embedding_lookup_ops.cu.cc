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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "group_embedding_lookup_sparse_forward_base_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

#define USING_BASE_CLASS_MEMBER                           \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_num_lookup; \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_dimension;  \
  using GroupLookupBaseCpuOp<TKey, TValue>::m_is_use_default_value_tensor;

template <typename TKey, typename TValue>
class GroupEmbeddingVariableLookupDenseGpuOp
    : public GroupLookupBaseCpuOp<TKey, TValue> {
  USING_BASE_CLASS_MEMBER
 public:
  explicit GroupEmbeddingVariableLookupDenseGpuOp(OpKernelConstruction* c)
      : GroupLookupBaseCpuOp<TKey, TValue>(c) {
    OP_REQUIRES_OK(c, c->GetAttr("is_use_default_value_tensor",
                                 &m_is_use_default_value_tensor));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

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

      EmbeddingVarContext<GPUDevice> ev_ctx(ctx);
      if (m_is_use_default_value_tensor) {
        embedding_var->GetEmbeddings(
            ev_ctx, dense_values, gather_embedding, nnz,
            reinterpret_cast<TValue*>(ctx->input(m_num_lookup * 4 + 1).data()),
            stream);
      } else {
        embedding_var->GetEmbeddings(ev_ctx, dense_values, gather_embedding,
                                     nnz, nullptr, stream);
        embedding_var->UpdateCache(dense_values_tensor, true, stream);
      }
    }
  }
};

#define REGISTER_GPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVarLookupDense")         \
          .Device(DEVICE_GPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVariableLookupDenseGpuOp<key_type, value_type>)

REGISTER_GPU_KERNELS(int32, float);
REGISTER_GPU_KERNELS(int64, float);
#undef REGISTER_GPU_KERNELS

#undef USING_BASE_CLASS_MEMBER
}  // namespace tensorflow

#endif  // GOOGLE_CUDA