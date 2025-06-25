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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

static Status HandleKvGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                           int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return OkStatus();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(c->Subshape(grad, 1, &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return OkStatus();
}

static Status KvApplyAdamAsyncShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return OkStatus();
}

#define REGISTER_OP_BY_NAME(name)                             \
  REGISTER_OP(name)                                           \
      .Input("var: resource")                                 \
      .Input("m: resource")                                   \
      .Input("v: resource")                                   \
      .Input("beta1_power: resource")                         \
      .Input("beta2_power: resource")                         \
      .Input("lr: T")                                         \
      .Input("beta1: T")                                      \
      .Input("beta2: T")                                      \
      .Input("epsilon: T")                                    \
      .Input("grad: T")                                       \
      .Input("indices: Tindices")                             \
      .Input("global_step: Tstep")                            \
      .Attr("T: numbertype")                                  \
      .Attr("Tindices: {int32, int64}")                       \
      .Attr("Tstep: {int32, int64}")                          \
      .Attr("use_locking: bool = false")                      \
      .Attr("apply_sparse_rmsprop: bool = false")             \
      .Attr("indices_as_pointer: bool = false")               \
      .SetShapeFn([](InferenceContext* c) {                   \
        return KvApplyAdamAsyncShapeFn(c, true /* sparse */); \
      })
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamAsync");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamAsync");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)                             \
  REGISTER_OP(name)                                           \
      .Input("var: resource")                                 \
      .Input("m: resource")                                   \
      .Input("v: resource")                                   \
      .Input("beta1_power: resource")                         \
      .Input("beta2_power: resource")                         \
      .Input("lr: T")                                         \
      .Input("beta1: T")                                      \
      .Input("beta2: T")                                      \
      .Input("epsilon: T")                                    \
      .Input("grad: T")                                       \
      .Input("indices: Tindices")                             \
      .Input("global_step: Tstep")                            \
      .Input("indices_counts: int64")                         \
      .Attr("T: numbertype")                                  \
      .Attr("Tindices: {int32, int64}")                       \
      .Attr("Tstep: {int32, int64}")                          \
      .Attr("use_locking: bool = false")                      \
      .Attr("apply_sparse_rmsprop: bool = false")             \
      .Attr("indices_as_pointer: bool = false")               \
      .SetShapeFn([](InferenceContext* c) {                   \
        return KvApplyAdamAsyncShapeFn(c, true /* sparse */); \
      })
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamAsyncWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamAsyncWithCounts");
#undef REGISTER_OP_BY_NAME

}  // namespace tensorflow
