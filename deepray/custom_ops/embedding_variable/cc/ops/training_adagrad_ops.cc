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

static Status KvResourceApplyAdagradShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return OkStatus();
}

#define REGISTER_OP_BY_NAME(name)                                   \
  REGISTER_OP(name)                                                 \
      .Input("var: resource")                                       \
      .Input("accum: resource")                                     \
      .Input("lr: T")                                               \
      .Input("grad: T")                                             \
      .Input("indices: Tindices")                                   \
      .Input("global_step: Tstep")                                  \
      .Attr("T: numbertype")                                        \
      .Attr("Tindices: {int32, int64}")                             \
      .Attr("Tstep: {int32, int64}")                                \
      .Attr("use_locking: bool = false")                            \
      .SetShapeFn([](InferenceContext* c) {                         \
        return KvResourceApplyAdagradShapeFn(c, true /* sparse */); \
      })                                                            \
      .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagrad");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagrad");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)                                   \
  REGISTER_OP(name)                                                 \
      .Input("var: resource")                                       \
      .Input("accum: resource")                                     \
      .Input("lr: T")                                               \
      .Input("grad: T")                                             \
      .Input("indices: Tindices")                                   \
      .Input("global_step: Tstep")                                  \
      .Input("indices_counts: int64")                               \
      .Attr("T: numbertype")                                        \
      .Attr("Tindices: {int32, int64}")                             \
      .Attr("Tstep: {int32, int64}")                                \
      .Attr("use_locking: bool = false")                            \
      .SetShapeFn([](InferenceContext* c) {                         \
        return KvResourceApplyAdagradShapeFn(c, true /* sparse */); \
      })                                                            \
      .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagradWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagradWithCounts");
#undef REGISTER_OP_BY_NAME

}  // namespace tensorflow
