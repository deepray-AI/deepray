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

static Status KvApplyGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  ShapeHandle grad = ShapeOrHandleShape(c, 2);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));
  return OkStatus();
}

#define REGISTER_OP_BY_NAME(name)               \
  REGISTER_OP(name)                             \
      .Input("var: resource")                   \
      .Input("alpha: T")                        \
      .Input("grad: T")                         \
      .Input("indices: Tindices")               \
      .Input("global_step: Tstep")              \
      .Attr("T: numbertype")                    \
      .Attr("Tindices: {int32, int64}")         \
      .Attr("Tstep: {int32, int64}")            \
      .Attr("use_locking: bool = false")        \
      .Attr("indices_as_pointer: bool = false") \
      .SetShapeFn(KvApplyGradientDescentShapeFn)
REGISTER_OP_BY_NAME("KvResourceSparseApplyGradientDescent");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyGradientDescent");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)               \
  REGISTER_OP(name)                             \
      .Input("var: resource")                   \
      .Input("alpha: T")                        \
      .Input("grad: T")                         \
      .Input("indices: Tindices")               \
      .Input("global_step: Tstep")              \
      .Input("counts: int64")                   \
      .Attr("T: numbertype")                    \
      .Attr("Tindices: {int32, int64}")         \
      .Attr("Tstep: {int32, int64}")            \
      .Attr("use_locking: bool = false")        \
      .Attr("indices_as_pointer: bool = false") \
      .SetShapeFn(KvApplyGradientDescentShapeFn)
REGISTER_OP_BY_NAME("KvResourceSparseApplyGradientDescentWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyGradientDescentWithCounts");
#undef REGISTER_OP_BY_NAME

}  // namespace tensorflow
