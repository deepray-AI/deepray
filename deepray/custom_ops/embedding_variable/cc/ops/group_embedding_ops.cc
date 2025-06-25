// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

REGISTER_OP("GroupEmbeddingVarLookupDense")
    .Input("resource: num_lookups * resource")
    .Input("dense_values: num_lookups * Tkeys")
    .Input("default_value: dtype")
    .Attr("is_use_default_value_tensor: bool = false")
    .Attr("dimension: int")
    .Output("output: num_lookups * dtype")
    .Output("unique_keys: num_lookups * Tkeys")
    .Output("unique_idx: num_lookups * int32")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .Attr("num_lookups: int >= 1")
    .Attr("is_inference: bool = false")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'} = 'mean'")  // placeholder
    .Attr("ignore_weights: bool = true")                  // placeholder
    .Attr("is_sequence: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      int num_lookups;
      TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));
      const std::vector<shape_inference::ShapeAndType>* shapes_and_types =
          nullptr;
      for (int i = 0; i < num_lookups; ++i) {
        shapes_and_types = c->input_handle_shapes_and_types(i);
        // LOG(INFO) << "shapes_and_types: shape="
        //           << c->DebugString(shapes_and_types->at(0).shape);

        ShapeHandle temp;
        TF_RETURN_IF_ERROR(
            c->WithRankAtLeast(c->input(num_lookups + i), 1, &temp));

        ShapeHandle unused;
        TF_RETURN_IF_ERROR(
            c->WithRankAtLeast(shapes_and_types->at(0).shape, 1, &unused));
        ShapeHandle params_subshape;
        params_subshape = shapes_and_types->at(0).shape;

        ShapeHandle indices_shape = c->input(num_lookups + i);
        ShapeHandle out;
        TF_RETURN_IF_ERROR(
            c->Concatenate(indices_shape, params_subshape, &out));
        c->set_output(i, out);
        c->set_output(num_lookups + i,
                      c->Vector(InferenceContext::kUnknownDim));
        // c->set_output(num_lookups * 2 + i, c->input(num_lookups+i));
      }

      return OkStatus();
    });

REGISTER_OP("GroupEmbeddingVarLookup")
    .Input("resource: num_lookups * resource")
    .Input("sp_values: num_lookups * Tkeys")
    .Input("sp_indices: num_lookups * int64")
    .Input("sp_weights: num_lookups * dtype")
    .Input("dense_shape: num_lookups * int64")
    .Input("default_value: dtype")
    .Attr("ignore_weights: bool = false")
    .Attr("is_use_default_value_tensor: bool = false")
    .Attr("is_sequence: bool = false")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("dimension: int")
    .Output("output: num_lookups * dtype")
    .Output("unique_keys: num_lookups * Tkeys")
    .Output("unique_idx: num_lookups * int32")
    .Output("batch_nums: num_lookups * int32")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .Attr("num_lookups: int >= 1")
    .Attr("is_inference: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      int num_lookups;
      TF_RETURN_IF_ERROR(c->GetAttr("num_lookups", &num_lookups));

      for (int i = 0; i < num_lookups; ++i) {
        auto shapes_and_types = c->input_handle_shapes_and_types(i);
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(
            c->WithRankAtLeast(shapes_and_types->at(0).shape, 1, &unused));
        TF_RETURN_IF_ERROR(
            c->WithRank(c->input(num_lookups * 2 + i), 2, &unused));
        // TF_RETURN_IF_ERROR(c->WithRank(c->input(num_lookups*3+i), 1,
        // &unused));
        ShapeHandle params_subshape;
        params_subshape = shapes_and_types->at(0).shape;

        ShapeHandle indices_shape = c->input(num_lookups + i);
        ShapeHandle out;
        TF_RETURN_IF_ERROR(
            c->Concatenate(indices_shape, params_subshape, &out));
        c->set_output(i, out);
        c->set_output(num_lookups + i,
                      c->Vector(InferenceContext::kUnknownDim));
        c->set_output(num_lookups * 2 + i, c->input(num_lookups + i));
        c->set_output(num_lookups * 3 + i,
                      c->Vector(InferenceContext::kUnknownDim));
      }

      return OkStatus();
    });

REGISTER_OP("GroupEmbeddingVariableLookupGrad")
    .Input("grads: num_lookups * dtype")
    .Input("embedding_resources: num_lookups * resource")
    .Input("unique_keys: num_lookups * Tkeys")
    .Input("sp_indices: num_lookups * int64")
    .Input("batch_nums: num_lookups * int32")
    .Output("nnz_grads: num_lookups * dtype")
    .Attr("dimension: int")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("num_lookups: int >=1")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_lookups = ctx->num_outputs();
      for (int i = 0; i < num_lookups; ++i) {
        ShapeHandle top_grad_shape;
        TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(i), 2, &top_grad_shape));
        DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);
        ctx->set_output(i,
                        ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      }
      return OkStatus();
    });

REGISTER_OP("GroupVariableLookup")
    .Input("emb_variables: num_lookups * dtype")
    .Input("sp_values: num_lookups * Tkeys")
    .Input("sp_indices: num_lookups * int64")
    .Input("sp_weights: num_lookups * dtype")
    .Input("dense_shape: num_lookups * int64")
    .Input("default_value: dtype")
    .Output("output: num_lookups * dtype")
    .Output("unique_keys: num_lookups * Tkeys")
    .Output("unique_idx: num_lookups * int32")
    .Output("batch_nums: num_lookups * int32")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("dimension: int")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .Attr("num_lookups: int >= 1")
    .Attr("ignore_weights: bool = false")
    .Attr("is_use_default_value_tensor: bool = false")
    .Attr("is_sequence: bool = false")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_lookups;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_lookups", &num_lookups));

      bool is_sequence;
      TF_RETURN_IF_ERROR(ctx->GetAttr("is_sequence", &is_sequence));

      for (int i = 0; i < num_lookups; ++i) {
        ShapeHandle temp;
        TF_RETURN_IF_ERROR(
            ctx->WithRank(ctx->input(num_lookups + i), 1, &temp));
        TF_RETURN_IF_ERROR(
            ctx->WithRank(ctx->input(2 * num_lookups + i), 2, &temp));
        // TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3*num_lookups+i), 1,
        // &temp));
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(i), 1, &unused));
        ShapeHandle params_subshape;
        TF_RETURN_IF_ERROR(ctx->Subshape(ctx->input(i), 1, &params_subshape));
        DimensionHandle emb_vec_size_dim = ctx->Dim(params_subshape, 0);
        DimensionHandle batch_dim = ctx->UnknownDim();
        if (is_sequence) {
          ShapeHandle output_shape =
              ctx->MakeShape({batch_dim, batch_dim, emb_vec_size_dim});
          ctx->set_output(i, output_shape);
        } else {
          ShapeHandle output_shape =
              ctx->MakeShape({batch_dim, emb_vec_size_dim});
          ctx->set_output(i, output_shape);
        }
        ctx->set_output(num_lookups + i,
                        ctx->Vector(InferenceContext::kUnknownDim));
        ctx->set_output(num_lookups * 2 + i, ctx->input(num_lookups + i));
        ctx->set_output(num_lookups * 3 + i,
                        ctx->Vector(InferenceContext::kUnknownDim));
      }

      return OkStatus();
    });

REGISTER_OP("GroupVariableLookupGrad")
    .Input("grads: num_lookups * float32")
    .Input("embedding_variables: num_lookups * dtype")
    .Input("unique_keys: num_lookups * Tkeys")
    .Input("sp_indices: num_lookups * int64")
    .Input("batch_nums: num_lookups * int32")
    .Output("nnz_grads: num_lookups * float32")
    .Attr("dimension: int")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("num_lookups: int >=1")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_lookups = ctx->num_outputs();
      for (int i = 0; i < num_lookups; ++i) {
        ShapeHandle top_grad_shape;
        TF_RETURN_IF_ERROR(
            ctx->WithRankAtLeast(ctx->input(i), 2, &top_grad_shape));
        DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);
        ctx->set_output(i,
                        ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      }
      return OkStatus();
    });

REGISTER_OP("GroupVariableLookupDense")
    .Input("emb_variables: num_lookups * dtype")
    .Input("dense_values: num_lookups * Tkeys")
    .Input("default_value: dtype")
    .Output("output: num_lookups * dtype")
    .Output("unique_keys: num_lookups * Tkeys")
    .Output("unique_idx: num_lookups * int32")
    .Attr("dimension: int")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64, int32}")
    .Attr("max_norm: float = -1.0")
    .Attr("num_lookups: int >= 1")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'} = 'mean'")  // placeholder
    .Attr("ignore_weights: bool = true")                  // placeholder
    .SetShapeFn([](InferenceContext* ctx) {
      int num_lookups;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_lookups", &num_lookups));

      for (int i = 0; i < num_lookups; ++i) {
        ShapeHandle temp;
        TF_RETURN_IF_ERROR(
            ctx->WithRankAtLeast(ctx->input(num_lookups + i), 1, &temp));
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(i), 1, &unused));
        ShapeHandle params_subshape;
        TF_RETURN_IF_ERROR(ctx->Subshape(ctx->input(i), 1, &params_subshape));
        DimensionHandle emb_vec_size_dim = ctx->Dim(params_subshape, 0);
        DimensionHandle batch_dim = ctx->UnknownDim();
        ShapeHandle output_shape =
            ctx->MakeShape({batch_dim, emb_vec_size_dim});
        ShapeHandle offset_shape = ctx->MakeShape({batch_dim, 1});
        ctx->set_output(i, output_shape);
        ctx->set_output(num_lookups + i,
                        ctx->Vector(InferenceContext::kUnknownDim));
        // ctx->set_output(num_lookups * 2 + i, ctx->input(num_lookups+i));
      }

      return OkStatus();
    });

}  // namespace tensorflow
