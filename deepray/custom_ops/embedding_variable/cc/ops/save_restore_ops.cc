#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SaveV3")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("ev_names: string")
    .Input("ev_resources: int64")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .Attr("ev_key_types: list(type) = []")
    .Attr("has_ev: bool = false")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      ShapeHandle s;
      DimensionHandle unused_dim;

      // Validate prefix.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

      // Validate tensor_names and shapes_and_slices.
      for (int i = 1; i <= 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &s));
        TF_RETURN_IF_ERROR(
            c->WithValue(c->Dim(s, 0), c->num_inputs() - 5, &unused_dim));
      }

      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &s));
      return OkStatus();
    });

REGISTER_OP("KvResourceImport")
    .Input("resource_handle: resource")
    .Input("value: dtype")
    .Input("empty_key: Tkeys")
    .Input("keys: Tkeys")
    .Input("values: dtype")
    .Input("versions: int64")
    .Attr("shape: shape")
    .Attr("Tkeys: {int64, int32}")
    .Attr("dtype: type")
    .Attr("steps_to_live: int = 0")
    .Attr("ht_type: string = ''")
    .Attr("ht_partition_num: int = 1000")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

      // TODO(dingchen): Validate keys and values shape.
      return OkStatus();
    })
    .Doc(R"doc(
Replaces the contents of the table with the specified keys and values.

The tensor `keys` must be of the same type as the keys of the table.
The tensor `values` must be of the type of the table values.

resource_handle: Handle to the table.
keys:  Any shape.  Keys to look up.
values: Values to associate with keys.
)doc");

REGISTER_OP("KvResourceImportV3")
    .Input("prefix: string")
    .Input("resource_self: resource")
    .Input("tensor_names: string")
    .Input("empty_key: Tkeys")
    .Attr("shape: shape")
    .Attr("partition_id: int = 0")
    .Attr("partition_num: int = 1")
    .Attr("Tkeys: {int64, int32}")
    .Attr("dtype: type")
    .Attr("reset_version: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return OkStatus();
    })
    .Doc(R"doc()doc");

REGISTER_OP("KvResourceExport")
    .Input("resource_handle: resource")
    .Output("keys: Tkeys")
    .Output("values: Tvalues")
    .Output("versions: int64")
    .Output("freqs: int64")
    .Attr("Tkeys: {int64, int32}")
    .Attr("Tvalues: type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle values = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(values, 2, &values));
      ShapeHandle keys = c->UnknownShapeOfRank(1);
      ShapeHandle versions = c->UnknownShapeOfRank(1);
      ShapeHandle freqs = c->UnknownShapeOfRank(1);
      c->set_output(0, keys);
      c->set_output(1, values);
      c->set_output(2, versions);
      c->set_output(3, freqs);
      return OkStatus();
    })
    .Doc(R"doc(
Outputs all keys and values in the kv resource.

resource_handle: Handle to the kvResource.
keys: Vector of all keys present in the table.
values: Tensor of all values in the table. Indexed in parallel with `keys`.
versions: Vector of all versions present in the table.
freqs: Vector of all freqs present in the table.
)doc");

}  // namespace tensorflow
