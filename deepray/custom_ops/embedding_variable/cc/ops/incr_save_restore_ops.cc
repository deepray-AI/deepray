#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("KvResourceIncrImport")
    .Input("prefix: string")
    .Input("resource_handle: resource")
    .Input("tensor_names: string")
    .Input("empty_key: Tkeys")
    .Input("value: dtype")
    .Attr("Tkeys: {int64, int32}")
    .Attr("dtype: type")
    .Attr("partition_id: int = 0")
    .Attr("partition_num: int = 1")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle handle;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
      return OkStatus();
    })
    .Doc(R"doc()doc");

REGISTER_OP("IncrSave")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("is_sparse: bool")
    .Input("tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) { return OkStatus(); });

REGISTER_OP("IncrRestore")
    .Input("prefix: string")
    .Input("tensor_names: string")
    .Input("shape_and_slices: string")
    .Input("is_sparse: bool")
    .Input("in_tensors: dtypes")
    .Output("out_tensors: dtypes")
    .Attr("dtypes: list(type)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) { return OkStatus(); });

REGISTER_OP("RecordSparseIndices")
    .Input("keys: TIndex")
    .Attr("var_name: string = ''")
    .Attr("TIndex: {int32, int64}")
    .Attr("auto_record: bool = false")
    .SetShapeFn([](InferenceContext* c) { return OkStatus(); });

REGISTER_OP("ActivateSparseRecorder")
    .Input("tensor_names: string")
    .SetShapeFn([](InferenceContext* c) { return OkStatus(); });

REGISTER_OP("CollectSparseIndices")
    .Output("indices: ktype")
    .Output("global_indices: ktype")
    .Attr("tensor_name: string")
    .Attr("config: string = ''")
    .Attr("part_idx: int = -1")
    .Attr("part_count: int = 0")
    .Attr("hash_bucket_size: int = 0")
    .Attr("part_mode: string = ''")
    .Attr("ktype: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) { return OkStatus(); });

}  // namespace tensorflow
