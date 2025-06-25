#include "deepray/custom_ops/embedding_variable/cc/embedding/embedding_var.h"
#include "deepray/custom_ops/embedding_variable/cc/lib/tensor_bundle.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {

namespace {

// Shared validations of the inputs to the SaveV2 and RestoreV2 ops.
void ValidateInputs(bool is_save_op, OpKernelContext* context,
                    const Tensor& prefix, const Tensor& tensor_names,
                    const Tensor& shape_and_slices, const int kFixedInputs) {
  const int num_tensors = static_cast<int>(tensor_names.NumElements());
  OP_REQUIRES(
      context, prefix.NumElements() == 1,
      errors::InvalidArgument("Input prefix should have a single element, got ",
                              prefix.NumElements(), " instead."));
  OP_REQUIRES(context,
              TensorShapeUtils::IsVector(tensor_names.shape()) &&
                  TensorShapeUtils::IsVector(shape_and_slices.shape()),
              errors::InvalidArgument(
                  "Input tensor_names and shape_and_slices "
                  "should be an 1-D tensors, got ",
                  tensor_names.shape().DebugString(), " and ",
                  shape_and_slices.shape().DebugString(), " instead."));
  OP_REQUIRES(context,
              tensor_names.NumElements() == shape_and_slices.NumElements(),
              errors::InvalidArgument("tensor_names and shape_and_slices "
                                      "have different number of elements: ",
                                      tensor_names.NumElements(), " vs. ",
                                      shape_and_slices.NumElements()));
  OP_REQUIRES(context,
              FastBoundsCheck(tensor_names.NumElements() + kFixedInputs,
                              std::numeric_limits<int>::max()),
              errors::InvalidArgument("Too many inputs to the op"));
  OP_REQUIRES(
      context, shape_and_slices.NumElements() == num_tensors,
      errors::InvalidArgument("Expected ", num_tensors,
                              " elements in shapes_and_slices, but got ",
                              context->input(2).NumElements()));
  if (is_save_op) {
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Got ", num_tensors, " tensor names but ",
                    context->num_inputs() - kFixedInputs, " tensors."));
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Expected a total of ", num_tensors + kFixedInputs,
                    " inputs as input #1 (which is a string "
                    "tensor of saved names) contains ",
                    num_tensors, " names, but received ", context->num_inputs(),
                    " inputs"));
  }
}

}  // namespace

class SaveV3 : public OpKernel {
 public:
  explicit SaveV3(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &tensor_types_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_key_types", &ev_key_types_));
    OP_REQUIRES_OK(context, context->GetAttr("has_ev", &has_ev_));
  }

  template <typename TKey, typename TValue>
  void DumpEvWithGlobalStep(OpKernelContext* context, const string& tensor_name,
                            EmbeddingVar<TKey, TValue>* ev,
                            BundleWriter& writer, DataType global_step_type) {
    if (global_step_type == DT_INT32) {
      DumpEv<TKey, TValue, int32>(context, ev, tensor_name, writer);
    } else {
      DumpEv<TKey, TValue, int64>(context, ev, tensor_name, writer);
    }
  }

  template <typename TKey, typename TValue, typename TGlobalStep>
  void DumpEv(OpKernelContext* context, EmbeddingVar<TKey, TValue>* variable,
              const string& tensor_name, BundleWriter& writer) {
    const Tensor& global_step = context->input(5);
    TGlobalStep global_step_scalar = global_step.scalar<TGlobalStep>()();
    core::ScopedUnref s(variable);
    embedding::ShrinkArgs shrink_args;
    shrink_args.global_step = global_step_scalar;
    const Tensor& prefix = context->input(0);
    const string& prefix_string = prefix.scalar<tstring>()();
    OP_REQUIRES_OK(context, variable->Save(tensor_name, prefix_string, &writer,
                                           shrink_args));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const Tensor& ev_names = context->input(3);
    const Tensor& ev_resources = context->input(4);
    const int kFixedInputs = 5;
    ValidateInputs(true /* is save op */, context, prefix, tensor_names,
                   shape_and_slices, kFixedInputs);
    if (!context->status().ok()) return;
    // Prefix, tensor names, shape_and_slices, ev names, ev resources.
    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    const int num_ev = static_cast<int>(ev_names.NumElements());
    const string& prefix_string = prefix.scalar<tstring>()();
    const auto& tensor_names_flat = tensor_names.flat<tstring>();
    const auto& ev_names_flat = ev_names.flat<tstring>();
    const auto& ev_resources_flat = ev_resources.flat<int64>();
    const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

    BundleWriter writer(Env::Default(), prefix_string);
    OP_REQUIRES_OK(context, writer.status());
    VLOG(1) << "BundleWriter, prefix_string: " << prefix_string;

    int start_index = 0;
    if (has_ev_) {
      start_index = 1;
    }

    for (int i = 0; i < num_ev; i++) {
      const string& ev_name = ev_names_flat(i);
      if (ev_key_types_[i] == DT_INT32) {
        EmbeddingVar<int32, float>* ev =
            reinterpret_cast<EmbeddingVar<int32, float>*>(ev_resources_flat(i));
        DumpEvWithGlobalStep(context, ev_name, ev, writer, tensor_types_[0]);
      } else if (ev_key_types_[i] == DT_INT64) {
        EmbeddingVar<int64, float>* ev =
            reinterpret_cast<EmbeddingVar<int64, float>*>(ev_resources_flat(i));
        DumpEvWithGlobalStep(context, ev_name, ev, writer, tensor_types_[0]);
      }
    }

    for (int i = start_index; i < num_tensors; ++i) {
      const string& tensor_name = tensor_names_flat(i);
      if (tensor_types_[i] == DT_RESOURCE) {
        auto& handle = HandleFromInput(context, i + kFixedInputs);

      } else {
        const Tensor& tensor = context->input(i + kFixedInputs);

        if (!shape_and_slices_flat(i).empty()) {
          const string& shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(tensor.dims());
          TensorShape slice_shape;

          OP_REQUIRES_OK(context,
                         checkpoint::ParseShapeAndSlice(shape_spec, &shape,
                                                        &slice, &slice_shape));
          OP_REQUIRES(
              context, slice_shape.IsSameSize(tensor.shape()),
              errors::InvalidArgument(
                  "Slice in shape_and_slice "
                  "specification does not match the "
                  "shape of the tensor to  save: ",
                  shape_spec, ", tensor: ", tensor.shape().DebugString()));

          OP_REQUIRES_OK(context,
                         writer.AddSlice(tensor_name, shape, slice, tensor));
        } else {
          OP_REQUIRES_OK(context, writer.Add(tensor_name, tensor));
        }
      }
    }
    OP_REQUIRES_OK(context, writer.Finish());
  }

 private:
  DataTypeVector tensor_types_;
  DataTypeVector ev_key_types_;
  bool has_ev_;
};
REGISTER_KERNEL_BUILDER(Name("SaveV3").Device(DEVICE_CPU), SaveV3);

}  // namespace tensorflow