set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.8-cudnn8.6-tensorrt8.4_config_cuda//crosstool:toolchain"
fi

bazel build $CUDA_FLAG //deepray/...
cp ./bazel-bin/deepray/custom_ops/image/_*_ops.so ./deepray/custom_ops/image/
cp ./bazel-bin/deepray/custom_ops/layers/_*_ops.so ./deepray/custom_ops/layers/
cp ./bazel-bin/deepray/custom_ops/seq2seq/_*_ops.so ./deepray/custom_ops/seq2seq/
cp ./bazel-bin/deepray/custom_ops/text/_*_ops.so ./deepray/custom_ops/text/
cp ./bazel-bin/deepray/custom_ops/text/_parse_time_op.so ./deepray/custom_ops/text/
cp ./bazel-bin/deepray/custom_ops/text/_parse_time_op.so ./deepray/custom_ops/text/
cp ./bazel-bin/deepray/custom_ops/parquet_dataset/_parquet_pybind.so ./deepray/custom_ops/parquet_dataset/
cp ./bazel-bin/deepray/custom_ops/parquet_dataset/_parquet_dataset_ops.so ./deepray/custom_ops/parquet_dataset/
