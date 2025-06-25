load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "fused_embedding_ops",
    srcs = [
        "compile_util.h",
        "fused_layer_normalize_ops.cc",
    ],
    gpu_deps = [
    ],
    gpu_srcs = [
    ],
    deps = [
        "@com_github_google_leveldb//:leveldb",
        "@sparsehash_c11//:dense_hash_map",
        # "@org_tensorflow//tensorflow/core:framework_headers_lib",
        # "@org_tensorflow//tensorflow/core/common_runtime:core_cpu",
        "@org_tensorflow//tensorflow/core/kernels:training_op_helpers",
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array",
        "@org_tensorflow//tensorflow/core/kernels:gather_functor",
    ],
)
