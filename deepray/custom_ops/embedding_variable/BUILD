load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_tsl//tsl/platform/default:build_config.bzl", "py_proto_library")
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "config_proto",
    srcs = ["config.proto"],
)

cc_proto_library(
    name = "config_proto_cc",
    deps = [":config_proto"],
)

py_proto_library(
    name = "config_proto_py_pb2",
    srcs = ["config.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY3",
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_library(
    name = "embedding_variable",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_group_embedding_ops.so",
        ":_kv_variable_ops.so",
    ],
    srcs_version = "PY3",
    deps = [
        ":config_proto_py_pb2",
        "//deepray/utils",
    ],
)

cc_library(
    name = "save_restore_tensor_ev",
    hdrs = [
        "cc/kernels/save_restore_tensor_ev.h",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/lib:tensor_bundle",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "kv_variable_util",
    srcs = ["cc/kernels/kv_variable_util.cc"],
    hdrs = [
        "cc/kernels/kv_variable_util.h",
    ],
    copts = ["-Wno-unused-result"],
    deps = [
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
    ],
)

cuda_library(
    name = "training_ali_lib",
    srcs = [
        "cc/kernels/training_ali_ops_gpu.cu.cc",
    ],
    hdrs = [
        "cc/kernels/training_ali_ops_gpu.h",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
    ],
)

GROUP_EMBEDDING_OP_SRCS = [
    "cc/group_embedding/group_embedding_lookup_ops.cc",
    "cc/group_embedding/group_embedding_lookup_sparse_backward_ops.cc",
    "cc/group_embedding/group_embedding_lookup_sparse_forward_base_ops.h",
    "cc/group_embedding/group_embedding_lookup_sparse_forward_ops.cc",
]

GROUP_EMBEDDING_OP_GPU_SRCS = [
    "cc/group_embedding/group_embedding_lookup_sparse_backward_ops.cu.cc",
    "cc/group_embedding/group_embedding_lookup_sparse_forward_ops.cu.cc",
    "cc/group_embedding/group_embedding_lookup_sparse_backward_base_ops.cu.h",
    "cc/group_embedding/group_embedding_lookup_sparse_forward_base_ops.cu.h",
]

cuda_library(
    name = "fused_embedding_common_cuh",
    hdrs = ["cc/fused_embedding/fused_embedding_common.cu.h"],
)

FUSED_EMBEDDING_OP_SRCS = [
    "cc/fused_embedding/embedding_lookup_sparse_post_op.cc",
    "cc/fused_embedding/embedding_lookup_sparse_pre_op.cc",
    "cc/fused_embedding/fused_embedding_ops.cc",
]

FUSED_EMBEDDING_OP_GPU_SRCS = [
    "cc/fused_embedding/fused_embedding_pre_ops_gpus.cu.cc",
    "cc/fused_embedding/fused_embedding_post_ops_gpus.cu.cc",
]

custom_op_library(
    name = "_kv_variable_ops.so",
    srcs = [
        "cc/kernels/kv_variable_lookup_ops.cc",
        "cc/kernels/kv_variable_ops.cc",
        "cc/kernels/kv_variable_restore_ops.cc",
        "cc/kernels/save_restore_ops.cc",
        "cc/kernels/training_adagrad_ops.cc",
        "cc/kernels/training_adam_async_ops.cc",
        "cc/kernels/training_adam_ops.cc",
        "cc/kernels/training_ali_op_helpers.h",
        "cc/kernels/training_ftrl_ops.cc",
        "cc/kernels/training_sgd_ops.cc",
        "cc/ops/kv_variable_ops.cc",
        "cc/ops/save_restore_ops.cc",
        "cc/ops/training_adagrad_ops.cc",
        "cc/ops/training_adam_async_ops.cc",
        "cc/ops/training_adam_ops.cc",
        "cc/ops/training_ftrl_ops.cc",
        "cc/ops/training_sgd_ops.cc",
    ],
    copts = ["-Wno-unused-result"] + if_cuda(["-DGOOGLE_CUDA=1"]),
    gpu_deps = [
        ":training_ali_lib",
    ],
    gpu_srcs = [
        "cc/kernels/training_ali_ops_gpu.h",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable:kv_variable_util",
        "//deepray/custom_ops/unique_ops:unique_ali_util",
        "//deepray/custom_ops/utils:spin_rw_lock",
        "@com_github_google_leveldb//:leveldb",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

py_test(
    name = "multiplex_1_test",
    size = "medium",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":embedding_variable",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)

custom_op_library(
    name = "_raw_ops.so",
    srcs = [
        "cc/kernels/embedding_collection.cc",
        "cc/kernels/embedding_collection.hpp",
        "cc/ops/embedding_collection.cc",
    ],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":hotness_calculate",
    ],
)

cuda_library(
    name = "hotness_calculate",
    srcs = [
        "cc/kernels/hotness_calculate.cu.cc",
    ],
    hdrs = [
        "cc/kernels/hotness_calculate.h",
    ],
    defines = [
        "TF_VERSION_MAJOR=2",
    ],
    deps = [
        "//deepray/custom_ops/utils:check_util",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@local_config_cuda//cuda:cudart",
    ],
)

custom_op_library(
    name = "_save_restore_ops.so",
    srcs = [
        "cc/kernels/save_restore_ops.cc",
        "cc/ops/save_restore_ops.cc",
    ],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
    ],
)

custom_op_library(
    name = "_group_embedding_ops.so",
    srcs = [
        "cc/ops/group_embedding_ops.cc",
    ] + GROUP_EMBEDDING_OP_SRCS,
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    gpu_deps = [
        ":fused_embedding_common_cuh",
    ],
    gpu_srcs = GROUP_EMBEDDING_OP_GPU_SRCS,
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
        "//deepray/custom_ops/unique_ops:unique_ali_util",
    ],
)

cc_test(
    name = "group_embedding_ops_test",
    size = "small",
    srcs = ["cc/group_embedding/group_embedding_lookup_ops_test.cc"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":_group_embedding_ops.so",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable:kv_variable_util",
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
        "//deepray/custom_ops/utils:fake_input",
        "//deepray/custom_ops/utils:kernel_benchmark_testlib",
        "//deepray/custom_ops/utils:ops_testutil",
        "//deepray/custom_ops/utils:tensor_testutil",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

custom_op_library(
    name = "_incr_save_restore_ops.so",
    srcs = [
        "cc/incr_save_restore/incr_save_restore_ops.cc",
        "cc/incr_save_restore/incr_save_restore_ops.h",
        "cc/ops/incr_save_restore_ops.cc",
    ],
    deps = [
        ":save_restore_tensor_ev",
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
    ],
)

cc_test(
    name = "incr_save_restore_ops_test",
    size = "small",
    srcs = ["cc/incr_save_restore/incr_save_restore_ops_test.cc"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":_incr_save_restore_ops.so",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable:kv_variable_util",
        "//deepray/custom_ops/embedding_variable/cc/embedding:embedding_ops_lib",
        "//deepray/custom_ops/utils:fake_input",
        "//deepray/custom_ops/utils:kernel_benchmark_testlib",
        "//deepray/custom_ops/utils:ops_testutil",
        "//deepray/custom_ops/utils:tensor_testutil",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
