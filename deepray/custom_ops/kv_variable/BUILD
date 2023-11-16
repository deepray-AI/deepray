load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "@com_google_protobuf//:protobuf.bzl",
    "cc_proto_library",
    "proto_gen",
    "py_proto_library",
)

proto_library(
    name = "checkpoint_state_ext_proto",
    srcs = ["python/training/checkpoint_state_extend.proto"],
)

py_proto_library(
    name = "checkpoint_state_extend_py_pb2",
    srcs = ["python/training/checkpoint_state_extend.proto"],
)

proto_library(
    name = "storage_config_proto",
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "storage_config_proto_cc",
    # srcs = glob(["kernels/hybrid_embedding/*.proto"]),
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
    include = "kernels/hybrid_embedding",
    # deps = [":storage_config_proto"]
)

py_proto_library(
    name = "storage_config_proto_py",
    srcs = ["kernels/hybrid_embedding/storage_config.proto"],
)

cc_library(
    name = "kv_variable_lib",
    srcs = [
        "kernels/kv_variable_ops.cc",
        "kernels/naming.cc",
        "kernels/training_ops.cc",
        "kernels/utility.cc",
        "ops/kv_variable_ops.cc",
        "ops/training_ops.cc",
        "utils/progress_bar.cc",
        "utils/utils.cc",
    ],
    hdrs = [
        "kernels/dynamic_restore.hpp",
        "kernels/dynamic_save.hpp",
        "kernels/embedding_value.h",
        "kernels/hashmap.h",
        "kernels/hybrid_embedding/embedding_context.h",
        "kernels/hybrid_embedding/storage_table.h",
        "kernels/hybrid_embedding/table_manager.h",
        "kernels/kv_variable.h",
        "kernels/kv_variable_cwise_op.h",
        "kernels/kv_variable_interface.h",
        "kernels/mutex.h",
        "kernels/naming.h",
        "kernels/utility.h",
        "utils/progress_bar.h",
        "utils/utils.h",
    ],
    copts = [
        "-std=c++17",
        "-fPIC",
        "-DNDEBUG",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-D__STDC_FORMAT_MACROS",
    ],
    linkstatic = 1,
    deps = [
        ":storage_config_proto_cc",
        "@farmhash",
        "@libcuckoo",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@murmurhash",
        "@sparsehash",
        "@tbb",
    ],
)

custom_op_library(
    name = "python/ops/_kv_variable_ops.so",
    srcs = [
        "kernels/dynamic_restore.hpp",
        "kernels/dynamic_save.hpp",
        "kernels/embedding_value.h",
        "kernels/hashmap.h",
        "kernels/hybrid_embedding/embedding_context.h",
        "kernels/hybrid_embedding/storage_table.h",
        "kernels/hybrid_embedding/table_manager.h",
        "kernels/kv_variable.h",
        "kernels/kv_variable_cwise_op.h",
        "kernels/kv_variable_interface.h",
        "kernels/kv_variable_ops.cc",
        "kernels/mutex.h",
        "kernels/naming.cc",
        "kernels/naming.h",
        "kernels/training_ops.cc",
        "kernels/utility.cc",
        "kernels/utility.h",
        "ops/kv_variable_ops.cc",
        "ops/training_ops.cc",
        "utils/progress_bar.cc",
        "utils/progress_bar.h",
        "utils/utils.cc",
        "utils/utils.h",
    ],
    copts = [
        "-std=c++17",
        "-pthread",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ],
    # linkshared = 1,
    deps = [
        ":storage_config_proto_cc",
        "@farmhash",
        "@libcuckoo",
        "@murmurhash",
        "@sparsehash",
        "@tbb",
    ],
)

cc_library(
    name = "kv_variable_opdef",
    srcs = [
        "ops/kv_variable_ops.cc",
        "ops/training_ops.cc",
    ],
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "kv_variable_test",
    size = "small",
    srcs = [
        "kernels/kv_variable_test.cc",
    ],
    copts = [
        "-std=c++17",
    ],
    linkopts = [
        "-lbz2",
        "-llzma",
    ],
    deps = [
        ":kv_variable_lib",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "kv_variable_interface",
    hdrs = [
        "kernels/kv_variable_interface.h",
        "kernels/mutex.h",
        "kernels/naming.h",
        "kernels/tensor_bundle.h",
    ],
    copts = [
        "-std=c++17",
        "-DNDEBUG",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@tbb",
    ],
)
