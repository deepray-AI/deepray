load("//deepray:deepray.bzl", "custom_op_library")
load("@local_config_tf//:build_defs.bzl", "CPLUSPLUS_VERSION")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//deepray:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "random",
    srcs = [
        "cc/kernels/random.cc",
        "cc/kernels/random.h",
    ],
    copts = [CPLUSPLUS_VERSION],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "random_test",
    srcs = ["cc/kernels/random_test.cc"],
    deps = [
        ":random",
        "@com_google_googletest//:gtest_main",
    ],
)

custom_op_library(
    name = "_unique_ops.so",
    srcs = [
        "cc/kernels/task_runner.h",
        "cc/kernels/unique_ali_op.cc",
        "cc/kernels/unique_ali_op_util.h",
        "cc/ops/unique_ops.cc",
    ],
    copts = [CPLUSPLUS_VERSION],
    cuda_srcs = [
        "cc/kernels/unique_ali_op_gpu.cu.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":random",
        "@com_google_absl//absl/container:flat_hash_map",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

py_library(
    name = "unique_ops",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_unique_ops.so",
    ],
)

py_test(
    name = "unique_ops_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":unique_ops",
    ],
)
