load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//deepray:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "unique_ali_util",
    srcs = [
        "cc/kernels/task_runner.h",
        "cc/kernels/unique_ali_op_util.h",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
        "//deepray/custom_ops/utils:random",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

custom_op_library(
    name = "_unique_ops.so",
    srcs = [
        "cc/kernels/unique_ali_op.cc",
        "cc/ops/unique_ops.cc",
    ],
    copts = [
        "-Wno-unused-variable",
        "-Wno-unused-result",
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    gpu_srcs = [
        "cc/kernels/unique_ali_op_gpu.cu.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":unique_ali_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
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
        "@pypi_numpy//:pkg",
        "@pypi_pytest//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
