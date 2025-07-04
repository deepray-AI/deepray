load("@rules_python//python:defs.bzl", "py_test")
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_embedding_bag_ops.so",
    srcs = [
        "cc/kernels/embedding_bag_ops.cc",
        "cc/kernels/embedding_bag_ops.h",
        "cc/ops/embedding_bag_ops.cc",
    ],
    gpu_deps = [
        "@local_config_cuda//cuda:cuda_runtime",
    ],
    gpu_srcs = [
        "cc/kernels/embedding_bag_ops.h",
        "cc/kernels/embedding_bag_ops_gpu.cu.cc",
        "cc/kernels/embedding_bag_backward_kernels.cu.cc",
    ],
)

py_library(
    name = "embedding_bag",
    srcs = glob(
        [
            "python/*.py",
            "*.py",
        ],
    ),
    data = [":_embedding_bag_ops.so"],
)

py_test(
    name = "embedding_bag_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    python_version = "PY3",
    deps = [
        ":embedding_bag",
        "//deepray/utils",
        "@pypi_pytest//:pkg",
        "@pypi_tensorflow//:pkg",
        "@pypi_typeguard//:pkg",
    ],
)
