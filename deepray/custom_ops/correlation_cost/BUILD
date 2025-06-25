load("@rules_python//python:defs.bzl", "py_test")
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

CORRELATION_COST_OP_SRCS = [
    "cc/kernels/correlation_cost_op.cc",
    "cc/ops/correlation_cost_op.cc",
]

custom_op_library(
    name = "_correlation_cost_ops.so",
    srcs = CORRELATION_COST_OP_SRCS + ["cc/kernels/correlation_cost_op.h"],
    gpu_deps = [
        "@cub_archive//:cub",
    ],
    gpu_srcs = [
        "cc/kernels/correlation_cost_op_gpu.cu.cc",
        "cc/kernels/correlation_cost_op.h",
    ],
)

py_library(
    name = "correlation_cost",
    srcs = glob(
        [
            "python/*.py",
            "*.py",
        ],
    ),
    data = [":_correlation_cost_ops.so"],
)

py_test(
    name = "correlation_cost_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    python_version = "PY3",
    deps = [
        ":correlation_cost",
        "//deepray/utils",
        "@pypi_pytest//:pkg",
        "@pypi_tensorflow//:pkg",
        "@pypi_typeguard//:pkg",
    ],
)
