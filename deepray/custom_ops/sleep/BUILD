# (non-blocking) op using AsyncOpKernel

load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "sleep_kernel.so",
    srcs = [
        "sleep_kernel.cc",
        "sleep_op.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
    ],
)

py_library(
    name = "sleep_op",
    srcs = ["sleep_op.py"],
    data = ["sleep_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "sleep_bin",
    srcs = ["sleep_bin.py"],
    srcs_version = "PY3",
    deps = [
        ":sleep_op",
        "@absl_py//absl:app",
    ],
)

py_test(
    name = "sleep_test",
    size = "medium",  # This test blocks using sleep,
    timeout = "short",  # but it still runs quickly.
    srcs = ["sleep_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":sleep_op",
        "@pypi_tensorflow//:pkg",
    ],
)
