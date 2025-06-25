# Build multiplex_2 custom ops example, which is similar to np.where.
# This example supports GPU (and CPU), in contrast to multiplex_1 which
# only supports CPU.

load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])

custom_op_library(
    name = "multiplex_2_kernel.so",
    srcs = [
        "multiplex_2_kernel.cc",
        "multiplex_2_kernel.h",
        "multiplex_2_op.cc",
    ],
    gpu_srcs = [
        "multiplex_2_kernel.h",
        "multiplex_2_kernel.cu.cc",
    ],
)

py_library(
    name = "multiplex_2_op",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "multiplex_2_test",
    size = "medium",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_2_op",
        "@pypi_numpy//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
