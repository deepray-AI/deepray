load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_beam_search_ops.so",
    srcs = [
        "cc/kernels/beam_search_ops.cc",
        "cc/kernels/beam_search_ops.h",
        "cc/ops/beam_search_ops.cc",
    ],
    gpu_srcs = [
        "cc/kernels/beam_search_ops.h",
        "cc/kernels/beam_search_ops_gpu.cu.cc",
    ],
)

py_library(
    name = "seq2seq",
    srcs = glob(
        [
            "python/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_beam_search_ops.so",
    ],
)

py_test(
    name = "seq2seq_test",
    size = "medium",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":seq2seq",
        "//deepray/utils",
        "@pypi_numpy//:pkg",
        "@pypi_pytest//:pkg",
        "@pypi_pytest_xdist//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
