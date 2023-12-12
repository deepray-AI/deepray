load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_correlation_cost_ops.so",
    srcs = [
        "cc/kernels/correlation_cost_op.cc",
        "cc/kernels/correlation_cost_op.h",
        "cc/ops/correlation_cost_op.cc",
    ],
    cuda_srcs = [
        "cc/kernels/correlation_cost_op.h",
        "cc/kernels/correlation_cost_op_gpu.cu.cc",
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
    data = [
        ":_correlation_cost_ops.so",
    ],
    deps = [
        "//deepray/utils",
    ],
)

py_test(
    name = "correlation_cost_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":correlation_cost",
    ],
)
