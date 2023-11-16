load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

custom_op_library(
    name = "_training_ops.so",
    srcs = [
        "cc/kernels/training_op_helpers.h",
        "cc/kernels/training_ops.cc",
        "cc/kernels/training_ops.h",
        "cc/ops/training_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/training_ops.h",
        "cc/kernels/training_ops_gpu.cu.cc",
    ],
)

py_library(
    name = "training_ops",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_training_ops.so",
    ],
    visibility = ["//visibility:public"],
)
