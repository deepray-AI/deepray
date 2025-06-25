load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

custom_op_library(
    name = "_training_ops.so",
    srcs = [
        "cc/kernels/training_ops.cc",
        "cc/kernels/training_ops.h",
        "cc/ops/training_ops.cc",
    ],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    gpu_srcs = [
        "cc/kernels/training_ops.h",
        "cc/kernels/training_ops_gpu.cu.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
        # "@org_tensorflow//tensorflow/core/kernels:training_op_helpers",
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
