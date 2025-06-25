load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

custom_op_library(
    name = "_ffm_ops.so",
    srcs = [
        "cc/kernels/ffm_kernels.cc",
        "cc/kernels/ffm_kernels.h",
        "cc/ops/ffm_ops.cc",
    ],
    copts = ["-Wno-unused-result"],
    gpu_srcs = [
        "cc/kernels/ffm_kernels.h",
        "cc/kernels/ffm_kernels.cu.cc",
    ],
    deps = [
        "//deepray/custom_ops/utils:ok_status_util",
    ],
)

py_library(
    name = "ffm_ops",
    srcs = glob(
        [
            "python/*.py",
            "python/**/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_ffm_ops.so",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//deepray/utils",
    ],
)

py_test(
    name = "ffm_ops_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":ffm_ops",
        "//deepray/layers",
        "@pypi_numpy//:pkg",
        "@pypi_pytest//:pkg",
        "@pypi_tensorflow//:pkg",
    ],
)
