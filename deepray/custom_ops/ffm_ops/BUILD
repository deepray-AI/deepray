load("//deepray:deepray.bzl", "custom_op_library")
load("@local_config_tf//:build_defs.bzl", "CPLUSPLUS_VERSION")

licenses(["notice"])  # Apache 2.0

custom_op_library(
    name = "_ffm_ops.so",
    copts = [CPLUSPLUS_VERSION],
    srcs = [
        "cc/kernels/ffm_kernels.cc",
        "cc/kernels/ffm_kernels.h",
        "cc/ops/ffm_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/ffm_kernels.h",
        "cc/kernels/ffm_kernels.cu.cc",
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
    ],
)
