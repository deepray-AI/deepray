licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//deepray:deepray.bzl", "custom_op_library")

custom_op_library(
    name = "_distort_image_ops.so",
    srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.cc",
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/ops/distort_image_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/adjust_hsv_in_yiq_op.h",
        "cc/kernels/adjust_hsv_in_yiq_op_gpu.cu.cc",
    ],
)

custom_op_library(
    name = "_image_ops.so",
    srcs = [
        "cc/kernels/connected_components.cc",
        "cc/kernels/connected_components.h",
        "cc/kernels/euclidean_distance_transform_op.cc",
        "cc/kernels/euclidean_distance_transform_op.h",
        "cc/ops/image_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/euclidean_distance_transform_op.h",
        "cc/kernels/euclidean_distance_transform_op_gpu.cu.cc",
    ],
)

custom_op_library(
    name = "_resampler_ops.so",
    srcs = [
        "cc/kernels/resampler_ops.cc",
        "cc/kernels/resampler_ops.h",
        "cc/ops/resampler_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/resampler_ops.h",
        "cc/kernels/resampler_ops_gpu.cu.cc",
    ],
)

py_library(
    name = "image",
    srcs = glob(["python/*.py"]),
    data = [
        ":_distort_image_ops.so",
        ":_image_ops.so",
        ":_resampler_ops.so",
        ":sparse_image_warp_test_data",
        "//deepray/testing",
        "//deepray/utils",
    ],
)

filegroup(
    name = "sparse_image_warp_test_data",
    srcs = glob(["python/tests/test_data/*.png"]),
)

py_test(
    name = "image_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":image",
    ],
)
