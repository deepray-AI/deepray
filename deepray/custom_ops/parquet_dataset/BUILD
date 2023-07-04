load("//deepray:deepray.bzl", "custom_op_library")
load("//deepray:tensorflow.bzl", "pybind_extension")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_parquet_dataset_ops.so",
    srcs = [
        "cc/kernels/parquet_dataset_ops.cc",
        "cc/kernels/parquet_dataset_ops.h",
        "cc/ops/parquet_ops.cc",
    ],
    deps = [
        ":arrow_util",
        ":parquet_batch_reader",
    ],
)

cc_library(
    name = "arrow_util",
    srcs = [
        "cc/kernels/arrow_util.cc",
        "cc/kernels/eigen.h",
    ],
    hdrs = [
        "cc/kernels/arrow_util.h",
    ],
    defines = [
        "DEEPREC_ARROW_HDFS",
        "DEEPREC_ARROW_ZEROCOPY",
    ],
    deps = [
        "@com_github_apache_arrow//:arrow",
        "@eigen3",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "parquet_batch_reader",
    srcs = [
        "cc/kernels/parquet_batch_reader.cc",
    ],
    hdrs = [
        "cc/kernels/parquet_batch_reader.h",
    ],
    deps = [
        ":arrow_util",
        "@com_github_apache_arrow//:arrow",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

pybind_extension(
    name = "_parquet_pybind",
    srcs = [
        "cc/kernels/parquet_pybind.cc",
    ],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    module_name = "_parquet_pybind",
    deps = [
        ":arrow_util",
        "@pybind11",
    ],
)
