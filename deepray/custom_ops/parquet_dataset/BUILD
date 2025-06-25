load("//deepray:deepray.bzl", "custom_op_library", "pybind_extension")

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
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@org_apache_arrow//:arrow",
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
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@org_apache_arrow//:arrow",
    ],
)

pybind_extension(
    name = "_parquet_pybind",
    srcs = [
        "cc/kernels/parquet_pybind.cc",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    module_name = "_parquet_pybind",
    deps = [
        ":arrow_util",
        "@pybind11",
    ],
)

py_library(
    name = "parquet_pybind",
    srcs = ["python/parquet_pybind.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":_parquet_pybind",
        ":dataframe",
    ],
)

py_library(
    name = "dataframe",
    srcs = ["python/dataframe.py"],
    srcs_version = "PY2AND3",
)

# TODO: Once TF exports symbols in a DLL we can enable parse_time_op for windows
# https://github.com/tensorflow/addons/issues/782
py_library(
    name = "parquet_dataset",
    srcs = glob(
        [
            "python/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_parquet_dataset_ops.so",
        "//deepray/testing",
        "//deepray/utils",
    ],
    deps = [
        ":dataframe",
        ":parquet_pybind",
        "//deepray/utils",
    ],
)

py_test(
    name = "parquet_dataset_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":parquet_dataset",
    ],
)

py_binary(
    name = "read_parquet_deepray",
    srcs = ["read_parquet_deepray.py"],
    deps = [
        ":parquet_dataset",
        "@pypi_fastparquet//:pkg",
        "@pypi_pandas//:pkg",
    ],
)
