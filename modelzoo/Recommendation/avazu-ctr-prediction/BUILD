licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "flen",
    srcs = glob(
        [
            "*.py",
            "**/*.py",
        ],
    ),
    # deps = [
    #     "//deepray/custom_ops/parquet_dataset",
    #     "//deepray/datasets:datapipeline",
    # ],
)

py_binary(
    name = "ali_parquet_dataset_test",
    srcs = ["ali_parquet_dataset_test.py"],
    deps = [
        ":ali_parquet_dataset",
    ],
)

sh_binary(
    name = "run_horovod",
    srcs = ["run_horovod.sh"],
    data = [
        ":flen",
    ],
)
