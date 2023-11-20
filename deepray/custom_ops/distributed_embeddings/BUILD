load("//deepray:deepray.bzl", "custom_op_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_distributed_embeddings_ops.so",
    srcs = [
        "cc/kernels/embedding_lookup.h",
        "cc/kernels/embedding_lookup_kernels.cc",
        "cc/ops/embedding_lookup_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/embedding_lookup_kernels.cu.cc",
    ],
    deps = [
        "@cuCollections//:cuco_hash_table",
    ],
)

py_library(
    name = "distributed_embeddings_ops",
    srcs = glob(
        [
            "python/*.py",
            "*.py",
        ],
    ),
    data = [
        ":_distributed_embeddings_ops.so",
        "//deepray:options.py",
    ],
    deps = [
        "//deepray/utils",
    ],
)

py_test(
    name = "distributed_embeddings_ops_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":distributed_embeddings_ops",
    ],
)
