licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//deepray:deepray.bzl", "custom_op_library")

custom_op_library(
    name = "_distributed_embeddings_ops.so",
    srcs = [
        "cc/kernels/embedding_lookup.h",
        "cc/kernels/embedding_lookup_kernels.cc",
        "cc/ops/embedding_lookup_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/embedding_lookup_kernels.cu",
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
    name = "embedding_bag_test",
    size = "small",
    srcs = glob(["python/tests/*"]),
    main = "python/tests/run_all_test.py",
    deps = [
        ":embedding_bag",
    ],
)
