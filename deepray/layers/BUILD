package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "layers",
    srcs = glob(["*.py"]),
    data = [
        "//deepray/custom_ops/correlation_cost:_correlation_cost_ops.so",
        "//deepray/custom_ops/embedding_bag:_embedding_bag_ops.so",
    ],
    deps = [
        "//deepray/activations",
        "//deepray/rnn",
        "//deepray/testing",
        "//deepray/utils",
    ],
)

py_test(
    name = "layers_test",
    size = "large",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":layers",
    ],
)
