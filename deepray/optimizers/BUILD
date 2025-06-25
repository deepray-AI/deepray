licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = glob(["*.py"]),
    deps = [
        "//deepray/testing",
        "//deepray/utils",
    ],
)

py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":optimizers",
        "//deepray/custom_ops/embedding_variable",
        "//deepray/custom_ops/training_ops",
    ],
)

py_test(
    name = "adam_test",
    size = "medium",
    srcs = glob(["tests/adam_test.py"]),
    main = "tests/adam_test.py",
    deps = [
        ":optimizers",
        # "//deepray/custom_ops/embedding_variable",
        "//deepray/custom_ops/training_ops",
    ],
)
