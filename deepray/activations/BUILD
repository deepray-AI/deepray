licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "activations",
    srcs = glob(["*.py"]),
    data = [
        "//deepray:options.py",
        "//deepray/testing",
        "//deepray/utils",
    ],
)

py_test(
    name = "activations_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "run_all_test.py",
    deps = [
        ":activations",
    ],
)
