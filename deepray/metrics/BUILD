licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "metrics",
    srcs = glob(["*.py"]),
    deps = [
        "//deepray/testing",
        "//deepray/utils",
    ],
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":metrics",
    ],
)
