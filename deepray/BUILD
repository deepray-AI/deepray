licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "deepray",
    srcs = glob(["*.py"]),
    deps = [
        "//deepray/activations",
        "//deepray/callbacks",
        "//deepray/image",
        "//deepray/layers",
        "//deepray/losses",
        "//deepray/metrics",
        "//deepray/optimizers",
        "//deepray/rnn",
        "//deepray/seq2seq",
        "//deepray/testing",
        "//deepray/text",
        "//deepray/utils",
    ],
)

py_test(
    name = "deepray_test",
    size = "small",
    srcs = glob(["tests/*"]),
    main = "tests/run_all_test.py",
    deps = [
        ":deepray",
    ],
)
