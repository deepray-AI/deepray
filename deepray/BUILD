licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "LICENSE",
])

package_group(
    name = "internal",
    includes = [
    ],
    packages = [
        "//...",
        "//deepray/...",
    ],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "deepray",
    srcs = glob(["**/*.py"]),
    deps = [
        "//deepray/activations",
        "//deepray/callbacks",
        "//deepray/core",
        "//deepray/custom_ops",
        "//deepray/datasets",
        "//deepray/layers",
        "//deepray/losses",
        "//deepray/metrics",
        "//deepray/models",
        "//deepray/optimizers",
        # "//deepray/seq2seq",
        "//deepray/testing",
        # "//deepray/text",
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
