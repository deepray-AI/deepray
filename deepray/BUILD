load("//deepray:tensorflow.bzl", "if_google")
load("@bazel_skylib//lib:selects.bzl", "selects")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

# Sometimes Bazel reports darwin_x86_64 as "darwin" and sometimes as
# "darwin_x86_64". The former shows up when building on a Mac x86_64 host for a Mac x86_64 target.
# The latter shows up when cross-compiling for Mac x86_64 from a Mac ARM machine and in internal
# Google builds.
config_setting(
    name = "macos_x86_64_default",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_x86_64_crosscompile",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
)

selects.config_setting_group(
    name = "macos_x86_64",
    match_any = [
        ":macos_x86_64_default",
        ":macos_x86_64_crosscompile",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    flag_values = if_google(
        {"//tools/cpp:cc_target_os": "apple"},
        {},
    ),
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon): Remove in favor of TSL version
selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
    visibility = ["//visibility:public"],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {"framework_shared_object": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos_with_framework_shared_object",
    match_any = [
        ":macos_x86_64_with_framework_shared_object",
        ":macos_arm64_with_framework_shared_object",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "deepray",
    srcs = glob(["*.py"]),
    deps = [
        "//deepray/activations",
        "//deepray/callbacks",
        "//deepray/core",
        "//deepray/custom_ops",
        "//deepray/datasets",
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
