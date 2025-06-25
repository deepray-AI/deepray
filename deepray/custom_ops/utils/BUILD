load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "spin_rw_lock",
    srcs = [
        "spin_rw_lock.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "spin_lock",
    srcs = [
        "spin_lock.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ok_status_util",
    srcs = [
        "ok_status_util.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "random",
    srcs = [
        "random.cc",
        "random.h",
    ],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "check_util",
    srcs = [
        "check.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_testutil",
    testonly = 1,
    srcs = ["tensor_testutil.cc"],
    hdrs = ["tensor_testutil.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "tensor_testutil_test",
    size = "small",
    srcs = ["tensor_testutil_test.cc"],
    deps = [
        ":tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "kernel_benchmark_testlib",
    testonly = 1,
    srcs = ["kernel_benchmark_testlib.cc"],
    hdrs = ["kernel_benchmark_testlib.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_benchmark//:benchmark",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "fake_input",
    testonly = 1,
    srcs = ["fake_input.cc"],
    hdrs = ["fake_input.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cuda_library(
    name = "ops_testutil",
    testonly = 1,
    srcs = ["ops_testutil.cc"],
    hdrs = ["ops_testutil.h"],
    deps = [
        ":tensor_testutil",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "ops_testutil_test",
    size = "small",
    srcs = ["ops_testutil_test.cc"],
    linkopts = [
        "-lm",
    ],
    deps = [
        ":fake_input",
        ":ops_testutil",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)
