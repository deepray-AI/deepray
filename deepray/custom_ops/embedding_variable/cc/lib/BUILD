load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_tsl//tsl/platform:build_config.bzl", "tsl_cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc",
        "allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        # ":ev_allocator",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "ev_allocator",
    srcs = [
        "ev_allocator.cc",
        "ev_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":ev_allocator_gpu",
        "//deepray/custom_ops/utils:spin_lock",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@readerwriterqueue_archive//:readerwriterqueue",
    ],
    alwayslink = 1,
)

cuda_library(
    name = "ev_allocator_gpu",
    srcs = [
        "ev_allocator_gpu.cu.cc",
    ],
    hdrs = [
        "ev_allocator.h",
    ],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        "//deepray/custom_ops/utils:spin_lock",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@readerwriterqueue_archive//:readerwriterqueue",
    ],
)

cc_test(
    name = "ev_allocator_test",
    size = "small",
    srcs = ["ev_allocator_test.cc"],
    deps = [
        ":ev_allocator",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "main",
    srcs = [
        "test.cc",
    ],
    deps = [
        ":allocator",
        "//deepray/custom_ops/utils:tensor_testutil",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "pmem_allocator",
    srcs = [
        "pmem_allocator.cc",
        "pmem_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        # ":ev_allocator_gpu",
        "//deepray/custom_ops/utils:spin_lock",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "experimental_pmem_allocator",
    srcs = [
        "experimental_pmem_allocator.cc",
        "experimental_pmem_allocator.h",
        "experimental_pmem_allocator_utils.cc",
        "experimental_pmem_allocator_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        # ":ev_allocator_gpu",
        "//deepray/custom_ops/utils:spin_lock",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "buffered_file",
    hdrs = ["buffered_file.h"],
    deps = [
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "tensor_bundle",
    srcs = [
        "tensor_bundle.cc",
    ],
    hdrs = [
        "tensor_bundle.h",
    ],
    deps = [
        ":buffered_file",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

tsl_cc_test(
    name = "tensor_bundle_test",
    srcs = ["tensor_bundle_test.cc"],
    linkopts = [
        "-lm",
    ],
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        ":tensor_bundle",
        "//deepray/custom_ops/utils:tensor_testutil",
        "@com_google_benchmark//:benchmark",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
