load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

cc_library(
    name = "embedding_variable_test_lib",
    hdrs = [
        "embedding_variable_test.h",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable:kv_variable_util",
        "//deepray/custom_ops/embedding_variable/cc/lib:tensor_bundle",
    ],
)

cc_test(
    name = "embedding_variable_ops_test",
    srcs = [
        "embedding_variable_ops_test.cc",
        "embedding_variable_test.h",
    ],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":embedding_variable_test_lib",
        "//deepray/custom_ops/utils:tensor_testutil",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_test(
    name = "embedding_variable_performance_test",
    srcs = [
        "embedding_variable_performance_test.cc",
        "embedding_variable_test.h",
    ],
    deps = [
        ":embedding_variable_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_test(
    name = "embedding_variable_memory_test",
    srcs = [
        "embedding_variable_memory_test.cc",
        "embedding_variable_test.h",
    ],
    deps = [
        ":embedding_variable_test_lib",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
