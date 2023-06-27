# Tensorflow default + linux implementations of tensorflow/core/platform libraries.
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//deepray/tsl:tsl.default.bzl", "filegroup")
load("//deepray/tsl:tsl.bzl", "if_not_fuchsia", "if_not_windows", "set_external_visibility", "tsl_copts")
load("//deepray/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//deepray:license"],
    default_visibility = set_external_visibility([
        "//deepray/core/lib/jpeg:__pkg__",
        "//deepray/core/platform:__pkg__",
        "//deepray/tsl/platform:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
)

cc_library(
    name = "context",
    hdrs = ["//deepray/tsl/platform:context.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["context.h"],
    deps = [
        "//deepray/tsl/platform",
    ],
)

cc_library(
    name = "cord",
    hdrs = ["cord.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["@com_google_absl//absl/strings:cord"],
)

cc_library(
    name = "cuda_libdevice_path",
    srcs = ["cuda_libdevice_path.cc"],
    hdrs = ["//deepray/tsl/platform:cuda_libdevice_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:types",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "dso_loader",
    srcs = ["dso_loader.cc"] + select({
        # include dynamic loading checker only for open source build
        "//deepray/tsl:oss": ["dlopen_checker.cc"],
        "//conditions:default": ["dlopen_checker_stub.cc"],
    }),
    hdrs = ["dso_loader.h"],
    compatible_with = [],
    copts = tsl_copts(),
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:env",
        "//deepray/tsl/platform:errors",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:status",
        "//deepray/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
)

cc_library(
    name = "env",
    srcs = [
        "posix_file_system.cc",
        "//deepray/tsl/platform:env.cc",
        "//deepray/tsl/platform:file_system.cc",
        "//deepray/tsl/platform:file_system_helper.cc",
        "//deepray/tsl/platform:threadpool.cc",
    ],
    hdrs = [
        "posix_file_system.h",
        "//deepray/tsl/platform:env.h",
        "//deepray/tsl/platform:file_system.h",
        "//deepray/tsl/platform:file_system_helper.h",
        "//deepray/tsl/platform:ram_file_system.h",
        "//deepray/tsl/platform:threadpool.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:blocking_counter",
        "//deepray/tsl/platform:context",
        "//deepray/tsl/platform:cord",
        "//deepray/tsl/platform:denormal",
        "//deepray/tsl/platform:env_time",
        "//deepray/tsl/platform:errors",
        "//deepray/tsl/platform:file_statistics",
        "//deepray/tsl/platform:load_library",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:platform_port",
        "//deepray/tsl/platform:protobuf",
        "//deepray/tsl/platform:regexp",
        "//deepray/tsl/platform:scanner",
        "//deepray/tsl/platform:setround",
        "//deepray/tsl/platform:status",
        "//deepray/tsl/platform:statusor",
        "//deepray/tsl/platform:str_util",
        "//deepray/tsl/platform:strcat",
        "//deepray/tsl/platform:stringpiece",
        "//deepray/tsl/platform:stringprintf",
        "//deepray/tsl/platform:threadpool_interface",
        "//deepray/tsl/platform:tracing",
        "//deepray/tsl/platform:types",
        "//deepray/tsl/protobuf:error_codes_proto_impl_cc",
        "//third_party/eigen3",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "env_impl",
    srcs = [
        "env.cc",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        ":env",
        "//deepray/tsl/platform:load_library",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:strcat",
        "//deepray/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"],
    hdrs = ["//deepray/tsl/platform:env_time.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = ["//deepray/tsl/platform:types"],
)

cc_library(
    name = "human_readable_json",
    srcs = ["human_readable_json.cc"],
    hdrs = ["//deepray/tsl/platform:human_readable_json.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:errors",
        "//deepray/tsl/platform:protobuf",
        "//deepray/tsl/platform:status",
        "//deepray/tsl/platform:strcat",
    ],
)

cc_library(
    name = "load_library",
    srcs = ["load_library.cc"],
    hdrs = ["//deepray/tsl/platform:load_library.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:errors",
        "//deepray/tsl/platform:status",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["//deepray/tsl/platform:logging.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["logging.h"],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:env_time",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "cord.h",
        "dynamic_annotations.h",
        "integral_types.h",
    ] + if_not_windows(["env_time.cc"]),
)

cc_library(
    name = "mutex",
    srcs = [
        "mutex.cc",
        "mutex_data.h",
        "//deepray/tsl/platform:mutex.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["mutex.h"],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:thread_annotations",
        "//deepray/tsl/platform:types",
        "@nsync//:nsync_cpp",
    ],
)

cc_library(
    name = "net",
    srcs = ["net.cc"],
    hdrs = [
        "//deepray/tsl/platform:net.h",
    ],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:strcat",
    ],
    alwayslink = True,
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:types",
    ],
)

cc_library(
    name = "platform_port",
    srcs = [
        "port.cc",
        "//deepray/tsl/platform:cpu_info.cc",
    ],
    hdrs = [
        "//deepray/tsl/platform:cpu_info.h",
        "//deepray/tsl/platform:demangle.h",
        "//deepray/tsl/platform:host_info.h",
        "//deepray/tsl/platform:init_main.h",
        "//deepray/tsl/platform:mem.h",
        "//deepray/tsl/platform:numa.h",
        "//deepray/tsl/platform:snappy.h",
        "//deepray/tsl/platform/profile_utils:cpu_utils.h",
    ],
    copts = tsl_copts(),
    defines = ["TF_USE_SNAPPY"] + select({
        # TF Additional NUMA defines
        "//deepray/tsl:with_numa_support": ["TENSORFLOW_USE_NUMA"],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:byte_order",
        "//deepray/tsl/platform:dynamic_annotations",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:types",
        "//deepray/tsl/platform/profile_utils:profile_utils_cpu_utils",
        "@com_google_absl//absl/base",
        "@snappy",
    ] + select({
        # TF Additional NUMA dependencies
        "//deepray/tsl:with_numa_support": [
            # Don't merge in a single line
            "@hwloc",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "resource",
    srcs = ["resource.cc"],
    hdrs = ["//deepray/tsl/platform:resource.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "resource_loader",
    testonly = 1,
    srcs = ["resource_loader.cc"],
    hdrs = ["//deepray/tsl/platform:resource_loader.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:test",
        "@bazel_tools//tools/cpp/runfiles",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    srcs = ["rocm_rocdl_path.cc"],
    hdrs = ["//deepray/tsl/platform:rocm_rocdl_path.h"],
    compatible_with = [],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:types",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    linkopts = ["-ldl"],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:abi",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = ["stacktrace_handler.cc"],
    hdrs = ["//deepray/tsl/platform:stacktrace_handler_hdrs"],
    linkstatic = 1,
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:stacktrace",
    ],
    alwayslink = 1,
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["//deepray/tsl/platform:subprocess.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["subprocess.h"],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "test",
    testonly = True,
    srcs = ["test.cc"],
    hdrs = ["//deepray/tsl/platform:test.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:net",
        "//deepray/tsl/platform:path",
        "//deepray/tsl/platform:strcat",
        "//deepray/tsl/platform:types",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tracing",
    srcs = [
        "tracing.cc",
        "//deepray/tsl/platform:tracing.cc",
    ],
    hdrs = ["//deepray/tsl/platform:tracing.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["tracing_impl.h"],
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:hash",
        "//deepray/tsl/platform:logging",
        "//deepray/tsl/platform:macros",
        "//deepray/tsl/platform:str_util",
        "//deepray/tsl/platform:strcat",
        "//deepray/tsl/platform:stringpiece",
        "//deepray/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "types",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["integral_types.h"],
)

cc_library(
    name = "unbounded_work_queue",
    srcs = ["unbounded_work_queue.cc"],
    hdrs = ["unbounded_work_queue.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    deps = [
        "//deepray/tsl/platform:env",
        "//deepray/tsl/platform:mutex",
        "//deepray/tsl/platform:notification",
        "//deepray/tsl/platform:platform_port",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "crash_analysis",
    srcs = [
        "crash_analysis.cc",
    ],
    hdrs = ["//deepray/tsl/platform:crash_analysis.h"],
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["crash_analysis.h"],
    visibility = set_external_visibility(["//deepray:__subpackages__"]),
    deps = [
        "//deepray/tsl/platform",
        "//deepray/tsl/platform:protobuf",
    ],
)

cc_library(
    name = "status",
    tags = [
        "manual",
        "no_oss",
        "nobuilder",
    ],
    textual_hdrs = ["status.h"],
    visibility = set_external_visibility(["//deepray:__subpackages__"]),
    deps = [
        "//deepray/tsl/platform:types",
        "//deepray/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

bzl_library(
    name = "cuda_build_defs_bzl",
    srcs = ["cuda_build_defs.bzl"],
    visibility = set_external_visibility(["//deepray:__subpackages__"]),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "additional_mobile_srcs_no_runtime",
    visibility = set_external_visibility(["//deepray/core/platform:__pkg__"]),
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "casts.h",
        "context.h",
        "dynamic_annotations.h",
        "env.cc",
        "integral_types.h",
        "load_library.cc",
        "port.cc",
        "posix_file_system.cc",
        "posix_file_system.h",
        "stacktrace.h",
        "status.h",
        "tracing_impl.h",
        "//deepray/tsl/platform/profile_utils:cpu_utils.h",
        "//deepray/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    visibility = set_external_visibility([
        "//deepray/core/platform:__pkg__",
        "//deepray/tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "casts.h",
        "cord.h",
        "mutex.h",
        "mutex_data.h",
        "notification.h",
        "unbounded_work_queue.cc",
        "unbounded_work_queue.h",
    ] + if_not_fuchsia([
        "subprocess.cc",
        "subprocess.h",
    ]),
    visibility = set_external_visibility(["//deepray/core/platform:__pkg__"]),
)

exports_files(
    srcs = glob(
        ["*"],
        exclude = [
            "integral_types.h",
            "logging.h",
            "test.cc",
        ],
    ),
    visibility = set_external_visibility(["//deepray/core/platform:__pkg__"]),
)

exports_files(
    srcs = [
        "integral_types.h",
        "logging.h",
        "test.cc",
    ],
    visibility = set_external_visibility([
        "//deepray/core:__pkg__",
        "//deepray/core/lib/gif:__pkg__",
        "//deepray/core/lib/jpeg:__pkg__",
        "//deepray/core/platform:__pkg__",
    ]),
)
