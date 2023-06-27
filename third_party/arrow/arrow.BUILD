# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

genrule(
    name = "arrow_util_config",
    srcs = ["cpp/src/arrow/util/config.h.cmake"],
    outs = ["cpp/src/arrow/util/config.h"],
    cmd = ("sed " +
           "-e 's/@ARROW_VERSION_MAJOR@/3/g' " +
           "-e 's/@ARROW_VERSION_MINOR@/0/g' " +
           "-e 's/@ARROW_VERSION_PATCH@/0/g' " +
           "-e 's/cmakedefine ARROW_USE_NATIVE_INT128/undef ARROW_USE_NATIVE_INT128/g' " +
           "-e 's/cmakedefine/define/g' " +
           "$< >$@"),
)

genrule(
    name = "parquet_version_h",
    srcs = ["cpp/src/parquet/parquet_version.h.in"],
    outs = ["cpp/src/parquet/parquet_version.h"],
    cmd = ("sed " +
           "-e 's/@PARQUET_VERSION_MAJOR@/1/g' " +
           "-e 's/@PARQUET_VERSION_MINOR@/5/g' " +
           "-e 's/@PARQUET_VERSION_PATCH@/1/g' " +
           "$< >$@"),
)

cc_library(
    name = "arrow_uriparser",
    srcs = glob(["cpp/src/arrow/vendored/uriparser/*.c"]),
    hdrs = glob(["cpp/src/arrow/vendored/uriparser/*.h"]),
    includes = ["cpp/src/arrow/vendored/uriparser"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/array/*.cc",
            "cpp/src/arrow/compute/*.cc",
            "cpp/src/arrow/compute/**/*.h",
            "cpp/src/arrow/compute/**/*.cc",
            "cpp/src/arrow/csv/*.cc",
            "cpp/src/arrow/dataset/*.cc",
            "cpp/src/arrow/filesystem/*.cc",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/arrow/json/*.cc",
            "cpp/src/arrow/tensor/*.cc",
            "cpp/src/arrow/util/*.cc",
            "cpp/src/arrow/vendored/optional.hpp",
            "cpp/src/arrow/vendored/string_view.hpp",
            "cpp/src/arrow/vendored/variant.hpp",
            "cpp/src/arrow/vendored/base64.cpp",
            "cpp/src/arrow/vendored/datetime/tz.cpp",
            "cpp/src/arrow/vendored/pcg/*.hpp",
            "cpp/src/arrow/**/*.h",
            "cpp/src/parquet/**/*.h",
            "cpp/src/parquet/**/*.cc",
            "cpp/src/generated/*.h",
            "cpp/src/generated/*.cpp",
            "cpp/thirdparty/flatbuffers/include/flatbuffers/*.h",
        ],
        exclude = [
            "cpp/src/**/*_benchmark.cc",
            "cpp/src/**/*_main.cc",
            "cpp/src/**/*_nossl.cc",
            "cpp/src/**/*test*.h",
            "cpp/src/**/*test*.cc",
            "cpp/src/**/*fuzz*.cc",
            "cpp/src/**/*gcsfs*.cc",
            "cpp/src/**/file_to_stream.cc",
            "cpp/src/**/stream_to_file.cc",
            "cpp/src/arrow/util/bpacking_avx2.cc",
            "cpp/src/arrow/util/bpacking_avx512.cc",
            "cpp/src/arrow/util/bpacking_neon.cc",
            "cpp/src/arrow/util/tracing_internal.cc",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "cpp/src/arrow/vendored/musl/strptime.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        # declare header from above genrule
        "cpp/src/arrow/util/config.h",
        "cpp/src/parquet/parquet_version.h",
    ],
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/std:c++14",
        ],
        "//conditions:default": [
            "-std=c++14",
        ],
    }),
    defines = [
        "ARROW_HDFS=ON",
        "ARROW_WITH_BROTLI",
        "ARROW_WITH_SNAPPY",
        "ARROW_WITH_LZ4",
        "ARROW_WITH_ZLIB",
        "ARROW_WITH_ZSTD",
        "ARROW_WITH_BZ2",
        "ARROW_STATIC",
        "ARROW_EXPORT=",
        "PARQUET_STATIC",
        "PARQUET_EXPORT=",
        "WIN32_LEAN_AND_MEAN",
        "ARROW_DS_STATIC",
        "URI_STATIC_BUILD",
    ],
    includes = [
        "cpp/src",
        "cpp/src/arrow/vendored/xxhash",
        "cpp/src/generated",
        "cpp/thirdparty/flatbuffers/include",
    ],
    textual_hdrs = [
        "cpp/src/arrow/vendored/xxhash/xxhash.c",
    ],
    deps = [
        ":arrow_uriparser",
        "@//third_party/hadoop:hdfs",
        "@aws-sdk-cpp//:identity-management",
        "@aws-sdk-cpp//:s3",
        "@boringssl//:crypto",
        "@com_github_apache_thrift//:thrift",
        "@com_github_facebook_zstd//:zstd",
        "@com_github_google_brotli//:brotli",
        "@com_github_google_double_conversion//:double-conversion",
        "@com_github_google_snappy//:snappy",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@com_github_xtensorstack_xsimd//:xsimd",
        "@lz4",
        "@org_bzip_bzip2//:bzip2",
        "@zlib",
    ],
)
