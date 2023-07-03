load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "openblas",
    build_args = [
        "--verbose",
        "-j `nproc`",
    ],
    # Values to be passed as -Dkey=value on the CMake command line;
    # here are serving to provide some CMake script configuration options
    cache_entries = {
        "NOFORTRAN": "on",
        "BUILD_WITHOUT_LAPACK": "no",
    },
    lib_source = ":all_srcs",
    out_lib_dir = select({
        "@platforms//os:linux": "lib",
        "//conditions:default": "lib",
    }),
    # linkopts = ["-lpthread"],
    # We are selecting the resulting static library to be passed in C/C++ provider
    # as the result of the build;
    # However, the cmake_external dependants could use other artefacts provided by the build,
    # according to their CMake script
    out_static_libs = ["libopenblas.a"],
    alwayslink = True,
)
