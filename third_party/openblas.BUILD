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
    cache_entries = {
        "BUILD_TESTING": "OFF",
    },
    copts = ["-Wno-unused-result"],
    lib_source = ":all_srcs",
    out_lib_dir = select({
        "@platforms//os:linux": "lib",
        "//conditions:default": "lib",
    }),
    out_static_libs = select({
        "@platforms//os:macos": ["libopenblas.a"],
        "@platforms//os:linux": ["libopenblas.a"],
        "@platforms//os:windows": ["openblas.lib"],
    }),
)
