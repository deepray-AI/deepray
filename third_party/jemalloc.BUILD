load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

configure_make(
    name = "jemalloc",
    args = [
        "-j `nproc`",
    ],
    autogen = True,
    configure_in_place = True,
    lib_source = ":all_srcs",
    linkopts = [
        "-pthread",
        "-ldl",
    ],
    out_shared_libs = select({
        "@platforms//os:macos": [
            "libjemalloc.dylib",
        ],
        "@platforms//os:linux": [
            "libjemalloc.so",
        ],
        "@platforms//os:windows": [
            "libjemalloc.dll",
        ],
    }),
    out_static_libs = [
        "libjemalloc.a",
    ],
)
