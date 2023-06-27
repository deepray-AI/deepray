load("//deepray:deepray.bzl", "custom_op_library")

custom_op_library(
    name = "kernel_example.so",
    srcs = [
        "kernel_example.cc",
        "kernel_example.h",
    ],
    cuda_deps = [
        "@cub_archive//:cub",
    ],
    cuda_srcs = [
        "kernel_example.cu.cc",
        "kernel_example.h",
    ],
    defines = [
        # "GOOGLE_CUDA",
        "EIGEN_USE_GPU",
    ],
)
