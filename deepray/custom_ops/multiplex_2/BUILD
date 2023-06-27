# Build multiplex_2 custom ops example, which is similar to np.where.
# This example supports GPU (and CPU), in contrast to multiplex_1 which
# only supports CPU.

load("//deepray:strict.default.bzl", "py_strict_library")
load("//deepray:deepray.bzl", "custom_op_library")
load("//deepray:tensorflow.default.bzl", "cuda_py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

custom_op_library(
    name = "multiplex_2_kernel.so",
    srcs = [
        "multiplex_2_kernel.cc",
        "multiplex_2_kernel.h",
        "multiplex_2_op.cc",
    ],
    cuda_srcs = [
        "multiplex_2_kernel.h",
        "multiplex_2_kernel.cu.cc",
    ],
)

py_strict_library(
    name = "multiplex_2_op",
    srcs = ["multiplex_2_op.py"],
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/examples/custom_ops_doc:__subpackages__"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "multiplex_2_test",
    size = "small",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_2_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
