"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@python//:defs.bzl", "interpreter")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

# Additional BUILD content that should be added to @pypi_tensorflow.
TF_ADDITIVE_BUILD_CONTENT = """
cc_library(
    name = "tf_headers",
    hdrs = glob(["site-packages/tensorflow/include/**"]),
    includes = ["site-packages/tensorflow/include"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = ["site-packages/tensorflow/libtensorflow_framework.so.2"],
)
"""

NUMPY_BUILD_CONTENT = """
cc_library(
    name = "numpy_headers_2",
    hdrs = glob(["site-packages/numpy/_core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/_core/include/",
)
cc_library(
    name = "numpy_headers_1",
    hdrs = glob(["site-packages/numpy/core/include/**/*.h"]),
    strip_include_prefix="site-packages/numpy/core/include/",
)
cc_library(
    name = "numpy_headers",
    deps = [":numpy_headers_2", ":numpy_headers_1"],
)
"""

def python_init_pip():
    ANNOTATIONS = {
        "numpy": package_annotation(
            additive_build_content = NUMPY_BUILD_CONTENT,
        ),
        "tensorflow": package_annotation(
            additive_build_content = TF_ADDITIVE_BUILD_CONTENT,
        ),
    }

    pip_parse(
        name = "pypi",
        annotations = ANNOTATIONS,
        python_interpreter_target = interpreter,
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
    )
