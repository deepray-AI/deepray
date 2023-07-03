"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_compressor//tensorflow:workspace2.bzl", rules_compressor_deps = "tf_workspace2")
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
load("@pybind11_bazel//:python_configure.bzl", "python_configure")

def _tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = str(Label("//third_party/python_runtime:headers")),
    )

def workspace():
    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    _tf_bind()
    python_configure(
        name = "local_config_python",
        python_version = "3",
    )
    rules_foreign_cc_dependencies()

    rules_compressor_deps()
    boost_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace0 = workspace
