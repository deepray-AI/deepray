"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
load("@rules_compressor//tensorflow:workspace2.bzl", rules_compressor_deps = "tf_workspace2")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

def workspace():
    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    # This sets up some common toolchains for building targets. For more details, please see
    # https://bazelbuild.github.io/rules_foreign_cc/0.10.1/flatten.html#rules_foreign_cc_dependencies
    rules_foreign_cc_dependencies()
    rules_pkg_dependencies()
    rules_compressor_deps()
    boost_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace0 = workspace
