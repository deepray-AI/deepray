"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("@rules_compressor//tensorflow:workspace2.bzl", rules_compressor_deps = "tf_workspace2")
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

def workspace():
    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    rules_foreign_cc_dependencies()
    rules_compressor_deps()
    boost_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
dp_workspace0 = workspace
