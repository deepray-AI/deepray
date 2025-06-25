#!/bin/bash
set -e

yes "" | bash ./configure || true

# --compilation_mode dbg \
bazel build build_pip_pkg \
    --copt=-O3 --copt=-march=native \
    -s

rm -rf artifacts/

bazel-bin/build_pip_pkg artifacts

pip uninstall deepray -y

pip install artifacts/deepray-*.whl

# sphinx-autobuild docs/ docs/_build/html/ --host 10.0.74.1
