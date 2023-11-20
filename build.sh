#!/bin/bash
set -e

yes "" | bash ./configure || true

# bazel build build_pip_pkg \
#     --action_env=HTTP_PROXY=http://127.0.0.1:7890 \
#     --action_env=HTTPS_PROXY=http://127.0.0.1:7890

bazel build build_pip_pkg

rm -rf artifacts/

bazel-bin/build_pip_pkg artifacts

pip uninstall deepray -y

pip install artifacts/deepray-*.whl

# sphinx-autobuild docs/ docs/_build/html/ --host 10.0.74.1
