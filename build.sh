#!/bin/bash
set -e

yes "" | bash ./configure || true

# --compilation_mode dbg \
bazel build build_pip_pkg \
    --sandbox_debug \
    --copt=-O3 --copt=-march=native \
    -s

rm -rf artifacts/ wheelhouse/

bazel-bin/build_pip_pkg artifacts

pip install --force-reinstall auditwheel
bash tools/releases/tf_auditwheel_patch.sh

auditwheel show artifacts/*.whl
python -m auditwheel repair --plat manylinux_2_31_x86_64 artifacts/*.whl
ls -al wheelhouse/

pip uninstall deepray -y

pip install wheelhouse/deepray-*.whl

# sphinx-autobuild docs/ docs/_build/html/ --host 10.0.74.1
