#!/usr/bin/env bash
set -e

docker run --rm -v $(pwd):/project -w /project \
	quay.io/pypa/manylinux_2_28_x86_64 bash -c "
        yum install -y libomp-devel &&
        /opt/python/cp312-cp312/bin/pip install pybind11 numpy build &&
        /opt/python/cp312-cp312/bin/python -m build --wheel &&
        auditwheel repair dist/*.whl -w /project/wheelhouse
    "
