docker run --rm -it \
	-v $(pwd):/app \
	-w /app \
	ubuntu:22.04 \
	bash -c "
    apt update &&
    apt install -y python3.11 python3.11-venv python3-pip build-essential cmake ninja-build libomp-dev &&
    python3.11 -m venv venv &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install numpy pybind11 pytest build numexpr pandas &&
    python -m build &&
    pip install dist/*.whl &&
    pytest -v
  "
