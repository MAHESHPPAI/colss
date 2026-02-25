import inspect
import numpy as np
from colss._colss import sigma as _sigma
from colss._colss import prod as _prod
from colss._colss import mean as _mean
from colss._colss import median as _median
from colss._colss import query as _query



def _collect_arrays(expr: str):
    frame = inspect.currentframe()
    frame = frame.f_back if frame else None

    arrays = {}
    scalars = {}

    while frame is not None:
        for name, val in frame.f_locals.items():
            if name in expr and name not in arrays and name not in scalars:
                if isinstance(val, np.ndarray):
                    arrays[name] = np.ascontiguousarray(val, dtype=np.float64)
                elif isinstance(val, (int, float)):
                    scalars[name] = float(val)

        if arrays or scalars:
            break

        frame = frame.f_back

    if not arrays and not scalars:
        raise RuntimeError(f"No variables found matching names in '{expr}'")

    # Determine the broadcast length from the first array found
    if arrays:
        n = next(iter(arrays.values())).shape[0]
    else:
        # All operands are scalars — treat as length-1 arrays
        n = 1

    # Broadcast scalars to full-length arrays
    for name, val in scalars.items():
        arrays[name] = np.full(n, val, dtype=np.float64)

    return arrays



def sigma(expr: str) -> float:
    return _sigma(expr, **_collect_arrays(expr))


def prod(expr: str) -> float:
    return _prod(expr, **_collect_arrays(expr))


def mean(expr: str) -> float:
    return _mean(expr, **_collect_arrays(expr))


def query(expr: str) -> np.ndarray:
    return _query(expr, **_collect_arrays(expr))


def median(expr: str) -> float:
    return _median(expr, **_collect_arrays(expr))
