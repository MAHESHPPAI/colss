import inspect
import numpy as np
from colss._colss import sigma as _sigma
from colss._colss import prod as _prod
from colss._colss import mean as _mean
from colss._colss import query as _query



def _collect_arrays(expr: str):
    frame = inspect.currentframe()
    frame = frame.f_back if frame else None

    kwargs = {}

    while frame is not None:
        for name, val in frame.f_locals.items():
            if isinstance(val, np.ndarray) and name in expr:
                kwargs[name] = np.ascontiguousarray(val, dtype=np.float64)

        if kwargs:
            break

        frame = frame.f_back

    if not kwargs:
        raise RuntimeError(f"No numpy arrays found matching variables in '{expr}'")

    return kwargs



def sigma(expr: str) -> float:
    return _sigma(expr, **_collect_arrays(expr))


def prod(expr: str) -> float:
    return _prod(expr, **_collect_arrays(expr))


def mean(expr: str) -> float:
    return _mean(expr, **_collect_arrays(expr))


def query(expr: str) -> np.ndarray:
    return _query(expr, **_collect_arrays(expr))
