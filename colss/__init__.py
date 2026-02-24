import inspect
import numpy as np
from colss._colss import sigma as _sigma
from colss._colss import prod as _prod
from colss._colss import mean as _mean
from colss._colss import query as _query

def _collect_vars(expr: str):
    frame = inspect.currentframe()
    frame = frame.f_back.f_back if frame and frame.f_back else None
    arrays = {}
    scalars = {}
    while frame is not None:
        for name, val in frame.f_locals.items():
            if name not in expr:
                continue
            if isinstance(val, np.ndarray):
                if name not in arrays:
                    arrays[name] = np.ascontiguousarray(val, dtype=np.float64)
            elif isinstance(val, (int, float)) and name not in scalars:
                scalars[name] = float(val)
        if arrays or scalars:
            break
        frame = frame.f_back
    if not arrays and not scalars:
        raise RuntimeError(f"No variables found matching '{expr}'")
    return arrays, scalars

def sigma(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _sigma(expr, scalars, **arrays)

def prod(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _prod(expr, scalars, **arrays)

def mean(expr: str) -> float:
    arrays, scalars = _collect_vars(expr)
    return _mean(expr, scalars, **arrays)

def query(expr: str) -> np.ndarray:
    arrays, scalars = _collect_vars(expr)
    return _query(expr, scalars, **arrays)
