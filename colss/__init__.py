import inspect
import numpy as np
from colss._colss import sigma as _sigma


def sigma(expr: str) -> float:
    frame = inspect.currentframe().f_back
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

    return _sigma(expr, **kwargs)
