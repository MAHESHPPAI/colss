# colss

colss is a lightweight expression evaluator for NumPy, Pandas, and Polars. It simplifies mathematical expressions while preserving memory efficiency and execution speed through a compiled C++ backend.

---

## Installation

Build using CMake:

```
mkdir build
cd build
cmake ..
make
```

Or build a Python wheel:

```
python -m build
pip install dist/colss-*.whl
```

---

## Requirements

All arrays passed to colss must be:

* 1D
* Preferably float64
* C-contiguous

If you have a 2D array:

```
a = a.ravel()
```

---

# Usage Examples

All functions accept string expressions.

## query

```
import numpy as np
import colss

a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

result = colss.query("a+b+7")
```

Ternary example:

```
colss.query("a > 1 ? 100 : 0")
```

Using mathematical functions:

```
colss.query("sqrt(a) + sin(a)")
colss.query("exp(a)")
```

---

## mean

```
import numpy as np
import colss

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

m = colss.mean("a")
```

Expression example:

```
colss.mean("a+10")
```

---

## sigma (Standard Deviation)

```
import numpy as np
import colss

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

s = colss.sigma("a")
```

Expression example:

```
colss.sigma("a*2")
```

---

## prod (Product)

```
import numpy as np
import colss

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

p = colss.prod("a")
```

Expression example:

```
colss.prod("a+1")
```

---

# Supported Operators

Arithmetic:

```
+  -  *  /  ^
```

Comparison:

```
>  <  >=  <=  ==  !=
```

Logical:

```
&&  ||
```

Ternary:

```
condition ? value_if_true : value_if_false
```

---

# Available Functions Inside Expressions

```
abs(x)
sqrt(x)
pow(x, y)
log(x)
log10(x)
exp(x)
sin(x)
cos(x)
tan(x)
floor(x)
ceil(x)
min(x, y)
max(x, y)
```

---

# Using with Pandas

```
import pandas as pd
import numpy as np
import colss

 df = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [4.0, 5.0, 6.0]
})

# Pandas evaluation
 df["c"] = df.eval("a + b + 7")

# colss evaluation
 a = df["a"].to_numpy(dtype=np.float64)
 b = df["b"].to_numpy(dtype=np.float64)

 df["d"] = colss.query("a+b+7")
```

---

# Notes

* All variables used in expressions must be registered inside colss before evaluation.
* All functions (`query`, `mean`, `sigma`, `prod`) accept string expressions.
* Designed for memory efficiency and predictable performance.
