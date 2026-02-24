# colss

colss is a lightweight C++-powered expression engine exposed to Python using pybind11. It evaluates mathematical expressions over NumPy arrays with fast element-wise execution.

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

---

## 1. query

```
import numpy as np
import colss

# Input arrays
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

result = colss.query("a+b+7")
print(result)
```

```
[12. 14. 16.]
```

---

### Ternary Example

```
result = colss.query("a > 1 ? 100 : 0")
print(result)
```

```
[  0. 100. 100.]
```

---

### Using Mathematical Functions

```
result = colss.query("sqrt(a) + sin(a)")
print(result)
```

Example using exponential:

```
colss.query("exp(a)")
```

---

## 2. mean

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

m = colss.mean("a")
print(m)
```

```
2.5
```

Expression example:

```
colss.mean("a+10")
```

---

## 3. sigma (Standard Deviation)

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

s = colss.sigma("a")
print(s)
```

```
1.118033988749895
```

Expression example:

```
colss.sigma("a*2")
```

---

## 4. prod (Product)

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

p = colss.prod("a")
print(p)
```

```
24.0
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

The following functions can be used inside `colss.query()` expressions:

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

pi
e

```

---

# Using with Pandas

```

import pandas as pd
import numpy as np
import colss

# Create DataFrame

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

print(df)

```

Columns `c` and `d` will match.

---

# Notes

- All variables used in expressions must be registered inside colss before evaluation.
- All functions (`query`, `mean`, `sigma`, `prod`) accept string expressions.
- Handle NaN values carefully when required.

```
