
# colss

colss is a lightweight C++ expression engine exposed to Python using pybind11. It evaluates mathematical expressions over NumPy arrays using a compiled backend.

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

## 1. query

```
import numpy as np
import colss

# Input arrays
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)

# Expression
result = colss.query("a+b+7")

print(result)
```

Output:

```
[12. 14. 16.]
```

Explanation:

1 + 4 + 7 = 12
2 + 5 + 7 = 14
3 + 6 + 7 = 16

---

### Ternary Example

```
result = colss.query("a > 1 ? 100 : 0")
print(result)
```

Output:

```
[  0. 100. 100.]
```

---

### Using Mathematical Functions

```
result = colss.query("sqrt(a) + sin(a)")
print(result)
```

---

## 2. mean

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

m = colss.mean(a)
print(m)
```

Output:

```
2.5
```

---

## 3. sigma (Standard Deviation)

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

s = colss.sigma(a)
print(s)
```

Output:

```
1.118033988749895
```

---

## 4. prod (Product)

```
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

p = colss.prod(a)
print(p)
```

Output:

```
24.0
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

# Supported Functions (ExprTk)

The following functions can be used inside expressions:

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

If constants are registered:

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

 df = pd.DataFrame({
    "a": [1.0, 2.0, 3.0],
    "b": [4.0, 5.0, 6.0]
})

# Pandas eval
df["c"] = df.eval("a + b + 7")

# colss evaluation
 a = df["a"].to_numpy(dtype=np.float64)
 b = df["b"].to_numpy(dtype=np.float64)

 df["d"] = colss.query("a+b+7")

print(df)
```

Expected result columns c and d will match.

---

# Notes

* All variables used in query must be registered inside colss before evaluation.
* Always validate expression compilation on the C++ side.
* NaN handling should be managed carefully (either via isnan or NumPy masks).

---

# License

Add your license information here.

