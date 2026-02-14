
## 1️⃣ `df.info()`

**Purpose:** Structural summary of the DataFrame.

It tells you:

* Number of rows
* Column names
* Data types (`int64`, `float64`, `object`, etc.)
* Non-null counts per column
* Memory usage

Example output type:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries
Data columns (total 21 columns):
...
```

### Use it when:

* Checking missing values
* Debugging dtype issues
* Validating preprocessing
* Inspecting dataset shape

It does **not** compute statistics.

---

## 2️⃣ `df.describe()`

**Purpose:** Statistical summary.

By default (numeric columns only), it returns:

* count
* mean
* std
* min
* 25%
* 50% (median)
* 75%
* max

Example:

```
       tenure  MonthlyCharges
count  7043.0     7043.0
mean     32.3       64.8
std      24.6       30.1
...
```

### Use it when:

* Understanding distribution
* Checking scaling
* Detecting outliers
* Validating normalization

---

## 🔥 Core Difference

| `df.info()`     | `df.describe()`        |
| --------------- | ---------------------- |
| Structure       | Statistics             |
| Data types      | Distribution metrics   |
| Missing counts  | Mean, std, percentiles |
| No calculations | Computes aggregates    |

---

## ⚠️ Important Edge Case

If you want categorical statistics:

```python
df.describe(include='object')
```

Otherwise it ignores non-numeric columns.

---

## Bottom Line

* `info()` → **What kind of data do I have?**
* `describe()` → **What does my numeric data look like statistically?**

If you’re doing ML preprocessing, you should run **both** before touching the data.
