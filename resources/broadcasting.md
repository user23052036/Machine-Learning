# NumPy Broadcasting — Element-wise Division

---

# Broadcasting Examples

---

## Example 1 — Column vector + scalar

$$
\begin{bmatrix}1\\2\\3\\4\end{bmatrix}_{(4,1)}
+\ 100
\;\longrightarrow\;
\begin{bmatrix}1\\2\\3\\4\end{bmatrix}
+
\begin{bmatrix}100\\100\\100\\100\end{bmatrix}
=
\begin{bmatrix}101\\102\\103\\104\end{bmatrix}
$$

The scalar `100` is broadcast to match the shape `(4, 1)`.

---

## Example 2 — Matrix + row vector `(m,n) + (1,n) → (m,n)`

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}_{(m,n)}
+
\begin{bmatrix}100&200&300\end{bmatrix}_{(1,n)}
$$

The row vector is broadcast **down the rows**:

$$
(1,n) \;\longrightarrow\; (m,n)
\quad\Rightarrow\quad
\begin{bmatrix}100&200&300\\100&200&300\end{bmatrix}_{(m,n)}
$$

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
+
\begin{bmatrix}100&200&300\\100&200&300\end{bmatrix}
=
\begin{bmatrix}101&202&303\\104&205&306\end{bmatrix}
$$

Each **column** gets a different addend — col 1 gets +100, col 2 gets +200, col 3 gets +300.

---

## Example 3 — Matrix + column vector `(m,n) + (m,1) → (m,n)`

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}_{(m,n)}
+
\begin{bmatrix}100\\200\end{bmatrix}_{(m,1)}
$$

The column vector is broadcast **across the columns**:

$$
(m,1) \;\longrightarrow\; (m,n)
\quad\Rightarrow\quad
\begin{bmatrix}100&100&100\\200&200&200\end{bmatrix}_{(m,n)}
$$

$$
\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}
+
\begin{bmatrix}100&100&100\\200&200&200\end{bmatrix}
=
\begin{bmatrix}101&102&103\\204&205&206\end{bmatrix}
$$

Each **row** gets a different addend — row 1 gets +100, row 2 gets +200.

---

## Summary

| Shape of operand | Broadcast direction | Each ___ gets a different value |
|---|---|---|
| `(1, n)` — row vector | Copied **down** rows | Column |
| `(m, 1)` — column vector | Copied **across** columns | Row |
| scalar | Copied in **all** directions | Every element |

---

## Key Concept

`A / cal` in NumPy is **not** matrix division. It is **element-wise division** governed by broadcasting rules.

| Operation | What it does |
|-----------|-------------|
| `A / cal` | Element-wise division (broadcasting) |
| `A * cal` | Element-wise multiplication (broadcasting) |
| `A @ cal` | Matrix multiplication (different shape rules) |

---

## Setup

```python
A = np.array([
    [56,   0,    4.4,  68 ],
    [1.2,  104,  52,   8  ],
    [1.8,  135,  99,   0.9]
])
# A.shape → (3, 4)

cal = A.sum(axis=0)
# cal → [59, 239, 155.4, 76.9]
# cal.shape → (4,)

cal = cal.reshape(1, 4)
# cal.shape → (1, 4)
```

$$
A = \begin{bmatrix}
56  & 0   & 4.4 & 68  \\
1.2 & 104 & 52  & 8   \\
1.8 & 135 & 99  & 0.9
\end{bmatrix}_{(3,4)}
\qquad
cal = \begin{bmatrix}59 & 239 & 155.4 & 76.9\end{bmatrix}_{(1,4)}
$$

---

## Case 1 — `100 * A / cal` with `cal.shape = (1, 4)`

### Broadcasting rule

NumPy compares dimensions **right to left**:

```
A   → (3, 4)
cal → (1, 4)
          ↑
   last dim: 4 == 4  ✓
   first dim: 3 vs 1 → broadcast cal
```

Conceptually, NumPy replicates `cal` down 3 rows:

$$
\begin{bmatrix}59 & 239 & 155.4 & 76.9\end{bmatrix}
\;\xrightarrow{\text{broadcast}}\;
\begin{bmatrix}
59 & 239 & 155.4 & 76.9 \\
59 & 239 & 155.4 & 76.9 \\
59 & 239 & 155.4 & 76.9
\end{bmatrix}_{(3,4)}
$$

> NumPy does **not** physically create the copies — this is just the mental model.

### Calculation

$$
100 \times
\begin{bmatrix}
56/59   & 0/239   & 4.4/155.4 & 68/76.9  \\
1.2/59  & 104/239 & 52/155.4  & 8/76.9   \\
1.8/59  & 135/239 & 99/155.4  & 0.9/76.9
\end{bmatrix}
$$

Row 1 result:

$$
\begin{bmatrix}94.92 & 0 & 2.83 & 88.43\end{bmatrix}
$$

### Intuition

Since `cal = A.sum(axis=0)`, each element in `cal` is the **column total**. Dividing `A` by `cal` gives each element as a **percentage of its column total** — i.e., column-wise normalisation.

---

## Case 2 — `100 * A.T / cal` with `cal.shape = (4, 1)`

### Setup

```python
cal = A.sum(axis=0).reshape(4, 1)
# cal.shape → (4, 1)

percentage = 100 * A.T / cal
```

After transpose:

$$
A^T = \begin{bmatrix}
56  & 1.2 & 1.8 \\
0   & 104 & 135 \\
4.4 & 52  & 99  \\
68  & 8   & 0.9
\end{bmatrix}_{(4,3)}
\qquad
cal = \begin{bmatrix}59 \\ 239 \\ 155.4 \\ 76.9\end{bmatrix}_{(4,1)}
$$

### Broadcasting rule

```
A.T → (4, 3)
cal → (4, 1)
          ↑
   last dim: 3 vs 1 → broadcast cal across columns
   first dim: 4 == 4  ✓
```

Conceptually, `cal` is replicated across 3 columns:

$$
\begin{bmatrix}59\\239\\155.4\\76.9\end{bmatrix}
\;\xrightarrow{\text{broadcast}}\;
\begin{bmatrix}
59    & 59    & 59    \\
239   & 239   & 239   \\
155.4 & 155.4 & 155.4 \\
76.9  & 76.9  & 76.9
\end{bmatrix}_{(4,3)}
$$

### Calculation — first two rows

**Row 1 (Apple):** $[56,\ 1.2,\ 1.8] \div 59$

$$
100 \times \frac{[56,\ 1.2,\ 1.8]}{[59,\ 59,\ 59]} = [94.915,\ 2.034,\ 3.051]
$$

**Row 2 (Orange):** $[0,\ 104,\ 135] \div 239$

$$
100 \times \frac{[0,\ 104,\ 135]}{[239,\ 239,\ 239]} = [0,\ 43.514,\ 56.485]
$$

### Mental picture

```
Original A (3, 4)                     After .T → A.T (4, 3)

         Apple  Orange  Guava  Cherry          Carbs  Protein  Fat
Carbs      56      0     4.4    68      Apple    56     1.2    1.8
Protein   1.2    104      52     8      Orange    0     104    135
Fat       1.8    135      99   0.9      Guava    4.4     52     99
                                        Cherry   68       8    0.9

cal (4, 1)       → broadcast across columns →   cal (4, 3)

Apple   59                                59    59    59
Orange  239                              239   239   239
Guava   155.4                          155.4 155.4 155.4
Cherry  76.9                            76.9  76.9  76.9
```

Now each row is one fruit, and `cal` divides each row by that fruit's total — **row-wise normalisation** of the transposed matrix.

---

## Comparison — Case 1 vs Case 2

| | Case 1 | Case 2 |
|---|---|---|
| Expression | `100 * A / cal` | `100 * A.T / cal` |
| `A` shape | `(3, 4)` | — |
| `A.T` shape | — | `(4, 3)` |
| `cal` shape | `(1, 4)` | `(4, 1)` |
| Broadcast direction | `cal` copied **down rows** | `cal` copied **across columns** |
| Normalisation | Each element ÷ column total | Each element ÷ row total (of transposed) |
| Result shape | `(3, 4)` | `(4, 3)` |

---

## Broadcasting Rules — Summary

1. Compare shapes **right to left**, dimension by dimension.
2. Dimensions are compatible if they are **equal**, or one of them is **1**.
3. A dimension of size `1` is stretched to match the other.
4. If shapes have different numbers of dimensions, prepend `1`s to the shorter shape.

```
(3, 4) / (1, 4) → (3, 4)   ✓
(4, 3) / (4, 1) → (4, 3)   ✓
(3, 4) / (4, 1) → Error     ✗  (3 vs 4, neither is 1)
```