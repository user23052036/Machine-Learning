
## **1. diff between PANDAS and NUMPY**

### 🔹 NumPy = Fast math toolkit

* Works mainly with **numeric arrays**.
* Great for scientific computing, vector algebra, matrices.
* Very fast (written in C).
* But **data must be uniform** (all numbers, same type).
  ✔ Ideal for ML math, matrix ops, linear algebra.

💡 Example:
`numpy.array([1,2,3,4])` — pure numeric array.

---

### 🔹 Pandas = Excel + NumPy on steroids

* Built **on top of NumPy**.
* Handles **tabular data** better (rows + columns).
* Supports **mixed data types** (string, date, numbers).
* Has indexing, filtering, grouping, time-series tools.
  ✔ Ideal for data cleaning, analysis, working with CSVs.

💡 Example:
A table like:

| Name   | Age | Score |
| ------ | --- | ----- |
| Souvik | 21  | 98    |

This is perfect for pandas, not NumPy.

---

### 🔥 Simple way to remember:

| Feature                 | NumPy             | Pandas                    |
| ----------------------- | ----------------- | ------------------------- |
| Data structure          | Array             | DataFrame/Series          |
| Best for                | Math, vectors, ML | Data analysis, cleaning   |
| Supports strings/dates? | Mostly no         | Yes                       |
| Fast matrix ops?        | Yes               | Uses NumPy under the hood |

---

### 🚀 When you should use which?

✔ **NumPy** when you need

* dot products
* matrix multiplication
* linear algebra
* neural network math

✔ **Pandas** when you need

* read CSV/Excel
* filter rows
* group, summarize, pivot
* join tables
* preprocess data before ML

## **2. groupby()**

### Short note: Inner working of `groupby` in Pandas

`groupby` in Pandas follows a **split–apply–combine** execution model and is **lazy by design**.

1. **Split**
   When you call:

   ```python
   group = df.groupby("Type1")
   ```

   Pandas scans the `Type1` column, identifies unique keys (e.g., Fire, Water, Grass), and builds an internal mapping from each key to the row indices belonging to that group.
   No computation on the data happens yet—only the grouping structure is prepared.

2. **Apply**
   When you specify an operation:

   ```python
   group["Height"].min()
   ```

   Pandas:

   * Selects the `Height` column
   * Iterates over each group’s row indices
   * Applies the `min()` reduction independently to each group’s subset

3. **Combine**
   The per-group results are then assembled into a single output object:

   * Index → group keys (`Type1`)
   * Values → computed statistic (`min Height`)
   * Output type → `Series` (or `DataFrame` for multi-column aggregations)

**Key design points**

* `groupby` itself returns a `GroupBy` object, not data.
* Computation is triggered only by aggregations (`min`, `mean`, `sum`, `agg`), transformations (`transform`), or iteration.
* This lazy approach improves performance and memory efficiency on large datasets.

In your example, Pandas computes the **minimum height per Pokémon type** and returns it as a labeled `Series`.

---

## **3. index and column confusion**
Good — this output confirms the diagnosis precisely. Now I’ll be blunt and exact.

---

## What is happening (no ambiguity)

* **`Name` is NOT a column**
* **`Name` is the DataFrame index**

You explicitly did this:

```python
pd.read_csv(..., index_col='Name')
```

So Pandas moved `Name` out of `df.columns` and into `df.index`.

That is why:

```python
df["Name"]
```

fails with `KeyError: 'Name'`.

This is not a Pandas bug. This is expected behavior.

---

## Correct mental model (lock this in)

* `df.columns` → actual data columns
* `df.index` → row labels (metadata, not a column)

Your data now looks like:

```
Index (Name) | No | Type1 | Type2 | Height | Weight | Legendary
```

So you must operate on **the index**, not on columns.

---

## The correct fix (smallest, cleanest)

### If your goal is to standardize Pokémon names (lowercase)

```python
df.index = df.index.str.lower()
```

That is it.
No `.loc`, no `.reset_index`, no hacks.

---

## If you *actually* want `Name` as a column again

Only do this if you **need** `Name` as data (joins, ML features, etc.).

```python
df = df.reset_index()
df["Name"] = df["Name"].str.lower()
df = df.set_index("Name")
```

This round-trip is intentional and explicit.

---

## Why your original line can NEVER work

```python
df["Name"] = df["Name"].str.lower()
```

Fails because:

* `df["Name"]` → Pandas searches **columns only**
* `Name` lives in the **index**
* Result → `KeyError`

This is the same as asking for a variable that does not exist.

---

## One-line rule (write this in your notes)

> If a column is passed as `index_col`, it no longer exists in `df.columns`; it must be accessed via `df.index`.

---

## Extra sanity check (recommended habit)

Before touching any column:

```python
print(df.columns)
print(df.index.name)
```

---

## **4. `str.strip()`**

```python
df["Last_Name"] = df["Last_Name"].str.lstrip("...")
```

### What `"..."` actually means here

It is **not** a number.
It is **not** a count.
It is **not** “remove three dots”.

`lstrip(chars)` treats its argument as a **set of characters**, not a string sequence.

So this line means:

> “From the **left side only**, remove **any combination** of the characters `'.'` until a non-dot is found.”

### Concrete examples (no ambiguity)

```python
"...Smith"     -> "Smith"
"..Smith"      -> "Smith"
".Smith"       -> "Smith"
"....Smith"    -> "Smith"
". . .Smith"   -> "Smith"   # spaces stop removal
```

It removes **dots only**, not exactly three of them.

---

### Now your question:

> “what does this 123 do ? can i write 1234”

If you do this:

```python
df["Last_Name"].str.lstrip("123")
```

It means:

> Remove **any leading `1`, `2`, or `3` characters**, in any order, until something else appears.

Examples:

```python
"123John"   -> "John"
"321John"   -> "John"
"11123John"-> "John"
"312John"  -> "John"
"4John"    -> "4John"   # stops immediately
```

If you write:

```python
.str.lstrip("1234")
```

Now it removes **1, 2, 3, or 4** from the left.

So yes, you *can* write `1234`, but **it does NOT mean “remove 1234 as a number”**.

---

### Critical mistake people make (avoid this)

❌ Expecting sequence removal:

```python
"123John".lstrip("12")   # NOT "3John"
```

Actual result:

```python
"John"
```

Because it removes **any of `{1,2}` repeatedly**, not `"12"` as a block.

---

### If you want to remove an exact prefix (correct way)

Use regex:

```python
df["Last_Name"] = df["Last_Name"].str.replace(r"^\.\.\.", "", regex=True)
```

or for digits:

```python
df["Last_Name"] = df["Last_Name"].str.replace(r"^123", "", regex=True)
```

---
