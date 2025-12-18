
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

## **2. **
