
# 0. Big picture (before math)

A **Decision Tree** does **NOT**:

* optimize weights
* use gradients
* use learning rate
* move downhill like logistic regression

Instead, it does this:

> **Repeatedly ask the best possible yes/no question on the data to reduce confusion as fast as possible.**

That’s it.

Everything else (entropy, gini, information gain) exists **only** to answer one question:

> ❓ *“Which split makes the data most pure?”*

---

# 1. What “learning” means in a Decision Tree

In logistic regression, learning =
👉 *adjusting weights to reduce loss*

In decision trees, learning =
👉 *choosing splits that reduce class mixing*

No weights.
No continuous optimization.
Only **greedy decisions**.

---

# 2. Start from raw data (Iris-like)

Imagine **only 10 samples** (simplified):

| Sample | Petal length | Class      |
| ------ | ------------ | ---------- |
| 1      | 1.4          | Setosa     |
| 2      | 1.3          | Setosa     |
| 3      | 1.5          | Setosa     |
| 4      | 4.7          | Versicolor |
| 5      | 4.5          | Versicolor |
| 6      | 4.9          | Versicolor |
| 7      | 5.8          | Virginica  |
| 8      | 6.1          | Virginica  |
| 9      | 5.9          | Virginica  |
| 10     | 6.0          | Virginica  |

Before splitting:

* Setosa = 3
* Versicolor = 3
* Virginica = 4

Mixed classes = **confusion**

The tree wants to **separate them cleanly**.

---

# 3. How does the tree measure “confusion”?

This is where **entropy** and **Gini impurity** come in.

They answer:

> “How mixed are the classes in this node?”

---

## 3.1 Entropy (concept first)

Entropy measures **uncertainty**.

* Pure node (only one class) → entropy = **0**
* Fully mixed → entropy = **high**

Formula:

$$
\text{Entropy} = - \sum p_i \log_2(p_i)
$$

Where:

* (p_i) = fraction of class (i)

---

### Entropy of root node

Class probabilities:

* Setosa = 3/10
* Versicolor = 3/10
* Virginica = 4/10

$$
H = -\Big(
0.3\log_2 0.3
\;+\; 0.3\log_2 0.3
\;+\; 0.4\log_2 0.4
  \Big)
  $$

≈ **1.57 bits**

👉 High entropy → very mixed

---

## 3.2 Gini Impurity (used by sklearn by default)

Simpler than entropy.

Formula:

$$
\text{Gini} = 1 - \sum p_i^2
$$

Same probabilities:

$$
G = 1 - (0.3^2 + 0.3^2 + 0.4^2)
= 1 - (0.09 + 0.09 + 0.16)
= 0.66
$$

Interpretation:

* 0 → pure
* higher → more mixed

---

# 4. Trying a split (this is the CORE)

Suppose the tree asks:

> **Is petal length ≤ 2.0 ?**

This creates **two child nodes**.

---

## Left child (≤ 2.0)

Samples: 1,2,3
All **Setosa**

* Entropy = 0
* Gini = 0

Perfectly pure.

---

## Right child (> 2.0)

Samples: 4–10
Classes:

* Versicolor = 3
* Virginica = 4

Probabilities:

* 3/7, 4/7

Gini:
$$
1 - (3/7)^2 - (4/7)^2 ≈ 0.49
$$

---

## Weighted impurity after split

Tree computes **weighted average**:

$$
\text{Gini}_{split}
= \frac{3}{10}(0) + \frac{7}{10}(0.49)
= 0.343
$$

---

# 5. Information Gain (decision rule)

The tree compares:

$$
\text{Gain} = \text{Impurity before} - \text{Impurity after}
$$

$$
= 0.66 - 0.343 = 0.317
$$

**Big reduction in confusion → good split**

---

# 6. The greedy algorithm (important)

The tree:

1. Tries **every feature**
2. Tries **many threshold values**
3. Computes impurity reduction
4. Picks the **best split**
5. Never revisits that decision

This is why decision trees are called **greedy**.

No global optimization like gradient descent.

---

# 7. Recursive splitting (tree growth)

After the first split:

* Left node is pure → stop
* Right node still mixed → split again

Example:

> petal length ≤ 5.0 ?

Eventually:

* Leaves become pure
* Or stopping criteria kicks in

---

# 8. Stopping criteria (why trees overfit)

Tree stops when:

* node is pure
* max_depth reached
* min_samples_split not satisfied

In your code:

```python
DecisionTreeClassifier(random_state=42)
```

Defaults:

* `max_depth=None` → tree grows **until pure**
* This often **overfits**

---

# 9. Prediction phase (very important)

Prediction is **dead simple**.

For one sample:

```text
petal length = 5.6
```

Tree:

```
petal length ≤ 2.0? → NO
petal length ≤ 5.0? → NO
→ Virginica
```

No probability computation unless asked.
No loss.
No gradients.

---

# 10. Multi-class vs binary

Binary classification:

* 2 classes → impurity simpler

Multi-class (Iris has 3):

* Same formulas
* More terms in entropy / gini
* Nothing fundamentally changes

---

# 11. Regression Trees (brief but exact)

Regression trees **don’t use entropy or gini**.

They minimize **variance / MSE**.

For a node:
$$
\text{Variance} = \frac{1}{n}\sum (y_i - \bar y)^2
$$

Split chosen to **reduce variance**, not class impurity.

---

# 12. Direct mapping to your code

```python
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

Internally:

* Computes Gini impurity
* Tries all splits
* Greedily builds tree
* Stores thresholds + feature indices

```python
plot_tree(clf, ...)
```

Visualizes:

* thresholds
* gini values
* samples per node
* predicted class

---

# 13. One-line intuition (lock this in)

> Logistic regression **pushes weights down a slope**
> Decision trees **cut the data to reduce confusion**

---
