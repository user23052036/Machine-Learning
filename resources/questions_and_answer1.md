
# 1 — What does `stratify=y` do in `train_test_split`?

**Short answer:** it forces the train and test splits to keep the same class proportions as the original `y`.

**Why:** If your dataset has class imbalance (e.g., 90% class A, 10% class B), a random split might, by chance, put very few (or zero) examples of the minority class into the test or train set. `stratify=y` preserves class ratios so both sets represent the original distribution.

**Example (scikit-learn):**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Effect illustration (toy counts):**
Original y: class0=900, class1=100 (10% class1)
After `stratify=y` with `test_size=0.2`: test will have ~20% of each class → class1 in test ≈ 20. If you did not stratify, you might randomly get only 5 class1 in test or even 0 for tiny datasets.

**Pitfall:** stratify requires at least one sample of every class in each split. If a class has fewer examples than the number of folds or the split would make 0 samples for a class, you’ll get an error.

---

# 2 — How is Stochastic Gradient Descent (SGD) different from (Batch) Gradient Descent?

**Definitions & update rules**

* **Batch (Vanilla) Gradient Descent:** compute gradient over the *entire* training set, then update weights once per iteration (epoch).

  * Update: (\theta := \theta - \eta \frac{1}{N}\sum_{i=1}^N \nabla_\theta L(x_i, y_i))
* **Stochastic Gradient Descent (SGD):** update weights using *one* sample at a time (or one label) — noisy but cheap per update.

  * Update per sample (i): (\theta := \theta - \eta \nabla_\theta L(x_i, y_i))
* **Mini-batch GD:** compromise — update per small batch of size `m` (typical in deep learning).

**Practical differences**

* **Speed & memory:** Batch GD needs full dataset in memory for each gradient computation; SGD/mini-batch are much faster per update and work with streaming/huge datasets.
* **Noise & convergence:** SGD’s noise helps escape shallow local minima and can lead to faster initial progress, but causes oscillation near optimum; requires careful learning-rate scheduling (decay).
* **Parallelism:** mini-batch enables vectorized GPU/BLAS operations — often the practical best choice.
* **Iterations:** SGD performs many more parameter updates per epoch (N updates) vs 1 update per epoch for batch GD.

**Pseudocode (SGD):**

```text
for epoch in range(epochs):
    shuffle(dataset)
    for x_i, y_i in dataset:
        g = gradient(theta, x_i, y_i)
        theta = theta - lr * g
    optionally: lr = lr * decay
```

**When to choose what**

* Small datasets where cost of full gradient is cheap → batch GD acceptable.
* Large datasets/streaming → SGD or mini-batch.
* Deep nets → mini-batch with Adam/SGD + momentum.

---

# 3 — Why split training set further into **train** and **validation**?

**Purpose:** the validation set is used for **model selection** and **hyperparameter tuning** (and early stopping). The test set is kept as a final, untouched estimate of generalization.

**Workflow:**

1. Split full data into train + test (e.g., 80/20 or 70/30) — *test is held out permanently*.
2. Within the training part, either:

   * split into train/validation (e.g., 60/20/20 total); **or**
   * use cross-validation (CV) on training data to pick hyperparameters.
3. After choosing hyperparameters with validation/CV, retrain on full training set (optionally train+validation) and evaluate once on test.

**Why not use test for tuning?** Because tuning on test leaks information about the test distribution into model selection, producing an optimistically biased estimate.

**Example choices:**

* Small dataset: use **k-fold CV** (e.g., 5-fold) instead of a fixed validation set. Consider **nested CV** for unbiased generalization error of model selection.
* Large dataset: fixed splits are fine (e.g., 70% train, 15% val, 15% test).

**Minimal robust change to most pipelines:** use `StratifiedKFold` or `train_test_split(..., stratify=y)` and keep test untouched until the very end.

---

# 4 — What is `SGDClassifier` and what is `warm_start`?

**`SGDClassifier` (scikit-learn):**

* A linear classifier (SVM/logistic/Perceptron style losses) trained with stochastic gradient descent.
* Supports `loss='hinge'` (linear SVM), `loss='log'` or `'log_loss'` (logistic regression), etc.
* Fast for large sparse data; supports `partial_fit` for true online learning.

**`warm_start` (general scikit-learn behavior):**

* `warm_start=True` tells the estimator to reuse the solution of the previous `.fit()` call and continue training, instead of reinitializing parameters.
* Useful if you want to incrementally increase `max_iter` or continue training with new data **via repeated fit calls**.
* **For incremental (streaming) learning, prefer `partial_fit`** because it’s explicit and safer.

**Concrete example: incremental learning with `partial_fit`:**

```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', max_iter=1, warm_start=True)  # warm_start optional
classes = np.unique(y)
for epoch in range(10):
    for X_batch, y_batch in stream_batches(X, y, batch_size=128):
        clf.partial_fit(X_batch, y_batch, classes=classes)
# partial_fit updates in place; no reinitialization.
```

**Caveat about `warm_start`:** if you call `.fit()` with `warm_start=True` but change critical hyperparameters (like penalty or loss), behavior can be surprising — parameters are reused but hyperparams inconsistent. `partial_fit` is clearer for streaming.

---

# 5 — Validation vs Testing — I’m confused

**Clear definitions:**

* **Training set:** used to **fit** model weights.
* **Validation set:** used to **select hyperparameters**, choose model architecture, or do early stopping. You may look at validation metrics repeatedly.
* **Test set:** used **only once** (or very rarely) after model selection to obtain the final unbiased performance estimate.

**Analogy:** Training = build the car; validation = test prototypes and choose the best; test = crash the final car once to report safety stats.

**Common mistakes (avoid these):**

* Tuning hyperparameters on test → test leakage → optimistic performance.
* Scaling or preprocessing *before* splitting (do scaling inside pipeline or fit scaler on train only then transform val/test).

**Good practice (scikit-learn):** build a `Pipeline` with preprocessing steps and pass the pipeline to `GridSearchCV` so scaling is done within CV folds (prevents leakage).

---

# 6 — What does `GridSearchCV` do?

**Short:** performs an exhaustive search over a grid of hyperparameter values using cross-validation and returns the best parameter combination.

**Key options:**

* `estimator` — the model or pipeline to tune.
* `param_grid` — dictionary of parameter names to lists of values.
* `cv` — cross-validation strategy (e.g., `5`, `StratifiedKFold(5)`).
* `scoring` — metric to optimize (`accuracy`, `f1`, `roc_auc` etc.)
* `refit=True` — after finding best params, refits estimator on full training set.

**Example (robust pipeline, classification):**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
])

param_grid = {
    'clf__loss': ['log_loss', 'hinge'],
    'clf__alpha': [1e-4, 1e-3, 1e-2],
    'clf__max_iter': [1000]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, refit=True)
gs.fit(X_train, y_train)   # NOTE: do this on training set (not test)
best_model = gs.best_estimator_
```

**Important cautions:**

* Use `Pipeline` so scaling and preprocessing happen inside CV folds (avoid leakage).
* For imbalanced data prefer `scoring='f1'` or `roc_auc` and use `StratifiedKFold`.
* Grid search is exhaustive — expensive. For larger hyperparameter spaces use `RandomizedSearchCV` or Bayesian optimization.

---

# Practical checklist (smallest changes to make your pipeline solid)

1. **If classification and imbalance exists → always use `stratify=y` in `train_test_split`.**
2. **Use a pipeline** (`StandardScaler` → model) and pass pipeline to `GridSearchCV` so preprocessing is inside folds.
3. **Use `StratifiedKFold` for classification** when doing CV.
4. **Reserve a single test set** and never use it during hyperparameter tuning; use validation or CV instead.
5. For large/streaming data prefer **`partial_fit`** (SGD) or mini-batch training rather than full batch GD.
6. **If you use `warm_start`,** be intentional — prefer `partial_fit` for incremental updates.

---

# Short examples / summary table

| Concept          |                                       One-line | Typical usage                                                    |
| ---------------- | ---------------------------------------------: | ---------------------------------------------------------------- |
| `stratify=y`     |               Keep class ratios same in splits | Use for classification, especially imbalanced classes            |
| Batch GD         |               Full dataset gradient per update | Small datasets or theoretical analysis                           |
| SGD / Mini-batch |                         Frequent noisy updates | Large datasets, neural nets — faster wall clock convergence      |
| `SGDClassifier`  |    scikit-learn linear models trained with SGD | Large sparse data, streaming via `partial_fit`                   |
| `warm_start`     | reuse previous fit when calling `.fit()` again | Continue training, but be careful — `partial_fit` usually better |
| Validation       |                           Tune hyperparameters | Use CV or holdout validation                                     |
| Test             |                      Final unbiased evaluation | Keep untouched until final evaluation                            |
| `GridSearchCV`   |       Exhaustive hyperparameter search with CV | Use with `Pipeline`, `StratifiedKFold`, and proper scoring       |

---

Excellent. This is the right thing to inspect.

Now let’s interpret what you’re actually seeing.

---

# Before Standardization

Example pixel vector:

```
[0, 0, 5, 13, 9, 1, ...]
```

Each value is:

* A grayscale intensity
* Range ≈ 0 to 16
* Raw feature values

Nothing fancy.

---

# After Standardization

Now you see values like:

```
[-0.8846, 1.5807, -1.0188, ...]
```

These are **z-scores**.

Each value now means:

[
z = \frac{x - \mu}{\sigma}
]

So:

* 0 → roughly average pixel value for that column
* Positive → above average brightness
* Negative → below average brightness
* Magnitude → how many standard deviations away

---

# Why Some Zeros Still Exist

You noticed:

```
0.0
```

That happens when:

* The original value equals the feature mean
* OR the column had zero variance (rare, but possible in some pixel positions)

---

# What Changed Conceptually?

Before:

* Feature scale was arbitrary (0–16)

After:

* Every pixel column:

  * Mean = 0
  * Std = 1

Now optimization treats all 64 dimensions equally.

---

# Important Insight

Standardization is done **per column**, not per image.

That means:

Each of the 64 pixel positions is scaled independently.

It does NOT normalize each image as a whole.

This is critical.

---

# Why Values Look “Random”

Because you’re no longer seeing pixel intensities.

You’re seeing:

“How unusual is this pixel compared to the average pixel in that location across the dataset?”

---

# The Real Purpose

SGD updates:

[
w := w - \eta \nabla L
]

If one feature has larger magnitude, it produces larger gradients.

Standardization prevents unstable updates.

---
