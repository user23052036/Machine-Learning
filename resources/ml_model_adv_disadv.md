
**Linear Regression** — *Supervised (regression)*<br>
Advantages:

1. Very simple to implement and interpret (coefficients = feature effect).
2. Fast to train and predict; works well when relationship is approximately linear.
3. Coefficients give easy statistical inference (p-values, confidence intervals).

Disadvantages:

1. Assumes linear relationship — underfits if real relationship is nonlinear.
2. Sensitive to outliers and multicollinearity between features.
3. Poor performance when features/targets are non-Gaussian or heteroscedastic.

Failure modes / when not to use: when target vs features are non-linear (unless you engineer features), when outliers dominate, when important interactions or thresholds exist.

---

**Logistic Regression** — *Supervised (classification)*<br>
Advantages:

1. Simple, fast, outputs calibrated probabilities (with proper regularization).
2. Works well for linearly separable classes and low-dimensional data.
3. Regularization (L1/L2) reduces overfitting and performs feature selection.

Disadvantages:

1. Can't capture complex non-linear decision boundaries without feature transforms.
2. Sensitive to outliers and heavily imbalanced classes (requires class weighting).
3. Requires careful feature scaling and feature engineering for interactions.

Failure modes: overlapping classes with non-linear boundary, extreme class imbalance, high dimensional sparse data without regularization.

---

**Polynomial Regression** — *Supervised (regression)*<br>
Advantages:

1. Extends linear model to capture simple, low-degree nonlinearity via polynomial features.
2. Still interpretable (with transformed features).
3. Easy to implement with linear solvers.

Disadvantages:

1. Degree selection critical — high degree → severe overfitting; low degree → underfit.
2. Polynomial features explode feature space (multicollinearity, numerical instability).
3. Poor extrapolation outside training range.

Failure modes: high-degree polynomials oscillate (Runge phenomenon); sensitive to scaling — always standardize and regularize.

---

**Support Vector Machine (SVM)** — *Supervised (classification / regression)*<br>
Advantages:

1. Powerful with kernel trick — can learn complex boundaries (RBF, polynomial kernels).
2. Effective in high-dimensional spaces; margin maximization often improves generalization.
3. Robust to some overfitting via margin/C parameter.

Disadvantages:

1. Computationally expensive on large datasets (training scales poorly).
2. Requires careful kernel and hyperparameter tuning (C, gamma).
3. Not naturally probabilistic (needs calibration); sensitive to feature scaling.

Failure modes: huge datasets (use linear SVM or other methods), poorly chosen kernel, noisy labels reduce margin effectiveness.

---

**K-Nearest Neighbors (KNN)** — *Supervised (classification / regression)*<br>
Advantages:

1. Extremely simple, no training cost (instance-based).
2. Can model arbitrary non-linear decision boundaries given enough data.
3. Works out of the box for mixed tasks; easy to understand.

Disadvantages:

1. Prediction cost is high (O(n)) and memory-intensive — poor for large datasets.
2. Performance collapses in high dimensions (curse of dimensionality).
3. Sensitive to scaling and irrelevant/noisy features; requires distance metric tuning and k choice.

Failure modes: sparse high-dimensional spaces, massively imbalanced classes, large datasets without approximate nearest-neighbor indexing.

---

**Naive Bayes** — *Supervised (classification)*<br>
Advantages:

1. Extremely fast to train and predict; works well with small data.
2. Performs surprisingly well on text/classic categorical problems.
3. Simple, probabilistic outputs and straightforward smoothing fixes (Laplace).

Disadvantages:

1. Strong conditional independence assumption — often violated.
2. Poor calibration when independence fails; zero-frequency problem without smoothing.
3. Not expressive — cannot model feature interactions well.

Failure modes: highly correlated features, continuous features without proper modeling (Gaussian assumption break), when probabilities need to be accurate.

---

**Random Forests** — *Supervised (classification / regression)*<br>
Advantages:

1. Ensemble reduces overfitting vs single trees; strong out-of-the-box performance.
2. Handles numeric and categorical features, robust to outliers and missing values.
3. Provides feature importance and reasonable default hyperparameters.

Disadvantages:

1. Less interpretable than a single tree; large memory and model size.
2. Can still overfit noisy data; biased when features have many categories.
3. Poor extrapolation for regression (like trees in general).

Failure modes: extremely high-dimensional sparse data (consider boosting/sparse methods), heavy class imbalance, need many trees for stability (compute cost).

---

**ID3 / Decision Tree (ID3)** — *Supervised (classification)*<br>
Advantages:

1. Easy to interpret and visualize; handles categorical data naturally.
2. No need for scaling or heavy preprocessing.
3. Fast to train on moderate datasets.
4. Not sensitive to outliers.
5. can be used for both classification and regression.

Disadvantages:

1. Prone to overfitting (greedy splits); high variance (unstable to small data changes).
2. Requires pruning or depth control; continuous features need discretization or split criteria.
3. Biased toward features with many levels.
4. Training time is relatively heigher.

Failure modes: noisy labels, small datasets (overfitting), unbalanced classes — require pruning, ensemble methods to stabilize.

---

**K-Means** — *Unsupervised (clustering)*<br>
Advantages:

1. Simple, scalable (linear per iteration) and fast for large datasets.
2. Works well for roughly spherical, equal-sized clusters.
3. Easy to implement and interpret cluster centers.

Disadvantages:

1. Must choose k; sensitive to initialization (k-means++ helps).
2. Assumes convex/spherical clusters and equal variance — fails on arbitrary shapes.
3. Sensitive to outliers and feature scaling.

Failure modes: clusters of different densities/sizes/shapes, high dimensionality (use dimensionality reduction first), presence of outliers.

---

**K-Medians** — *Unsupervised (clustering)*<br>
Advantages:

1. Uses medians (L1), so more robust to outliers than k-means.
2. Good when median is a better center measure (categorical ordinal data).
3. Simple extension of k-means logic.

Disadvantages:

1. Typically slower (non-differentiable objective), harder to optimize.
2. Still requires k and assumes cluster shape similar to k-means.
3. Less common tooling and less interpretability for some data types.

Failure modes: high computation on big datasets, complex cluster shapes, variable cluster densities.

---

**Hierarchical Clustering** (agglomerative / divisive) — *Unsupervised (clustering)*<br>
Advantages:

1. No need to pre-specify k (dendrogram lets you pick cut level).
2. Reveals multi-scale structure and is highly interpretable for small datasets.
3. Works with any distance/linkage metric (flexible).

Disadvantages:

1. Computationally expensive (O(n²) time & memory) — not scalable to very large datasets.
2. Sensitive to linkage choice and noise; early merging/splitting decisions are irrevocable.
3. Hard to tune for large, high-dimensional data.

Failure modes: large datasets, noisy features, inappropriate linkage metric (single vs complete vs average) yields very different trees.

---

**DBSCAN (Density-Based Spatial Clustering)** — *Unsupervised (clustering)*<br>
Advantages:

1. Finds arbitrarily shaped clusters and identifies outliers as noise.
2. No need to predefine cluster count k.
3. Good when clusters are well separated by density.

Disadvantages:

1. Sensitive to hyperparameters (epsilon, minPts); difficult when densities vary.
2. Struggles in high dimensions (distance metrics degrade).
3. Not ideal for clusters with varying densities or very noisy datasets.

Failure modes: varying density clusters, high-dimensional sparse data, choosing eps incorrectly (merging or splitting clusters).

---

**Apriori (Association Rule Mining)** — *Unsupervised / descriptive (association rules)*<br>
Advantages:

1. Produces easy-to-understand association rules (if-then) for transactional data.
2. Useful for market-basket analysis, recommendations, exploratory insight.
3. Well-studied with clear support/confidence metrics.

Disadvantages:

1. Computationally expensive — combinatorial explosion of candidate itemsets.
2. Produces many trivial or spurious rules; needs careful thresholding and post-filtering.
3. Requires transactional categorical data and enough support to be meaningful.

Failure modes: very large itemsets / catalogs, low support items, noisy or continuous data without discretization; use FP-Growth or constraint-based methods to scale.

---
