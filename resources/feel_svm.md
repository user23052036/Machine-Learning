
# Concept 1 — **The Margin (just this, nothing else)**

**Plain language.**
Imagine two classes on a sheet of paper — red on the left, blue on the right. Draw a straight line that separates them. Now draw two parallel lines: one touching the closest red point, the other touching the closest blue point, with the separating line exactly halfway between them. The distance between those two parallel lines is the **margin**.

Why it matters: the margin is the classifier’s safety buffer. A wider margin gives more robustness to small perturbations in future data. SVM chooses the separating line that makes that buffer as wide as possible.

**Visual idea (ASCII):**

```
red   red   |   <---- margin ---->   |   blue   blue
   x   x    |                       |     o   o
     x      |                       |      o
```

**Crisp definition (one sentence).**
Margin = perpendicular distance between the nearest point of class A and the nearest point of class B, measured across the decision boundary.

**Important distinction.**

* The *decision boundary* is a center hyperplane (where the classifier output is 0).
* The *margin* is the band (two parallel hyperplanes) around that boundary.

---

# Concept 2 — **The Equation of the Decision Boundary**

### 1. The boundary (2D)

Any straight line in 2D can be written as
$$
w_1 x_1 + w_2 x_2 + b = 0,
$$
or compactly
$$
w \cdot x + b = 0,
$$
where (x=(x_1,x_2)), (w=(w_1,w_2)), and (b) is the bias.

### 2. The signed score

For any point (x) compute
$$
f(x) = w \cdot x + b.
$$

* If (f(x) > 0) → point lies on one side.
* If (f(x) < 0) → point lies on the other side.
* If (f(x) = 0) → point is on the boundary.

Classifier:
$$
\hat{y} = \mathrm{sign}(w \cdot x + b).
$$

### 3. The geometry of (w)

* (w) is **normal (perpendicular)** to the decision boundary.
* Direction of (w) determines which side is positive; magnitude of (w) controls how rapidly the score changes across space.

### 4. Role of (b)

* (b) shifts the boundary (moves it away from the origin).
* (w) controls orientation; (b) controls position.

### 5. Small numeric example

Let (w=(1,0), b=-2). Boundary:
$$
x_1 = 2.
$$
Point ((4,5)): (f=1\cdot 4 + 0\cdot 5 - 2 = 2 > 0).
Point ((1,3)): (f=1 -2 = -1 < 0).

**Takeaway:** decision boundary is (w\cdot x + b = 0); sign of (f(x)) gives class.

---

# Concept 3 — **Distance From a Point to the Decision Boundary**

We want the perpendicular (shortest) distance from a point (x) to the hyperplane (w \cdot x + b = 0).

For a 2D line (ax + by + c = 0), distance from ((x_0,y_0)) is
$$
d = \frac{|a x_0 + b y_0 + c|}{\sqrt{a^2 + b^2}}.
$$

Translating to SVM notation ((a,w_1); (b,w_2); (c,b)) gives
$$
\boxed{ \displaystyle d(x) = \frac{|w \cdot x + b|}{|w|} }
$$
where
$$
|w| = \sqrt{w_1^2 + w_2^2 + \dots }.
$$

**Why the denominator?** Multiplying (w,b) by a scalar doesn’t move the geometric boundary but scales the numerator. Dividing by (|w|) cancels that scaling and returns geometric distance.

**Numeric check:** (w=(1,1), b=0), point (x=(3,3)):
$$
w\cdot x + b = 6,\quad |w|=\sqrt{2},\quad d=\frac{6}{\sqrt{2}}\approx 4.24.
$$

**Key fact:** distance = (\dfrac{|w\cdot x + b|}{|w|}). This denominator ties margin to (|w|).

---

# Concept 4 — **What the Margin Really Is (using the distance formula)**

We use the distance formula to define the margin.

* Let (x^+) be the nearest positive point to the boundary and (x^-) be the nearest negative point.
* Margin is the sum of their perpendicular distances to the decision boundary:
  $$
  \text{margin} = \frac{|w\cdot x^+ + b|}{|w|} + \frac{|w\cdot x^- + b|}{|w|}.
  $$

However, the equation (w\cdot x + b = 0) is scale-invariant: scaling ((w,b)) by any positive constant leaves the geometric line unchanged but rescales numerators. To fix units we adopt the standard SVM convention:

**±1 scaling convention:** scale (w,b) so that
$$
w\cdot x^+ + b = +1,\qquad w\cdot x^- + b = -1.
$$
(These two equalities hold for the support vectors.)

Then each support vector’s distance to the boundary is (\dfrac{1}{|w|}), hence the total margin is:
$$
\boxed{ \displaystyle \text{margin} = \frac{2}{|w|} }.
$$

Thus maximizing margin ⇔ minimizing (|w|). For calculus convenience we minimize (\tfrac12|w|^2) instead.

---

# Concept 5 — **The Classification Constraints (hard-margin SVM)**

If we adopt the ±1 convention, to ensure all points are correctly classified and outside the margin we require:

* For positive points ((y_i=+1)): (w\cdot x_i + b \ge 1).
* For negative points ((y_i=-1)): (w\cdot x_i + b \le -1).

Combine both as:
$$
\boxed{ \displaystyle y_i \big(w\cdot x_i + b\big) \ge 1 \quad \text{for all } i.}
$$

Full hard-margin optimization:
$$
\boxed{ \displaystyle \min_{w,b}\ \tfrac12 |w|^2
\quad\text{subject to}\quad
y_i\big(w\cdot x_i + b\big)\ge 1\ \forall i. }
$$

This is a constrained convex optimization problem — we cannot simply do unconstrained gradient descent; we use Lagrange multipliers.

---

# Concept 6 — **Lagrange multipliers for the SVM constraints**

Form the Lagrangian by introducing one multiplier (\alpha_i \ge 0) per constraint:
$$
L(w,b,\alpha)
= \tfrac12|w|^2

* \sum_{i=1}^n \alpha_i\big(y_i(w\cdot x_i + b) - 1\big).
  $$

We look for a saddle point: minimize (L) over ((w,b)) and maximize over (\alpha\ge 0).

Stationarity conditions (derivatives w.r.t. primal variables set to zero):

1. Derivative w.r.t. (w):
   $$
   \frac{\partial L}{\partial w}
   = w - \sum_{i=1}^n \alpha_i y_i x_i = 0
   \quad\Longrightarrow\quad
   \boxed{ \displaystyle w = \sum_{i=1}^n \alpha_i y_i x_i }.
   $$

2. Derivative w.r.t. (b):
   $$
   \frac{\partial L}{\partial b}
   = -\sum_{i=1}^n \alpha_i y_i = 0
   \quad\Longrightarrow\quad
   \boxed{ \displaystyle \sum_{i=1}^n \alpha_i y_i = 0 }.
   $$

**Interpretation:** the optimal (w) is a weighted combination of the training examples; points with (\alpha_i=0) don’t contribute (they are irrelevant to the final classifier). Those with (\alpha_i>0) are the support vectors.

---

# Concept 7 — **Deriving the Dual Problem**

Start with the Lagrangian:
$$
L(w,b,\alpha)
= \tfrac12|w|^2

* \sum_i \alpha_i y_i (w\cdot x_i)
* b\sum_i \alpha_i y_i

- \sum_i \alpha_i.
  $$

Use (\sum_i \alpha_i y_i = 0) to drop the (b) term. Substitute (w=\sum_j \alpha_j y_j x_j) and simplify:

* Note:
  $$
  |w|^2 = \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i\cdot x_j).
  $$
* Also:
  $$
  \sum_i \alpha_i y_i (w\cdot x_i) = |w|^2.
  $$

After substitution and simplification the dual objective (depends only on (\alpha)) is:
$$
\boxed{ \displaystyle
\max_{\alpha}\ \sum_{i=1}^n \alpha_i
;-; \tfrac12 \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (x_i\cdot x_j)
}
$$
subject to
$$
\alpha_i \ge 0,\qquad \sum_{i=1}^n \alpha_i y_i = 0.
$$

This is the classic SVM dual.

---

# Concept 8 — **Why the Dual Formulation Matters**

1. **Support vectors appear naturally.** From (w=\sum_i \alpha_i y_i x_i), many (\alpha_i) are zero at the optimum; only support vectors (nonzero (\alpha_i)) define (w).

2. **Data appears only as dot products.** The dual objective uses (x_i\cdot x_j) everywhere — that observation enables the kernel trick later.

3. **Computational reasons.** When feature dimension is huge but number of samples is moderate, solving the dual can be more efficient.

4. **Kernelization.** Replacing (x_i\cdot x_j) with (K(x_i,x_j)) lets SVM operate in an implicit high-dimensional feature space without explicitly computing (\phi(x)).

---

# Concept 9 — **Soft-Margin SVM (real world)**

Hard margin requires perfect separability. In practice we allow violations via slack variables (\xi_i \ge 0):
$$
y_i(w\cdot x_i + b)\ \ge\ 1 - \xi_i.
$$

Penalize violations in the objective:
$$
\boxed{ \displaystyle
\min_{w,b,\xi} \ \tfrac12|w|^2 + C\sum_{i=1}^n \xi_i
\quad\text{subject to}\quad
y_i(w\cdot x_i + b)\ge 1-\xi_i,\ \xi_i\ge 0.
}
$$

* If (y_i(w\cdot x_i + b) \ge 1) then (\xi_i=0).
* If (0<y_i(w\cdot x_i + b)<1) then (0<\xi_i<1) (inside margin but correct side).
* If (y_i(w\cdot x_i + b) < 0) then (\xi_i>1) (misclassified).

This is equivalent to minimizing hinge loss with regularization:
$$
\sum_{i=1}^n \max(0,,1 - y_i(w\cdot x_i + b)) + \frac{\lambda}{2}|w|^2,
$$
with (C) and (\lambda) linked by convention.

**Interpretation of (C):**

* Large (C): penalize violations heavily → small margin, risk of overfitting.
* Small (C): allow violations → wider margin, more bias.

---

# Concept 10 — **Why in soft margin we get** $$0 \le \alpha_i \le C$$

When you form the Lagrangian for the soft-margin problem you introduce multipliers (\alpha_i \ge 0) for the margin constraints and (\mu_i \ge 0) for (\xi_i \ge 0). Part of the Lagrangian derivative calculations give:
$$
\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \quad\Longrightarrow\quad C = \alpha_i + \mu_i.
$$
Because (\mu_i \ge 0) and (\alpha_i \ge 0), it follows
$$
\boxed{ \displaystyle 0 \le \alpha_i \le C }.
$$

**Regimes:**

* (\alpha_i = 0): point outside margin (irrelevant).
* (0<\alpha_i<C): point exactly on margin (clean support vector).
* (\alpha_i = C): point inside margin or misclassified (error support vector).

Intuition: (C) caps how much influence any one point can have on the boundary.

---

# Concept 11 — **The Kernel Trick**

We want nonlinear boundaries. Idea: map (x) to a higher-dimensional feature vector (\phi(x)) and do a linear SVM there:
$$
f(x)=w\cdot\phi(x)+b.
$$
The dual only depends on inner products (\phi(x_i)\cdot\phi(x_j)). If we define a kernel function
$$
K(x_i,x_j)=\phi(x_i)\cdot\phi(x_j),
$$
we can substitute (K) for the dot product in the dual and never compute (\phi(x)) explicitly. That gives nonlinear decision boundaries in original space.

**Common kernels:**

* Linear: (K(x,z)=x\cdot z)
* Polynomial: (K(x,z)=(\gamma x\cdot z + r)^d)
* RBF/Gaussian: (K(x,z)=\exp(-\gamma|x-z|^2)) — infinite-dimensional feature map

**Prediction with kernel** (after solving dual):
$$
f(x)=\sum_{i\in SV}\alpha_i y_i K(x_i,x) + b.
$$

---

# Hinge loss vs Logistic loss (mathematical comparison)

Let the margin score be
$$
z = y(w\cdot x + b).
$$

## Hinge loss

$$
L_{\text{hinge}}(z) = \max(0,,1 - z) =
\begin{cases}
1 - z & z < 1,[4pt]
0 & z \ge 1.
\end{cases}
$$

* Zero for (z\ge 1) (no loss if outside margin).
* Linear penalty when (z<1).
* Not differentiable at (z=1).

## Logistic loss

$$
L_{\text{log}}(z)=\log(1+e^{-z}).
$$

* Smooth, always positive, never exactly zero.
* Decreases exponentially for large (z).

## Behavior

* Misclassified (z<0): hinge grows linearly; logistic grows logarithmically.
* Inside margin (0<z<1): both penalize; hinge is linear, logistic curved.
* Confidently correct (z\gg1): hinge loss = 0, logistic loss > 0 (tiny).

## Gradients (w.r.t. (z))

* Hinge (subgradient): (\frac{dL}{dz} = -1) for (z<1), and 0 for (z>1) (undefined but usable as subgradient at (z=1)).
* Logistic: (\frac{dL}{dz} = -\frac{1}{1+e^{z}}) (smooth, strictly between (-1) and 0).

**Consequences:**

* Hinge gives sparse solutions (only support vectors carry gradient once margin satisfied).
* Logistic uses all points; no true sparsity.
* SVM is margin-focused; logistic aims at probabilistic outputs (via log-likelihood).

---

# Appendix — Short glossary & reminders

* **Support vectors:** training examples with (\alpha_i>0). They determine the boundary.
* **Margin width:** (2/|w|).
* **Hard-margin SVM:** requires perfect separability (no slack).
* **Soft-margin SVM:** includes slack (\xi_i) and tradeoff parameter (C).
* **Dual variable (\alpha_i):** Lagrange multiplier connected to how strongly the (i)-th constraint influences the optimum.
* **KKT complementary slackness (summary):** (\alpha_i\big(y_i(w\cdot x_i + b) - 1 + \xi_i\big) = 0) (for soft margin including (\xi_i)). This enforces either slack active or (\alpha_i=0).

---

Good. Here is your **condensed one-page SVM cheat sheet** — no storytelling, no repetition, only the core structure.

You can paste this directly into your notebook.

---

# 📌 Support Vector Machine — One-Page Cheat Sheet

---

# 1️⃣ Decision Boundary

### Hyperplane

$$
w \cdot x + b = 0
$$

### Prediction

$$
\hat{y} = \text{sign}(w \cdot x + b)
$$

* (w) → normal vector
* (b) → shift
* (|w|) → controls margin size

---

# 2️⃣ Distance From a Point to Boundary

$$
\text{distance} = \frac{|w \cdot x + b|}{|w|}
$$

---

# 3️⃣ Margin (Hard Margin SVM)

Using ±1 scaling:

Support vectors lie on:

$$
w \cdot x + b = +1
$$

$$
w \cdot x + b = -1
$$

### Margin width:

$$
\boxed{
\text{margin} = \frac{2}{|w|}
}
$$

Maximize margin
⇔ Minimize ( |w| )
⇔ Minimize ( \frac{1}{2}|w|^2 )

---

# 4️⃣ Hard-Margin Optimization

### Objective:

$$
\min_{w,b} \frac{1}{2}|w|^2
$$

### Constraint:

$$
y_i (w \cdot x_i + b) \ge 1
$$

---

# 5️⃣ Lagrangian (Hard Margin)

$$
L(w,b,\alpha)
=

\frac{1}{2}|w|^2

\sum_i \alpha_i
\big(y_i(w \cdot x_i + b) - 1\big)
$$

Conditions:

$$
w = \sum_i \alpha_i y_i x_i
$$

$$
\sum_i \alpha_i y_i = 0
$$

$$
\alpha_i \ge 0
$$

---

# 6️⃣ Dual Problem

$$
\max_{\alpha}
\sum_i \alpha_i
=

\frac{1}{2}
\sum_i \sum_j
\alpha_i \alpha_j y_i y_j
(x_i \cdot x_j)
$$

Subject to:

$$
\alpha_i \ge 0
$$

$$
\sum_i \alpha_i y_i = 0
$$

---

# 7️⃣ Final Classifier (Dual Form)

After solving α:

$$
w = \sum_{i \in SV} \alpha_i y_i x_i
$$

Prediction:

$$
f(x)
=

\sum_{i \in SV}
\alpha_i y_i (x_i \cdot x)
+
b
$$

Only support vectors matter.

---

# 8️⃣ Soft Margin SVM

### Add slack:

$$
y_i(w \cdot x_i + b) \ge 1 - \xi_i
$$

### Objective:

$$
\min
\frac{1}{2}|w|^2
+
C \sum_i \xi_i
$$

Equivalent to hinge form:

$$
\sum_i
\max(0, 1 - y_i(w \cdot x_i + b))
+
\frac{\lambda}{2}|w|^2
$$

---

# 9️⃣ Important Result (Soft Margin)

$$
\boxed{
0 \le \alpha_i \le C
}
$$

* ( \alpha_i = 0 ) → outside margin
* ( 0 < \alpha_i < C ) → on margin
* ( \alpha_i = C ) → inside margin / misclassified

---

# 🔟 Hinge Loss vs Logistic Loss

Let:
$$
z = y(w \cdot x + b)
$$

### Hinge:

$$
L = \max(0, 1 - z)
$$

### Logistic:

$$
L = \log(1 + e^{-z})
$$

| Property                  | Hinge       | Logistic |
| ------------------------- | ----------- | -------- |
| Zero region               | Yes (z ≥ 1) | No       |
| Smooth                    | No          | Yes      |
| Uses only support vectors | Yes         | No       |
| Probabilistic             | No          | Yes      |

---

# 1️⃣1️⃣ Kernel Trick

Replace dot product:

$$
x_i \cdot x_j
$$

with:

$$
K(x_i, x_j)
$$

Classifier becomes:

$$
f(x)
=

\sum_{i \in SV}
\alpha_i y_i
K(x_i, x)
+
b
$$

Common kernels:

* Linear: ( x \cdot z )
* Polynomial: ( (x \cdot z + c)^d )
* RBF: ( \exp(-\gamma |x - z|^2) )

---

# 🔥 Core Insight

SVM does NOT fit data.

It maximizes geometric margin:

$$
\text{margin} = \frac{2}{|w|}
$$

Only points touching the margin determine the solution.

Everything else disappears.

---