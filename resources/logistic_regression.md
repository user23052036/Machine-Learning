# Logistic Regression — Complete & Self-Explanatory Guide

This document explains Logistic Regression from first principles.
No LaTeX. No broken formulas. Renders correctly everywhere.

---

## 1. What is Logistic Regression?

Logistic Regression is a **binary classification algorithm**.

It predicts:
- Probability that output y = 1
- Then converts probability into a class (0 or 1)

Typical uses:
- Spam detection
- Disease prediction
- Fraud detection
- Pass / Fail decisions

---

## 2. Linear Model (Core Idea)

First, Logistic Regression computes a **linear score**:

z = w1*x1 + w2*x2 + ... + wn*xn + b

Where:
- w = weights
- x = input features
- b = bias
- z = raw score (can be any real number)

This is **NOT** a probability yet.

---

## 3. Sigmoid Function (Very Important)

To convert z into a probability, we use the **sigmoid function**.

Sigmoid formula (plain text):

sigmoid(z) = 1 / (1 + exp(-z))

What sigmoid does:
- Converts any number into range (0, 1)
- Makes output interpretable as probability

Examples:

z = -100 → sigmoid ≈ 0  
z = 0    → sigmoid = 0.5  
z = +100 → sigmoid ≈ 1  

---

## 4. From Probability to Class

Predicted probability:

p = sigmoid(z)

Decision rule (default):

- If p ≥ 0.5 → Class 1
- If p < 0.5 → Class 0

The value 0.5 corresponds to:
z = 0

So the **decision boundary** is:

w·x + b = 0

---

## 5. Why Not Use Linear Regression?

Linear regression:
- Output range: (-∞, +∞)
- Not suitable for probabilities

Logistic regression:
- Output range: (0, 1)
- Interpretable as probability
- Designed for classification

---

## 6. Loss Function (Binary Cross-Entropy)

We need a way to measure error.

Binary cross-entropy loss:

loss = - [ y*log(p) + (1-y)*log(1-p) ]

Where:
- y = true label (0 or 1)
- p = predicted probability

Why this loss?
- Strongly penalizes confident wrong predictions
- Comes from maximum likelihood theory

---

## 7. Training Using Gradient Descent

Goal:
Minimize average loss over all samples.

Key result (very important):

Gradient with respect to weights:

gradient_w = (1/N) * Xᵀ · (p - y)

Gradient with respect to bias:

gradient_b = mean(p - y)

Update rules:

w = w - learning_rate * gradient_w  
b = b - learning_rate * gradient_b  

This is why logistic regression is efficient and convex.

---

## 8. Regularization (Prevent Overfitting)

### L2 Regularization (Ridge)

Add penalty:

loss = original_loss + (lambda / 2) * sum(w²)

Effect:
- Shrinks weights
- Handles multicollinearity
- Improves generalization

Updated gradient:

gradient_w = gradient_w + lambda * w

---

### L1 Regularization (Lasso)

Penalty:

loss = original_loss + lambda * sum(|w|)

Effect:
- Drives some weights to zero
- Feature selection

Harder to optimize than L2.

---

## 9. From-Scratch Python Implementation (Stable)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000, reg_lambda=0.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.verbose = verbose

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.epochs):
            z = X.dot(self.w) + self.b
            p = sigmoid(z)

            error = p - y
            dw = (1/n_samples) * X.T.dot(error)
            db = np.mean(error)

            if self.reg_lambda > 0:
                dw += self.reg_lambda * self.w

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.verbose and i % (self.epochs // 10) == 0:
                loss = -np.mean(y*np.log(p+1e-12) + (1-y)*np.log(1-p+1e-12))
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 10. Usage Example

```python
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])

model = LogisticRegressionScratch(lr=0.5, epochs=2000, verbose=True)
model.fit(X, y)

print("Probabilities:", model.predict_proba(X))
print("Predictions:", model.predict(X))
```

---

## 11. Using scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

preds = clf.predict(X)
print("Accuracy:", accuracy_score(y, preds))
```

Notes:
- C is inverse regularization strength
- Smaller C → stronger regularization
- solver="saga" supports L1

---

## 12. Assumptions (Important)

Logistic regression assumes:
- Log-odds are linear in features
- Observations are independent
- No perfect multicollinearity

Violating these hurts performance.

---

## 13. Failure Modes & Fixes

Problem → Fix

- Predicts only one class  
  → Check class imbalance, use class_weight

- Loss becomes NaN  
  → Clip probabilities, lower learning rate

- Very slow training  
  → Scale features

- Overfitting  
  → Add L2 regularization

- Poor accuracy on nonlinear data  
  → Add polynomial features or change model

---

## 14. When Logistic Regression is a Bad Choice

- Highly nonlinear decision boundary
- Image / audio raw data
- Extremely imbalanced classes (without care)

Use trees, SVMs, or neural networks instead.

---

## 15. Key Takeaways

- Logistic Regression = Linear model + Sigmoid
- Outputs probabilities, not just classes
- Convex optimization → global minimum
- Strong baseline for classification

---
