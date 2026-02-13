import numpy as np


class Logistic_Regression:

    def __init__(self, learning_rate, no_of_iterations):
        self.lr = learning_rate
        self.iterations = no_of_iterations

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)   # numerical stability (ignore for now)
        return 1 / (1 + np.exp(-z))

    def cost(self, Y, y_hat):
        m = Y.shape[0]
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        return - (1/m) * np.sum(
            Y * np.log(y_hat) +
            (1 - Y) * np.log(1 - y_hat)
        )

    def fit(self, X, Y):
        m, n = X.shape

        # initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0

        # training history
        self.cost_history = []
        self.weight_history = []
        self.bias_history = []

        for i in range(self.iterations):
            
            # forward pass
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.sigmoid(z)
            
            # gradient descent
            dw = (1/m) * np.dot(X.T, (y_hat - Y))
            db = (1/m) * np.sum(y_hat - Y)

            # updated weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # recompute after update (correct cost logging)
            z_new = np.dot(X, self.weights) + self.bias
            y_hat_new = self.sigmoid(z_new)

            # storing history of cost, weights and bias
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)
            self.cost_history.append(self.cost(Y,y_hat_new))
            

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)
        return np.where(y_hat >= 0.5, 1, 0) # boundary condition when probability >= 50% class = 1



"""
Design Notes for This Logistic Regression Implementation
--------------------------------------------------------

This implementation intentionally avoids storing the training dataset
(X and Y) inside the model object after calling `fit()`.

Why?

1. Separation of Model Parameters and Data
   ----------------------------------------
   A trained machine learning model should depend only on its learned
   parameters (weights and bias). It should not depend on the dataset
   used during training.

   Storing X and Y inside the class tightly couples the model to the
   training data, which is poor design and reduces flexibility.

2. Memory Efficiency
   -------------------
   If the training dataset is large (e.g., millions of samples),
   storing it inside the model object wastes memory and can cause
   unnecessary duplication, especially if multiple model instances
   are created.

3. Production and Deployment Readiness
   -------------------------------------
   In real-world usage:
       model.fit(X_train, y_train)
       model.predict(X_test)

   The model must work on unseen data. Therefore, `predict(X)` takes
   input explicitly rather than using internally stored training data.

4. Model Serialization
   ---------------------
   When saving a trained model (e.g., via pickle or joblib), we want
   to store only:
       - weights
       - bias
       - hyperparameters

   Storing the full dataset would dramatically increase file size and
   may unintentionally leak training data.

5. Numerical Stability Improvements
   ----------------------------------
   Two safeguards were added:

   (a) Sigmoid clipping:
       Extreme values of z can cause overflow in exp().
       Clipping prevents numerical instability.

   (b) Probability clipping in cost():
       log(0) is undefined (-inf). If y_hat becomes exactly 0 or 1
       due to floating-point precision limits, the loss calculation
       will break. Clipping ensures finite, stable loss values.

6. Stateless Gradient Updates
   ----------------------------
   Gradient calculations use the data passed into `fit()` rather than
   relying on stored attributes like self.X or self.Y. This allows:

       - Easy extension to mini-batch gradient descent
       - Clearer reasoning about data flow
       - Cleaner architecture

Summary
-------
This design reflects production-style machine learning principles:
clear separation of concerns, numerical robustness, memory awareness,
and extensibility.
"""
