import numpy as np

class SVM_classifier():

    # initiating the hyperparameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter


    # fitting the dataset to SVM Classifier
    def fit(self, X, Y):

        m, n = X.shape

        # initialize weight vector and bias
        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.no_of_iterations):
            self.update_weights(X, Y)


    # function for updating weight and bias
    def update_weights(self, X, Y):

        # convert labels to {-1, +1}
        y_label = np.where(Y <= 0, -1, 1)

        for index, x_i in enumerate(X):

            condition = y_label[index] * (np.dot(x_i, self.w) + self.b)

            if condition >= 1:
                # only regularization
                dw = 2 * self.lambda_parameter * self.w
                db = 0.0
            else:
                # regularization + hinge loss gradient
                # NumPy applies scalar broadcasting.
                # It multiplies each element of x_i by the scalar.

                dw = 2 * self.lambda_parameter * self.w - y_label[index] * x_i
                db = -y_label[index]   # <-- FIXED SIGN

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db


    # predict labels
    def predict(self, X):

        output = np.dot(X, self.w) + self.b   # <-- FIXED SIGN
        predicted_labels = np.sign(output)

        return np.where(predicted_labels <= 0, 0, 1)
    


"""
Difference Between Batch Gradient Descent and Stochastic Gradient Descent (SGD)

In Linear Regression and Logistic Regression (as usually taught), 
we compute the total loss over the entire dataset first:

    J(w) = (1/m) * sum_i L_i

Then we compute the gradient of this full cost:

    ∇J(w) = (1/m) * sum_i ∇L_i

And finally update parameters once per epoch:

    w = w - η * ∇J(w)

This method is called Batch Gradient Descent.

------------------------------------------------------------

In this SVM implementation, we do something different.

For each individual training example:

    1. Compute its hinge loss gradient
    2. Update w and b immediately

That means we use:

    ∇L_i

instead of the full dataset gradient.

This method is called Stochastic Gradient Descent (SGD).

------------------------------------------------------------

Key Difference:

Batch Gradient Descent:
    - Uses full dataset to compute gradient
    - One update per epoch
    - Stable but slower

Stochastic Gradient Descent:
    - Uses one sample at a time
    - Updates after every sample
    - Noisy but faster
    - Often converges faster in practice

------------------------------------------------------------

Important Insight:

Even though we update per sample,
we are still minimizing the same objective:

    λ||w||² + Σ max(0, 1 - y_i(w·x_i + b))

We are just approximating the full gradient
using individual sample gradients.

Over many iterations, SGD converges toward
the same minimum.

------------------------------------------------------------

Why SVM is commonly trained with SGD:

- Hinge loss is piecewise (not smooth)
- Many samples produce zero gradient
- Only margin-violating points update the model
- SGD naturally handles this efficiently
"""