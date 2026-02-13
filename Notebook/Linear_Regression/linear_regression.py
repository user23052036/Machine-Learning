import numpy as np

# model parameters -> weight, bias
# hyperparameters -> learning rate, no of iterations

class Linear_Regression():

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations


    # fit the data into out model
    def fit(self, X, Y):
        # number of training examples and number of features
        # X already contains only input features (years_of_experience); target (salary) was manually separated before checking shape
        self.m, self.n = X.shape # no of rows and columns(here n=1)

        self.w = np.zeros(self.n) # innitialize weight
        self.b = 0 # innitialized bias
        self.X = X
        self.Y = Y

        # implementing gradient descent
        for i in range(self.no_of_iterations):
            self.update_weights()
    

    def update_weights(self):
        Y_predicted = self.predict(self.X)
        error = self.Y - Y_predicted

        # calculate gradients, in python dot product is the summation
        dw = (-2/self.m)*((self.X.T).dot(error))
        db = (-2/self.m)*np.sum(error)

        # updating weights and bias
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
    
    def predict(self, X):
        return X.dot(self.w) + self.b
    


"""
import numpy as np


class Linear_Regression():

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.w = None
        self.b = 0.0

    def fit(self, X, Y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.no_of_iterations):

            Y_predicted = X.dot(self.w) + self.b
            error = Y - Y_predicted

            dw = (-2/m) * X.T.dot(error)
            db = (-2/m) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b

"""