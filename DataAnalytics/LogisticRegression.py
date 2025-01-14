# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:37:20 2025

@author: ssj55
"""
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.num_iterations = num_iterations  # Number of iterations to run gradient descent
    
    def sigmoid(self, z):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """ Train the logistic regression model using gradient descent """
        m, n = X.shape  # m = number of samples, n = number of features
        self.theta = np.zeros(n)  # Initialize weights (parameters)
        self.bias = 0  # Initialize bias term
        
        # Gradient descent loop
        for i in range(self.num_iterations):
            # Linear model prediction (z = X.dot(theta) + bias)
            linear_model = np.dot(X, self.theta) + self.bias
            
            # Apply sigmoid to get predicted probabilities
            predictions = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (predictions - y))  # Gradient with respect to weights
            db = (1/m) * np.sum(predictions - y)  # Gradient with respect to bias
            
            # Update parameters
            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict(self, X):
        """ Predict binary labels (0 or 1) for given input data X """
        linear_model = np.dot(X, self.theta) + self.bias
        probabilities = self.sigmoid(linear_model)
        
        # Return the predicted labels: 1 if probability >= 0.5, else 0
        return [1 if prob >= 0.5 else 0 for prob in probabilities]
    
    def predict_proba(self, X):
        """ Return the predicted probabilities for the given input data X """
        linear_model = np.dot(X, self.theta) + self.bias
        return self.sigmoid(linear_model)

# Example usage:
if __name__ == "__main__":
    # Sample data (features X and labels y)
    X = np.array([[1, 2], [1, 3], [2, 2], [2, 3], [3, 3]])
    y = np.array([0, 0, 0, 1, 1])  # Corresponding labels (binary)

    # Create the LogisticRegression model
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)

    # Train the model
    model.fit(X, y)

    # Predict on new data
    predictions = model.predict(X)
    print(f"Predictions: {predictions}")

    # Get predicted probabilities
    probas = model.predict_proba(X)
    print(f"Predicted Probabilities: {probas}")
    
  

