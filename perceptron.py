'''
The perceptron algorithm is a simple algorithm that can be used to classify data into two classes 
The algorithm is simple to implement and understand

To calculate the biasa and weights, the perceptron algorithm uses the following formula:
    y = w1*x1 + w2*x2 + ... + wn*xn + b
    where y is the output, w is the weight, x is the input and b is the bias

The perceptron algorithm is trained using the following steps:
    1. Initialize the weights and bias to zero
    2. For each input data, calculate the output using the formula above
    3. Update the weights and bias using the formula:
        w = w + learning_rate * error * x
        b = b + learning_rate * error
    4. Repeat step 2 and 3 until the error is zero or the maximum number of epochs is reached

The implemtation below is an OOP implementation of the perceptron algorithm

'''

import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting

class Perceptron:

    def __init__(self, input_size, learning_rate=0.01, epochs=10000):
        self.input_size = input_size
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors = []


    def predict(self, x):
        x = np.atleast_1d(x)  # Ensure x is at least 1-dimensional
        if x.shape[0] != self.input_size:
            raise ValueError(f"Input shape {x.shape} does not match expected shape ({self.input_size},)")
        #print(f"Predict input x: {x}")  # Debug print
        result = np.dot(self.weights, x) + self.bias
        #print(f"Dot product result: {result}")  # Debug print
        return 1 if result > 0 else 0

    
    def train(self, X, y):
        for _ in range(self.epochs):
            total_error = 0
            for i in range(y.shape[0]):
                y_hat = self.predict(X[i])
                error = y[i] - y_hat
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)
            self.errors.append(total_error)
    
    def accuracy(self, X, y):
        correct = 0
        for i in range(y.shape[0]):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return correct / y.shape[0]
    
    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def plot(self):
        plt.plot(self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.show()

# Example usage

def gate_logic(X, y, epochs=1000):
    input_size = X.shape[1]
    perceptron = Perceptron(input_size=input_size, epochs=epochs)
    perceptron.train(X, y)

    print(f"weights: {perceptron.get_weights()}")
    print(f"bias: {perceptron.get_bias()}")

    print(f"\nPredictions")
    # for each input pair, print the predicted output
    for i in range(X.shape[0]):
        print(f"inputs {X[i]} -> {perceptron.predict(X[i])}")

    accuracy = perceptron.accuracy(X, y)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    #print(f"Errors: {perceptron.errors}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")


    #perceptron.plot()

if __name__ == '__main__':

    print("AND gate")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    gate_logic(X, y)

    print("\nOR gate")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    gate_logic(X, y)

    print("\nNOT gate")
    X = np.array([[0], [1]])
    y = np.array([1, 0])

    gate_logic(X, y)

    print("\nNAND gate")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, 1, 0])

    gate_logic(X, y)

    print("\nNOR gate")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 0])

    gate_logic(X, y)

    print("\nXOR gate output=1 only when inputs are different")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    gate_logic(X, y)

    print("\nXNOR gate output=1 only when inputs are the same")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])

    gate_logic(X, y)

