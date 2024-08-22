'''
No, a single-layer perceptron cannot model XOR gates. The XOR function is not linearly separable, 
meaning there is no single straight line that can separate the output classes in the input space. 
A single-layer perceptron can only solve problems that are linearly separable.

To model an XOR gate, you need a multi-layer perceptron (MLP) or a neural network with at least one hidden layer. 
This allows the network to learn non-linear decision boundaries.

To understand what "linearly separable" means, consider a simple 2D plot with two classes of points. 
If you can draw a single straight line that separates all the points of one class from all the points of the other class, 
then the data is linearly separable.

Here's a diagram to illustrate the AND gate, which is linearly separable:

    y
    ^
    |
  1 |    0    1
    |
  0 |    0    0
    |
  0 +-------------> x
    0    0    0   

Line: y = x


The XOR gate outputs 1 only when the inputs are different. 
Here's a diagram to illustrate the XOR gate, which is not linearly separable:

    y
    ^
    |
  1 |    1    0
    |
  0 |    0    1
    +--------------> x
    0    0    1

    
The NXOR gate outputs 1 only when the inputs are the same. 
Here's a diagram to illustrate the NXOR gate, which is not linearly separable:

    y
    ^
    |
  1 |    0    1
    |
  0 |    1    0
    +--------------> x
    0    0    1

'''

from sklearn.neural_network import MLPClassifier
import numpy as np

def gate_logic(X, y, max_iter=10000):

    # Create a multi-layer perceptron model
    mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=max_iter)

    # Train the model
    mlp.fit(X, y)

    # Evaluate the model
    accuracy = mlp.score(X, y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Predict the output for XOR inputs
    predictions = mlp.predict(X)
    print("Predictions:", predictions)

if __name__ == '__main__':

    print("\nXOR gate output=1 only when inputs are different")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    gate_logic(X, y, 10000)

    print("\nXNOR gate output=1 only when inputs are the same")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])

    gate_logic(X, y, 300000)
