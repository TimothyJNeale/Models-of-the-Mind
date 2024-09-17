'''
A Hopfield network is a form of recurrent artificial neural network. It consists of a single layer of neurons, 
with connections between each neuron and every other neuron. It is a form of associative memory, and is capable of recalling patterns 
that have been stored in the network.

The Hopfield network is fully connected and symmetric. The weights are updated according to the Hebbian learning rule,
which is a form of unsupervised learning. The weights are updated according to the following rule:

    w_ij = 1/N * sum(x_i * x_j)
    
 where w_ij is the weight between neuron i and neuron j, N is the number of neurons, and x_i and x_j are the states of neurons i and j, respectively.

The activation of the neurons is updated according to the following rule:
    
        x_i = 1 if sum(w_ij * x_j) > 0
        x_i = -1 if sum(w_ij * x_j) < 0

where x_i is the state of neuron i, w_ij is the weight between neuron i and neuron j, and x_j is the state of neuron j.

The Hopfield network is capable of storing a number of patterns, and can recall these patterns when presented with a partial or noisy version of the pattern.
The network is capable of converging to a stable state, which is a stored pattern, or a spurious state, which is not a stored pattern.


THe following file sets up a hopfield networke and allows it to be trained in order to rmeber patterns and then recall them.
The recall function allows a pattern which is not exactly one that is remebered to be entered and returns the mostlikely match

'''

import numpy as np

class HopfieldNetwork:
        def __init__(self, n_neurons):
                self.n_neurons = n_neurons
                self.weights = np.zeros((n_neurons, n_neurons))
                
        def remember(self, X):
                n_samples = X.shape[0]
                for i in range(self.n_neurons):
                        for j in range(self.n_neurons):
                                if i == j:
                                        self.weights[i, j] = 0
                                else:
                                        self.weights[i, j] = 1/n_samples * np.sum(X[:, i] * X[:, j])

                
        def recall(self, pattern, n_iterations=100):
                for _ in range(n_iterations):
                        for i in range(self.n_neurons):
                                activation = np.dot(self.weights[i], pattern)
                                pattern[i] = 1 if activation > 0 else -1
                return pattern
        
        def energy(self, pattern):
                return -0.5 * np.dot(pattern, np.dot(self.weights, pattern))
        
        def get_weights(self):
                return self.weights
        
        def set_weights(self, weights):
                self.weights = weights

        def is_stabel(self, pattern):
                return np.array_equal(pattern, self.recall(pattern))
        
# Example usage
if __name__ == "__main__":

        network = HopfieldNetwork(15)

        # First test with 3 patterns
        patterns = np.array([[1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
                                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
                                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1]])

        network.remember(patterns)
        print("Patterns remembered:")
        for pattern in patterns:
                print(pattern)
        # weights = network.get_weights()
        # print("\nWeights:\n", weights)
        
        # Recall a pattern
        print("\n---------------------------------")
        pattern = np.array([1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1])
        print("Pattern to recall:", pattern)
        recalled_pattern = network.recall(pattern)
        print("Recalled pattern: ", recalled_pattern)
        stable = network.is_stabel(recalled_pattern)
        print("Stable: ", stable)
        
        energy = network.energy(recalled_pattern)
        print("Energy of recalled pattern:", energy)
        
        # show energy for all patterns
        print("\nEnergy of all patterns:")
        for pattern in patterns:
                energy = network.energy(pattern)
                print(pattern, ":", energy)
        print("---------------------------------\n")

        pattern = np.array([1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
        print("Pattern to recall:", pattern)
        recalled_pattern = network.recall(pattern)
        print("Recalled pattern:", recalled_pattern)
        stable = network.is_stabel(recalled_pattern)
        print("Stable: ", stable)
        
        energy = network.energy(recalled_pattern)
        print("Energy of recalled pattern:", energy)
        
        # show energy for all patterns
        print("\nEnergy of all patterns:")
        for pattern in patterns:
                energy = network.energy(pattern)
                print(pattern, ":", energy)
        print("---------------------------------\n")
