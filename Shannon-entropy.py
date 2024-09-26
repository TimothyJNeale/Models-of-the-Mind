'''
To calculate the Shannon entropy of the weights in a neural network, we need to treat the distribution of the weights as a probability distribution. 
The Shannon entropy measures the uncertainty in a distribution, so for weights of a neural network, it reflects how uncertain (or spread out) 
the weight values are.

Steps to Calculate Shannon Entropy:
Get the Weights: Extract the weights from the neural network.

Normalize the Weights: Treat the weights as a probability distribution. This can be done by creating a histogram of the weights, then normalizing 
the values (i.e., dividing by the total count to get probabilities).

Apply Shannon Entropy Formula: 
The Shannon entropy for a discrete probability distribution 
    p(x) is given by H(X) = -Σ p(x) log2 p(x), where the sum is taken over all possible values of x.

Where he more spread out or uncertain the weights are, the higher the entropy.

'''

import torch
import torch.nn as nn
import numpy as np

# Define a sample neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)   # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)    # Hidden layer to output layer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network
model = SimpleNN()

# Calculate Shannon entropy for weights
def calculate_entropy(weights, num_bins=100):
    # Flatten the weights into a 1D array
    weights = weights.detach().numpy().flatten()
    
    # Create a histogram of the weights with num_bins bins
    hist, bin_edges = np.histogram(weights, bins=num_bins, density=True)
    
    # Normalize the histogram to obtain probabilities (sum of probabilities = 1)
    probs = hist / np.sum(hist)
    
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy

# Function to compute the total entropy of the model's weights
def calculate_model_entropy(model, num_bins=100):
    total_entropy = 0.0
    for param in model.parameters():
        if param.requires_grad:  # Consider only trainable parameters (i.e., weights, not biases)
            total_entropy += calculate_entropy(param.data, num_bins)
    return total_entropy

# Calculate the entropy of the model's weights
model_entropy = calculate_model_entropy(model)
print(f"Total Shannon Entropy of the model's weights: {model_entropy:.4f} bits")
