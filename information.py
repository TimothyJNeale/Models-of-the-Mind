'''
To calculate the amount of information stored in a deep neural network (DNN), we can think of the weights and biases of the network as
encoding information. A deep neural network consists of multiple layers of neurons, each with connections represented by weights.
The parameters of the network (weights and biases) are the key elements determining the amount of information.

Approach to Estimate the Information Stored:
Number of Parameters: 
The total number of weights and biases in the network gives an estimate of how much information the network can store. 
Each parameter can be seen as holding information, and the precision of these parameters (e.g., 32-bit floating-point numbers) determines how much 
information can be stored.

Entropy of Parameters: 
If you want a more sophisticated approach, you can estimate the Shannon entropy of the weights and biases. 
The entropy can give a measure of the uncertainty or information content in the parameters.

Steps to Calculate Information:
Total Number of Parameters: 
For each layer in the network, calculate the number of weights and biases, then sum them up.

Precision of Parameters: 
Typically, parameters are stored as 32-bit floating-point numbers, so you can use this to calculate the number of bits 
of information. However, neural networks can also use lower precision (e.g., 16-bit or 8-bit).

'''
import torch
import torch.nn as nn

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

# Calculate the number of parameters
def calculate_total_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Calculate the amount of information in bits
def calculate_information_content(model, precision_bits=32):
    total_params = calculate_total_parameters(model)
    total_bits = total_params * precision_bits
    return total_bits

# Example usage:
total_parameters = calculate_total_parameters(model)
information_in_bits = calculate_information_content(model, precision_bits=32)

print(f"Total parameters in the model: {total_parameters}")
print(f"Information content in the model: {information_in_bits} bits")

