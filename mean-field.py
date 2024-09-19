'''
The mean field model for a balanced network stats with N neurons both excittory and inhibitory wherein the 
Neurind recieve both exteranla and recurrent input. For the recurrent input each neuron recieves K excitatory and K inhibitory inputs.
K is assumed to be much smaller than N. 

                1 << K << N

The recurrent input is given by the sum of the input from the other neurons of the same type. The external input is assumed to be Gaussian 
with mean m and standard deviation s.

The activity of the neurons is given by the following equation:

                r_i = tanh(I_i + sum(J_ij * r_j))

where r_i is the activity of neuron i, I_i is the external input to neuron i, J_ij is the synaptic weight from neuron j to neuron i, and the sum is 
over all neurons j.

The synaptic weights are drawn from a Gaussian distribution with mean 0 and standard deviation g/sqrt(N). The activity of the neurons is updated
using the above equation for T time steps.

The code below simulates the mean field model for a balanced network and plots the activity of the neurons and the synaptic weights.It also plots the
distribution of the synaptic weights, the activity of the neurons, the external input, and the total input to the neurons.

'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000    # Number of neurons
K = 100     # Number of inputs per neuron
g = 5.0     # Synaptic coupling

# External input
m = 25.0    # Mean
s = 1.0     # Standard deviation

# Time vector
T = 1000
dt = 0.1

# Initialize the variables
r = np.zeros((2,N)) # Activity  of the neurons
J = np.zeros((2,N,N)) # Synaptic weights

# Initialize the synaptic weights
J = [g * np.random.normal(0, 1, (N, N)) / np.sqrt(N) for _ in range(2)]

# Initialize the external input
I = np.zeros((2, N))
I[0] = np.random.normal(m, s, N)
I[1] = np.random.normal(m, s, N)

# Initialize the activity
r = np.zeros((2, N))

# Time evolution
for t in range(T):
    r = np.tanh(I + np.array([np.dot(J[i], r[i]) for i in range(2)]))
    # if t % 100 == 0:
    #     plt.plot(r[0], r[1], 'k.', markersize=1)
    #     plt.show()

# Plot the activity of the neurons
plt.plot(r[0], r[1], 'k.', markersize=1)
plt.show()

# Plot the synaptic weights
plt.plot(J[0].flatten(), J[1].flatten(), 'k.', markersize=1)
plt.show()

# Plot the distribution of the synaptic weights
plt.hist(J[0].flatten(), bins=100)
plt.show()
plt.hist(J[1].flatten(), bins=100)
plt.show()


# Plot the distribution of the activity
plt.hist(r[0],bins=100)
plt.show()
plt.hist(r[1],bins=100)
plt.show()

# Plot the distribution of the external input
plt.hist(I[0],bins=100)
plt.show()
plt.hist(I[1],bins=100)
plt.show()

# Plot the distribution of the input
plt.hist(I[0]+np.einsum('ijk,jk->ji',J,r)[0],bins=100)
plt.show()
plt.hist(I[1]+np.einsum('ijk,jk->ji',J,r)[1],bins=100)
plt.show()

# Plot the distribution of the input
plt.hist(np.einsum('ijk,jk->ji',J,r)[0],bins=100)
plt.show()
plt.hist(np.einsum('ijk,jk->ji',J,r)[1],bins=100)
plt.show()

