'''
THis model is based on the Hodgkin-Huxley model of the action potential in the squid giant axon.

The model is based on the following differential equations:

Cm * dV/dt = I - gNa * m^3 * h * (V - ENa) - gK * n^4 * (V - EK) - gL * (V - EL)

dm/dt = alpha_m * (1 - m) - beta_m * m

dh/dt = alpha_h * (1 - h) - beta_h * h

dn/dt = alpha_n * (1 - n) - beta_n * n

where:
    V is the membrane potential
    I is the external current   
    m, h, n are the gating variables    
    Cm is the membrane capacitance
    gNa, gK, gL are the conductances of the sodium, potassium and leak channels
    ENa, EK, EL are the Nernst potentials for sodium, potassium and leak channels
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n are the rate constants for the gating variables

The model is implemented in the following code:

'''

import numpy as np
import matplotlib.pyplot as plt

# Constants
Cm = 1.0
gNa = 120.0
gK = 36.0
gL = 0.3
ENa = 50.0
EK = -77.0
EL = -54.387
Vm = -65.0

# Rate constants
def alpha_m(V):
    return 0.1 * (V + 40.0) / (1.0 - np.exp(-0.1 * (V + 40.0)))

def beta_m(V):
    return 4.0 * np.exp(-0.0556 * (V + 65.0))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65.0))

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-0.1 * (V + 35.0)))

def alpha_n(V):
    return 0.01 * (V + 55.0) / (1.0 - np.exp(-0.1 * (V + 55.0)))

def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65.0))

# Time parameters
dt = 0.01
T = 100
t = np.arange(0, T, dt)


# Input current
I = np.zeros(len(t))
I[600:7000] = 10

# Initialize variables
V = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
n = np.zeros(len(t))

V[0] = Vm
m[0] = alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))
h[0] = alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))
n[0] = alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

print(f"Initial V: {V[0]}, m: {m[0]}, h: {h[0]}, n: {n[0]}")

# Main loop
for i in range(1, len(t)):
    V[i] = V[i - 1] + dt * (I[i - 1] - gNa * m[i - 1] ** 3 * h[i - 1] * (V[i - 1] - ENa) - gK * n[i - 1] ** 4 * (V[i - 1] - EK) - gL * (V[i - 1] - EL)) / Cm
    m[i] = m[i - 1] + dt * (alpha_m(V[i - 1]) * (1 - m[i - 1]) - beta_m(V[i - 1]) * m[i - 1])
    h[i] = h[i - 1] + dt * (alpha_h(V[i - 1]) * (1 - h[i - 1]) - beta_h(V[i - 1]) * h[i - 1])
    n[i] = n[i - 1] + dt * (alpha_n(V[i - 1]) * (1 - n[i - 1]) - beta_n(V[i - 1]) * n[i - 1])


# Plot results
plt.figure(figsize=(10, 8))  # Define the size of the figure
plt.subplot(2, 1, 1)
plt.plot(t, V, 'k')
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.title('Hodgkin-Huxley Model')

plt.subplot(2, 1, 2)
plt.plot(t, m, 'r', label='^N+')
plt.plot(t, h, 'g', label='~N+')
plt.plot(t, n, 'b', label='^K+')
plt.xlabel('Time (ms)')
plt.ylabel('Gating variables')

plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()

