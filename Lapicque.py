"""
This modeule recreates the earliset attempts at producing a modle of how nerves behave
In 1907 he speculated that the nerve cell membrane could be modeled as a simple electrical circuit
with a capacitor and a resistor in parallel. This model is known as the Lapicque model.

dV = (V_rest - V + R*I) / (R*C) * dt

Where:
    V_rest is the resting potential
    V is the membrane potential
    R is the resistance
    I is the input current
    C is the capacitance
    dt is the time step
    dV is the change in membrane potential
    
    The model of a capcitor is given by:
        I = C * dV/dt

    The model of a resistor is given by:
        I = V/R

    The model of a capacitor and resistor in parallel is given by:
        I = V/R + C * dV/dt
    
        I - V/R = C * dV/dt
        dt(I - V/R) = C * dV
        dV = dt(I - V/R) / C
        dV = dt(RI - V) / RC
        dV = (RI - V) / RC * dt

    Therefore, the change in membrane potential is given by:
        dV = (V_rest - V + R*I) / (R*C) * dt
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.5 # Capacitance
R = 1.0 # Resistance

V_rest = 0.0 # Resting potential
# A resting potential is the membrane potential of a neuron when it is not being altered by excitatory or inhibitory postsynaptic potentials.
# Typically, this value is around -70 mV.

V_th = 1.0 # Threshold potential
# A threshold potential is the critical level to which a membrane potential must be depolarized to initiate an action (or spike) potential.

V_spike = 0.4 # Spike potential
# A spike potential, also known as an action potential, is a rapid and temporary change in the electrical membrane potential of a cell, 
# typically a neuron, that occurs when the cell is activated by a stimulus. 
# This change is characterized by a sudden depolarization followed by repolarization of the membrane potential.

# Time
dt = 0.01
t = np.arange(0, 10, dt)

# Input current
I = np.zeros(len(t))
I[100:600] = 1.5

# Membrane potential
V = np.zeros(len(t))
V[0] = V_rest

# Simulation
for i in range(1, len(t)):
    # Calculate the change in membrane potential
    # dV = (V_rest - V + R*I) / (R*C) * dt
    dV = (V_rest - V[i-1] + R*I[i-1]) / (R*C) * dt

    # Update the membrane potential
    V[i] = V[i-1] + dV

    # Check for spike,if there is a spike, set the membrane potential to the spike potential value
    if V[i] >= V_th:
        V[i] = V_spike

    print(f"Step: {i}, Current: {I[i]} ,Voltage: {V[i]}")

# Plot
plt.figure()
plt.plot(t, V)
plt.title("Lapicque: Membrane Potential Over Time")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (V)')
plt.show()

