'''
Reinforcement Learning: Bellman Equation

The Bellman equation is used in dynamic programming to describe the value of making a decision at a 
particular state, in terms of the payoff from the current decision and the value of the future state that
the decision leads to. The Bellman equation is a necessary condition for optimality in the sense that
any policy that is optimal must satisfy the Bellman equation.

The Bellman equation: 

    V(s) = max_a (R(s, a) + γ * Σ_s' P(s' | s, a) * V(s'))
    
    where:
        V(s) is the value of state s,
        R(s, a) is the reward of taking action a in state s,
        γ is the discount factor,
        P(s' | s, a) is the probability of transitioning to state s' from state s by taking action a,
        V(s') is the value of state s'.
'''
import matplotlib.pyplot as plt

# --- Define the Bellman equation ---
def bellman(V, R, P, gamma):
    '''
    The Bellman equation.
    
    Arguments:
        V: dict, the value function.
        R: dict, the reward function.
        P: dict, the transition probability function.
        gamma: float, the discount factor.
        
    Returns:
        dict, the updated value function.
    '''
    # Initialize the updated value function
    V_new = {}
    
    # Loop over all states
    for s in V:
        # Initialize the maximum value
        max_value = float('-inf')
        
        # Loop over all actions
        for a in R[s]:
            # Initialize the value of the state
            value = R[s][a]
            
            # Loop over all possible next states
            for s_next in P[s][a]:
                value += gamma * P[s][a][s_next] * V[s_next]
            
            # Update the maximum value
            max_value = max(max_value, value)
        
        # Update the value of the state
        V_new[s] = max_value
    
    return V_new

# --- Test the Bellman equation ---
# Define the value function
V = {'s1': 0, 's2': 0}

# Define the reward function
R = {
    's1': {'a1': 1, 'a2': 2},
    's2': {'a1': 3, 'a2': 4}
}

# Define the transition probability function
P = {
    's1': {
        'a1': {'s1': 0.5, 's2': 0.5},
        'a2': {'s1': 0.1, 's2': 0.9}
    },
    's2': {
        'a1': {'s1': 0.2, 's2': 0.8},
        'a2': {'s1': 0.3, 's2': 0.7}
    }
}

# Define the discount factor
gamma = 0.9

# Update the value function using the Bellman equation
V_new = bellman(V, R, P, gamma)

# Print the updated value function
print(V_new)

# plot i=a diagram of the value function


plt.bar(V_new.keys(), V_new.values())
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Value Function')
plt.show()
