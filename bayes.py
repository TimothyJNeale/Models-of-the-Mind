'''
The full form of Bayes' theorem is:
    P(A|B) = P(B|A) * P(A) / P(B)
Where:
    P(A|B) is the probability of event A given event B.
    P(B|A) is the probability of event B given event A.
    P(A) is the probability of event A.
    P(B) is the probability of event B.

The theorem is used to calculate the probability of an event given the probability of another event that is 
related to the first event.

In the context of machine learning, Bayes' theorem is used in the Naive Bayes algorithm, which is a 
classification algorithm based on Bayes' theorem. 

The Naive Bayes algorithm assumes that the features are independent of each other, which is a naive 
assumption, but it simplifies the calculation of the probabilities.
'''

import numpy as np

def bayes_theorem(p_a, p_b_given_a, p_b):
    return p_b_given_a * p_a / p_b

p_a = 0.01
p_b_given_a = 0.9
p_b = 0.02

result = bayes_theorem(p_a, p_b_given_a, p_b)
print(f"P(A|B) = {result}")
