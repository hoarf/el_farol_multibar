* [2015-08-23 Sun] This is my attempt to implement the Q-Learning algorithm.

#+BEGIN_SRC python
import numpy as np

# A table containing the action-value pairs
Q = []

# A table containing the state-action pairs
N = []

# Last action taken
a = None

# Previous state
i = None

# Previous state's reward
r = None

def alpha(utility):
    return utility

def max_utility(j):
    return np.max(Q, axis=j)
    
def argmax_utility(j):
    return a

def Q-learn-agent(percept):
    j = percept['state']
    if i:
        N[a,i] += 1
        Q[a,i] = Q[a,i] + alpha(r + max_utility(j) - Q[a, i])
    if percept['terminal']:
        i = None
    else:
        i = j
        r = percept['reward']
    a = argmax_utiltiy(f, j)
    return a
#+END_SRC
