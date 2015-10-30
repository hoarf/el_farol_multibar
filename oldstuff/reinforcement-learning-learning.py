
# coding: utf-8

# In[114]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ## Notation

# 'The Book': This refers to the book: Russell, Stuart, and Peter Norvig. "Artificial intelligence: a modern approach." (1995).
# 
# State: State here is read as the index of a matrix. For instance in the Q matrix the position Q[0,1] corresponds to all the Q-values associated wit the state (0,1) for any given action. **Note that all the index here are 0-based as opposed to 1-based index in the book**
# 
# Action: This is one of the five possible choices for the agent at any given state, namely: LEFT, TOP, RIGHT, BOTTOM, STAND_STILL, that will be represented in the algorith by the index 0,1,2,3,4 respectivelly

# ## Reads the evironment data

# In[131]:

rewards_data = pd.read_csv('/vagrant_data/src/reinforcement-learning-practice/rewards.csv', index_col=[0,1])
initial_utility_data = pd.read_csv('/vagrant_data/src/reinforcement-learning-practice/initial_utility_data.csv', index_col=[0,1,2])


# ## Converts the data into a more mathy format
# 
# This enables the handling of a state as the index of a matrix, I.E rewards[1,0] corresponds to the reward associated with the state (1,0) which is convenient and makes the syntax more like the one in the book

# In[132]:

rewards = rewards_data.as_matrix().reshape((3,4))
initial_utility = initial_utility_data.as_matrix().reshape((3,4,5))


# ## Model parameters

# In[133]:

# This is constant, a percept represents an environment stimulus
START_PERCEPT = {
  'state': (0,0),
  'reward': rewards[0,0],
  'terminal': False
}

# Optimistic reward estimation
R_PLUS = 2

# At least how many times each state should be tried
N_E = 5

# Learning rate
alpha = .1


# ## Initializations

# In[134]:

# A table containing the action-value pairs
# Domain is State-index0 x State-index1 x Action
Q = initial_utility

# A table containing the state-action pairs
# Domain is State-index0 x State-index1 x Action
N = np.zeros((3,4,5))

# Last action taken
a = None

# Previous state
i = None

# Previous state's reward
r = None

# 'SOMETHING SOMETHING MADNESS SOMETHING SOMETHING DIFFERENT RESULTS' - Einstein
np.random.seed = 10


# # The main thing

# In[135]:

# This is the main algorithm that decides the new action for the agent for a given percept
def Q_learn_agent(percept):
  global i, a, r, Q, N
  j = percept['state']
  if i:
    N[i][a] += 1
    Q[i][a] = Q[i][a] + alpha*(r + np.max(Q[j]) - Q[i][a])
  if percept['terminal']:
    i = None
  else:
    i = j
    r = percept['reward']
  a = choose_action(j)
  return a

# The exploratory function written in the literature, it decides wheter or not we already explored enough and are ready to be greedy
def exploratory_fuction(u, n):
  return u if u == -9999 else R_PLUS if n < N_E else u

# This makes our exploratry_function accept vectors as argument instead of scalars
v_exploratory_function = np.vectorize(exploratory_fuction)

# At state: 'j' choses something to do.
def choose_action(j):
  utility = v_exploratory_function(Q[j],N[j])
  
  # This returns an array with True as value if the position in the array is maximum. 
  #  Ex: [False True True False False] for the utilities: [ 0 2 2 1 .5]
  all_max_mask = utility == np.max(utility) 

  # This maps the previous array to the relative frequency of True values. Basicaly I'm giving equal probabability of 
  # being randomly chosen for each max value on the utility array
  action_frequencies = map(lambda x: 1.0/count_nonzero(all_max_mask) if x else 0, all_max_mask)
   
  # This choses an action represented by the range(0,5) given the probabilities set previously.   
  return np.random.choice(range(0,5),1,p=action_frequencies)

# This is the environment deciding what feedback to give to the agent
def new_percept(new_action):
  new_state = calculate_new_state_based_on_action(i, new_action)
  return {
    'state': new_state,
    'reward': rewards[new_state],
    'terminal': new_state == (1,3) or new_state == (2,3)
  }

# This calculates the coordinates of the new state based on the previous state plus the action
# Action LEFT=0 ; TOP=1; RIGHT=2; BOTTOM=3; STANT_STILL=4;
def calculate_new_state_based_on_action(state, action):
  state = (state[0]  , state[1]-1) if action[0] == 0 else state
  state = (state[0]+1, state[1]  ) if action[0] == 1 else state    
  state = (state[0]  , state[1]+1) if action[0] == 2 else state
  state = (state[0]-1, state[1]  ) if action[0] == 3 else state
  return state

# this is where the whole process start
start_state_utility = []
def learn(nepochs=300):
  global start_state_utility, debug, last_actions
  for epoch in xrange(1,nepochs):
    print "Epoch %i - Done." % epoch
    current_percept = START_PERCEPT
    start_state_utility.append(Q[0,0])
    while True:
      new_action = Q_learn_agent(current_percept)
      if not i: # means we reached a terminal state, let's start over a new epoch
        break
      np = new_percept(new_action)
      current_percept = np
  print("Done. Everything Now.")


# # Results

# In[136]:

nepoch = 10
learn(nepoch)


# ## Utility of start state (0,0) over the epochs

# In[137]:

plt.plot(xrange(1,nepoch), start_state_utility, label=range(0,5))


# In[138]:

print Q.reshape(5,3,4) # The utility for every action, from 0 to 4


# In[139]:

print N.reshape(5,3,4) # The frequency for every action from 0 to 1


# In[140]:

print Q[:,:,4] # Utility for every state to go to bottom


# In[141]:

print Q[0,0,:] # Utility for state (0,0) for every action

