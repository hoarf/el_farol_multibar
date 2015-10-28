"""
Code to simulate agents in the ElFarol problem using the wonderful life
utility as rewards.

Author: Alan Ficagna
Version: 1.0.1
Expample of Usage:

    from wlu import ElFarolWLU, plot_attendances, plot_q_values, plot_world_utilities
    tresholds = [0.3,0.5]
    nr_agents = 10
    att, wu, aqv = ElFarolWLU(nr_weeks, tresholds, nr_agents)
    plot_attendances(att)
    plot_world_utilities(wu)
    plot_q_values(aqv)

"""
import numpy as np
import gmpy2
from gmpy2 import mpfr
from matplotlib import pyplot as plt
from itertools import cycle

# PARAMETERS
MAXREWARD = 1000.0
ALPHA = .010
DECAY = 0.999

# CONSTANTS
MAX_METHOD = 0
BAR_RESULT_BAD = 0
BAR_RESULT_GOOD = 1
ACTION_STAY_HOME = 0
INITIAL_EXPLORATION_CHANCE = .3
GREEDY = 0
EXPLORE = 1

class Agent:

  def __init__(self, maxactions):
    """
    maxactions: how many actions the agent has
    """
    self.choices = xrange(maxactions)
    self.action_q_values = np.zeros(maxactions)

  def __repr__(self):
    """
    this makes a prettier print(agent)
    """
    return repr(self.action_q_values)

  def chose_action(self, p):
    """
    checks its Q-table and updates it's exploration probability based on a factor of decay
    p: the probability for agent to choose to explore rather than use its Q-table
    """
    if np.random.choice([EXPLORE, GREEDY], p=[p,1-p]) == GREEDY:
      best_actions = np.argwhere(self.action_q_values == np.max(self.action_q_values)).flatten()
      self.action = np.random.choice(best_actions)
    else:
      self.action = np.random.choice(self.choices)

  def update_utilities(self, reward):
    """
    takes the rewards vector and updates the Q-table
    """
    q_a = self.action_q_values[self.action]
    self.action_q_values[self.action] = q_a + ALPHA*(reward - q_a)

class World:

  def __init__(self, thresholds=[0.3, 0.5], nr_agents=100):
    """
    nr_agents: how many agents there are in the world as integer
    thresholds: list of optimum ocupation thresholds for each bar
    """
    self.thresholds = thresholds
    self.maxbar = len(thresholds)
    self.maxactions = self.maxbar + 1
    self.attendances = np.zeros(self.maxactions, dtype=np.float128)
    self.agents = [ Agent(self.maxactions) for x in xrange(nr_agents) ]
    self.week = 0
    self.p = INITIAL_EXPLORATION_CHANCE
    self.reward_function = self.get_reward_discrete if nr_agents <= 100 else self.get_reward

  def __repr__(self):
    """
    this makes a prettier print(world)
    """
    return repr(self.agents)

  def get_reward(self, attendance, threshold):
    """
    attendance: list of attendance counts for each bar
    threshold: list of thresholds preferences for each bar
    returns: agent's reward as a real value
    """
    a = ((attendance-threshold)**2)*MAXREWARD #DIFF
    b = a**2
    return MAXREWARD/np.exp(b) if b < 40 else 0

  def get_reward_discrete(self, attendance, threshold):
    """
    attendance: list of attendance counts for each bar
    threshold: list of thresholds preferences for each bar
    returns: agent's reward as a real value, but using a discrete version of the function get_reward()
    """
    if attendance/len(self.agents) == threshold:
      return MAXREWARD
    elif (threshold - 0.1 <= attendance) and (threshold + 0.1 >= attendance):
      return 679.0
    else:
      return 0

  def calculate_world_utility(self, agent_set):
    """
    agent_set: list of Agents that is used to calculate the rewards
    returns: the world utility (average utility over all agents)
    """
    self.attendances = np.zeros(self.maxactions, dtype=np.float128)
    self.bar_results = np.zeros(self.maxbar, dtype=np.float128)
    self.rewards = np.zeros(self.maxactions, dtype=np.float128)

    nr_agents = len(agent_set)

    # Updates attendences counts
    for agent in agent_set:
      self.attendances[agent.action] += 1.0

    # Updates the bar results
    for bar in xrange(self.maxbar):
      self.bar_results[bar] = BAR_RESULT_GOOD if self.attendances[bar+1]/nr_agents <= self.thresholds[bar] else BAR_RESULT_BAD

    home_good = not reduce(lambda x, y: x or y, self.bar_results)

    # Updates the rewards
    self.calculate_bar_rewards(home_good, nr_agents)

    return np.mean([self.rewards[a.action] for a in agent_set])

  def calculate_bar_rewards(self, home_good, nr_agents):
    """
    updates the rewards associated with each bar
    home_good: wheter or not it was good to stay home
    """
    for action in xrange(self.maxactions):
      if action == ACTION_STAY_HOME:
        self.rewards[action] = MAXREWARD/self.maxbar if home_good else 0
      else:
        self.rewards[action] = self.bar_results[action-1]*self.reward_function(self.attendances[action]/nr_agents, self.thresholds[action-1])

  def step(self):
    """
    performs a time step and updates the world
    """
    for agent in self.agents:
      agent.chose_action(self.p)

    self.G = self.calculate_world_utility(self.agents)
    if self.week % 500 == 0:
        print("Week %s" % self.week)
        print("Agent 0 q-values before receiving reward %s" % self.agents[0].action_q_values)
        print("Agent 0 action %s" % self.agents[0].action)
        print("Bar attendances %s" % self.attendances)
        print("Bar results %s" % self.bar_results)
        print("List of rewards %s" % self.rewards)
        print("Exploration probability %s" % self.p)
    self.update_agents_utilities()
    if self.week % 500 == 0:
        print("Agent 0 q-values after reward %s" % self.agents[0].action_q_values)
        print('-----------------')
    self.week += 1


  def update_agents_utilities(self):
    """
    for each agent removes him from the agent_set and recalculates the world utility,
    then updates Q-table with the WLU
    """
    # reserva = self.agents.pop(0)
    for agent in self.agents:
        agent.update_utilities(self.rewards[agent.action])
      # agent.update_utilities(self.G - self.calculate_world_utility(self.agents))
      # self.agents.append(reserva)
      # reserva = self.agents.pop(0)
    # self.agents.append(reserva)

    # this last operation recalculates the rewards and attendances for the complete
    # agent set; this is a flaw in the design of this algorithm in the sense that this
    # extra run over it is not needed but it don't seem to be worth correcting for now
    # self.calculate_world_utility(self.agents)

def ElFarolWLU(nr_weeks=5000, thresholds=[0.3,0.5], nr_agents=100):
  """
  nr_wks: number of weeks as an integer
  thresholds: list of thresholds for each bar
  nr_agents: number of agents as an integer
  returns: the attendance for each bar for each week and the
           world utility for each week as a tuple
  """
  nr_actions = len(thresholds) + 1
  weekly_attendance = np.zeros((nr_actions,nr_weeks))
  w = World(thresholds, nr_agents)
  world_utilities = np.zeros(nr_weeks)
  agent_q_values = np.zeros((nr_agents, nr_actions, nr_weeks))
  for week in xrange(nr_weeks):
    w.step()
    world_utilities[week] = w.G
    for action in xrange(nr_actions):
      weekly_attendance[action][week] = w.attendances[action]/nr_agents
      for ag_ix,_ in enumerate(w.agents):
        agent_q_values[ag_ix][action][week] = w.agents[ag_ix].action_q_values[action]

  return weekly_attendance, world_utilities, agent_q_values

def plot_attendances(data, ylim=1):
  """
  data: matrix of NUMBER_OF_BARS x NUMBER_OF_WEEKS
  """
  nr_weeks = data.shape[1]
  nr_actions = data.shape[0]
  x = np.linspace(0,nr_weeks-1,nr_weeks)
  f, axis = plt.subplots(nrows=nr_actions)
  for index, ax in enumerate(axis):
    ax.scatter(x, [data[index][k] for k in x], s=1)
    ax.set_title("bar" + str(index) if index != 0 else "stay")
    ax.set_ylim(0, ylim)
    ax.set_xlim(0, nr_weeks)
    ax.set_yticks(np.arange(0,ylim,0.1))
    ax.set_xticks(np.arange(0,nr_weeks,nr_weeks/10))
    ax.set_ylabel("Attendance")
    ax.set_xlabel("Weeks")
    ax.grid()

  f.set_size_inches(10.5, 10.1)
  f.tight_layout()

def plot_world_utilities(world_utilities):
  """
  world_utilities: list of world utilities of size nr_weeks
  """
  nr_weeks = world_utilities.shape[0]
  axis = plt.subplot(111)
  x = np.linspace(0,nr_weeks-1,nr_weeks)
  axis.scatter(x, world_utilities, s=1)

def plot_q_values(agent_q_values, ylim=4000):
  """
  agent_q_values: matrix of shape NR_AGENTSxNR_ACTIONSxNR_WEEKS
  """
  col_gen = cycle('bgrcmk')
  linestyles_gen = cycle(['-'])
  nr_weeks = agent_q_values.shape[2]
  cols = [col_gen.next() for _ in xrange(agent_q_values.shape[1])]
  linestyles = [linestyles_gen.next() for _ in xrange(agent_q_values.shape[1])]
  x = np.linspace(0,nr_weeks-1,nr_weeks)
  nr_agents = agent_q_values.shape[0]
  nr_actions = agent_q_values.shape[1]
  f, axis = plt.subplots(ncols=nr_agents)
  for index, ax in enumerate(axis):
    for bar in xrange(nr_actions):
      ax.plot(x, agent_q_values[index][bar], c=cols[bar], ls=linestyles[bar], label='stay' if bar == 0 else 'bar%i'% bar )
    ax.set_ylabel("Q-Value")
    ax.set_xlabel("Weeks")
    if ylim:
        ax.set_ylim(-4000, ylim)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
  f.set_size_inches(15, 3)
  f.tight_layout()
