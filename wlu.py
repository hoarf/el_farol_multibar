"""
Code to simulate agents in the ElFarol problem using the wonderful life
utility as rewards.

Author: Alan Ficagna
"""
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle
import math

# Raise exception on numeric problems
np.seterr(all='raise')

# CONSTANTS
MAX_METHOD = 0
MAXREWARD = 1000.0
BAR_RESULT_BAD = 0
BAR_RESULT_GOOD = 1
ACTION_STAY_HOME = 0
GREEDY = 0
EXPLORE = 1
MAX_NR_AGENTS_FOR_DISCRETE_FUNCTION = 100

class Agent:

  def __init__(self):
    self.choices = xrange(NR_ACTIONS)
    self.action_q_values = INITIAL_Q_VALUES()

  def chose_action(self, p):
    """
    checks its Q-table and updates it's exploration probability based on a
      factor of decay
    p: the probability for agent to choose to explore rather than use its
      Q-table
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
    new_value = q_a + ALPHA*(reward - q_a)
    self.action_q_values[self.action] = new_value

class World:

  def __init__(self):
    self.attendances = np.zeros(NR_ACTIONS, dtype=np.float128)
    self.agents = [ Agent() for _ in xrange(NR_AGENTS) ]
    self.week = 0
    self.p = INITIAL_EXPLORATION_CHANCE

    if CONTINUOUS:
      self.reward_function = self.get_reward_continuous
    elif NR_AGENTS <= MAX_NR_AGENTS_FOR_DISCRETE_FUNCTION:
      self.reward_function = self.get_reward_discrete
    else:
      self.reward_function = self.get_reward

    if USE_WLU:
      self.update_rule = self.update_agents_wlu
    else:
      self.update_rule = self.update_agents

    if CONTINUOUS:
      self.is_home_good_fn = self.is_home_good_continuous
    else:
      self.is_home_good_fn = self.is_home_good

  def get_reward_continuous(self, attendance, threshold):
    """
    attendance: list of attendance counts for each bar
    threshold: list of THRESHOLDS preferences for each bar
    returns: agent's reward as a real value
    """
    return REWARD_CONT(attendance, threshold)

  def relative_attendances(self, nr_agents):
    """
    returns: the list of relative attendances
    """
    return self.attendances/nr_agents

  def get_reward(self, attendance, threshold):
    """
    attendance: list of attendance counts for each bar
    threshold: list of THRESHOLDS preferences for each bar
    returns: agent's reward as a real value
    """
    a = ((attendance-threshold)**2)*MAXREWARD #DIFF
    b = a**2
    return MAXREWARD/np.exp(b) if b < 40 else 0

  def get_reward_discrete(self, attendance, threshold):
    """
    attendance: list of attendance counts for each bar
    threshold: list of THRESHOLDS preferences for each bar
    returns: agent's reward as a real value, but using a discrete version of the function get_reward()
    """
    if attendance == threshold:
      return MAXREWARD
    elif (threshold - 0.1 <= attendance) and (threshold + 0.1 >= attendance):
      return 679.0
    else:
      return 0

  def is_home_good(self, bar_results):
    """
    bar_results: list containing the flag that indicates whether or not it
        was good to go to the given bar
    returns: true if no bar had more than the threshold
    """
    return not reduce(lambda x, y: x or y, bar_results)

  def is_home_good_continuous(self, bar_results):
    """
    bar_results: list containing the flag that indicates whether or not it
        was good to go to the given bar
    returns: true if no bar had more than the threshold
    """
    return HOME_GOOD_CONT_FN(sum(bar_results)) >= 0

  def calculate_world_utility(self, agent_set):
    """
    agent_set: list of Agents that is used to calculate the rewards
    returns: the world utility (average utility over all agents)
    """
    self.attendances = np.zeros(NR_ACTIONS)
    self.bar_results = np.zeros(NR_BARS)
    self.rewards = np.zeros(NR_ACTIONS)

    nr_agents = len(agent_set)
    self.update_attendance_count(agent_set)
    self.update_bar_results(self.relative_attendances(nr_agents))
    home_good = self.is_home_good_fn(self.bar_results)
    self.update_bar_rewards(home_good, nr_agents)
    return np.mean([self.rewards[a.action] for a in agent_set])

  def update_attendance_count(self, agent_set):
    for agent in agent_set:
      self.attendances[agent.action] += 1.0

  def update_bar_results(self, attendances):
    for bar in xrange(NR_BARS):
      result = attendances[bar+1] <= THRESHOLDS[bar]
      self.bar_results[bar] = BAR_RESULT_GOOD if result else BAR_RESULT_BAD

  def update_bar_rewards(self, home_good, nr_agents):
    """
    updates the rewards associated with each bar
    home_good: wheter or not it was good to stay home
    """
    for action in xrange(NR_ACTIONS):
      if action == ACTION_STAY_HOME:
        self.rewards[action] = MAXREWARD/NR_BARS if home_good else 0
      else:
        a = self.relative_attendances(nr_agents)[action]
        t = THRESHOLDS[action-1]
        result = self.bar_results[action-1]
        self.rewards[action] = result*self.reward_function(a,t)

  def update_bar_rewards(self, home_good, nr_agents):
    """
    updates the rewards associated with each bar
    home_good: wheter or not it was good to stay home
    """
    for action in xrange(NR_ACTIONS):
      if action == ACTION_STAY_HOME:
        self.rewards[action] = MAXREWARD/NR_BARS if home_good else 0
      else:
        a = self.relative_attendances(nr_agents)[action]
        t = THRESHOLDS[action-1]
        result = self.bar_results[action-1]
        self.rewards[action] = result*self.reward_function(a,t)
  def step(self):
    """
    performs a time step and updates the world
    """
    for agent in self.agents:
      agent.chose_action(self.p)

    self.G = self.calculate_world_utility(self.agents)
    self.trace(full=True)
    self.update_rule()
    self.trace()
    self.week += 1
    self.p = DECAY_FUNCTION(self.p, self.week)

  def trace(self, full=False):
    if DEBUG:
      if self.week % 500 == 0:
        if full:
          print('-----------------')
          print("Week %s" % self.week)
          print("Agent N q-values before receiving reward %s"
                 % self.agents[-1].action_q_values)
          print("Agent N action %s" % self.agents[-1].action)
          print("Bar attendances %s" % self.attendances)
          print("Bar results %s" % self.bar_results)
          print("List of rewards %s" % self.rewards)
          print("Exploration probability %s" % self.p)
        else:
          print("Agent N q-values after receiving reward %s"
                 % self.agents[-1].action_q_values)
          if USE_WLU:
            print("World utility: %s" % self.G)
            print("World utility w/o Agent N: %s" % self.GN)

  def update_agents_wlu(self):
    """
    for each agent removes him from the agent_set and recalculates the world utility,
    then updates Q-table with the WLU
    """
    for _ in xrange(NR_AGENTS):
      reserva = self.agents.pop(0)
      self.GN = self.G - self.calculate_world_utility(self.agents)
      reserva.update_utilities(self.GN)
      self.agents.append(reserva)

    # this last operation recalculates the rewards and attendances for the
    # complete agent set; this is a flaw in the design of this algorithm in the
    # sense that this extra run over it is not needed but it don't seem to be
    # worth correcting for now
    self.G = self.calculate_world_utility(self.agents)

  def update_agents(self):
    """
    Simply forwards the rewads to the agent
    """
    for agent in self.agents:
        agent.update_utilities(self.rewards[agent.action])

class Experiment:

  def __init__(self, nr_weeks=5000, p=1.0, alpha=.01, thresholds=[0.3,0.5],
               nr_agents=100, use_wlu=False, debug=False, decay=None,
               init_q_value="zeros", continuous=True):
    """
    nr_weeks: number of weeks as an integer
    p: the exploration probability
    alpha: the learning rate
    thresholds: list of THRESHOLDS for each bar
    nr_agents: number of agents as an integer
    use_wlu: whether or not the reward is the wonderful life utility
    debug: if information should be printed at every 500 steps
    """
    global INITIAL_EXPLORATION_CHANCE, ALPHA, THRESHOLDS, DECAY_FUNCTION
    global NR_AGENTS, USE_WLU, NR_WEEKS, DEBUG, NR_ACTIONS, NR_BARS
    global INITIAL_Q_VALUES, HOME_GOOD_CONT_FN, CONTINUOUS, REWARD_CONT

    DECAYS = {
      "exponential": lambda x, w: x*.999,
      None: lambda x, w: x,
      "partial": lambda x, w: x if w < NR_WEEKS/2.0 else 0.0
    }

    ALPHA = alpha
    INITIAL_EXPLORATION_CHANCE = p
    THRESHOLDS = thresholds
    NR_AGENTS = nr_agents
    NR_WEEKS = nr_weeks
    USE_WLU = use_wlu
    DEBUG = debug
    NR_BARS = len(THRESHOLDS)
    NR_ACTIONS = NR_BARS + 1
    DECAY_FUNCTION = DECAYS[decay]
    CONTINUOUS = continuous
    HOME_GOOD_CONT_FN = lambda x: np.tanh(x-NR_BARS)

    # GenLogistic Probability density function adapted to mode = threshold
    B = 0.02 #SCALE
    C = 0.2 #SHAPE
    REWARD_CONT = lambda x, t: ((C/B)*np.exp(-((x-(t-B*np.log(C)))/B))*(1+np.exp(-((x-(t-B*np.log(C)))/B)))**(-C-1))*MAXREWARD/6.0

    Q_VALUES_INITIALIZERS = {
        "zeros": lambda: np.zeros(NR_ACTIONS),
        "rand": lambda: np.random.random_sample(NR_ACTIONS)*MAXREWARD,
    }

    INITIAL_Q_VALUES = Q_VALUES_INITIALIZERS[init_q_value]

    print(" --- Running experiment ---")
    if DEBUG:
      print("DEBUG MODE: %s" % DEBUG)
      print(" Parameters: ")

      print("LEARNING RATE: %s" % ALPHA)
      print("THRESHOLDS: %s" % THRESHOLDS)
      print("NUMBER OF AGENTS: %s" % NR_AGENTS)
      print("NUMBER OF BARS: %s" % NR_BARS)
      print("NUMBER OF WEEKS: %s" % NR_WEEKS)
      print("INITIAL EXPLORATION CHANCE %s" % INITIAL_EXPLORATION_CHANCE)
      print("DECAY: %s" % (decay if decay else "No"))
      print("REWARD TYPE: %s" % ("WLU" if USE_WLU else "REGULAR"))

  def run(self):
    """
    runs the experiment
    returns:
        weekly_attendance (NR_ACTIONSxNR_WEEKS) the attendance at every week
            for each action
        world_utilities (NR_WEEKS) the world_utility at every week
        agent_q_values (NR_AGENTSxNR_ACTIONSxNR_WEEKS) the agent's q values
            for each action for each week
        rewards (NR_AGENTSxNR_ACTIONSxNR_WEEKS) the agent's rewards
            for each action for each week
        actions (NR_AGENTSxNR_WEEKS) the action of each agent, each week
    """

    weekly_attendance = np.zeros((NR_ACTIONS,NR_WEEKS))
    world_utilities = np.zeros(NR_WEEKS)
    agent_q_values = np.zeros((NR_AGENTS, NR_ACTIONS, NR_WEEKS))
    rewards = np.zeros((NR_AGENTS, NR_ACTIONS, NR_WEEKS))
    actions = np.zeros((NR_AGENTS, NR_WEEKS))

    w = World()

    for week in xrange(NR_WEEKS):
      w.step()
      world_utilities[week] = w.G
      for action in xrange(NR_ACTIONS):
        weekly_attendance[action][week] = w.relative_attendances(NR_AGENTS)[action]
        for ag_ix,_ in enumerate(w.agents):
          q_value = w.agents[ag_ix].action_q_values[action]
          agent_q_values[ag_ix][action][week] = q_value
          rewards[ag_ix][action][week] = w.rewards[w.agents[ag_ix].action]
          actions[ag_ix][week] = w.agents[ag_ix].action

    return weekly_attendance, world_utilities, agent_q_values, rewards, actions

  def plot_attendances(self, attendances, ylim=1):
    """
    data: matrix of NUMBER_OF_BARS x NUMBER_OF_WEEKS
    """
    x = np.linspace(0,NR_WEEKS-1,NR_WEEKS)
    f, axis = plt.subplots(nrows=NR_ACTIONS)
    for index, ax in enumerate(axis):
      ax.scatter(x, [attendances[index][k] for k in x], s=3)
      ax.set_title("bar" + str(index) if index != 0 else "stay")
      if index != 0:
        ax.axhline(y=THRESHOLDS[index-1]-(0.05 if NR_AGENTS > 100 else 0.1), c='r')
        ax.axhline(y=THRESHOLDS[index-1], c='r')
      ax.set_ylim(0, ylim)
      ax.set_xlim(0, NR_WEEKS)
      ax.set_yticks(np.arange(0,ylim,0.1))
      ax.set_xticks(np.arange(0,NR_WEEKS,NR_WEEKS/10))
      ax.set_ylabel("Attendance")
      ax.set_xlabel("Weeks")
      ax.grid()

    f.set_size_inches(10.5, NR_BARS*5)
    f.tight_layout()

  def plot_world_utilities(self, world_utilities):
    """
    world_utilities: list of world utilities of size NR_WEEKS
    """
    axis = plt.subplot(111)
    x = np.linspace(0,NR_WEEKS-1,NR_WEEKS)
    axis.scatter(x, world_utilities, s=1)

  def plot_q_values(self, agent_q_values, ylim=None):
    """
    agent_q_values: matrix of shape NR_AGENTSxNR_ACTIONSxNR_WEEKS
    """
    col_gen = cycle('bgrcmk')
    linestyles_gen = cycle(['-'])
    cols = [col_gen.next() for _ in xrange(NR_ACTIONS)]
    linestyles = [linestyles_gen.next() for _ in xrange(NR_ACTIONS)]

    x = np.linspace(0,NR_WEEKS-1,NR_WEEKS)
    f, axis = plt.subplots(ncols=agent_q_values.shape[0])

    for agent, ax in enumerate(axis):
      for action in xrange(NR_ACTIONS):
        ax.plot(x, agent_q_values[agent][action],
                c=cols[action], ls=linestyles[action],
                label='stay' if action == 0 else 'bar%i'% action )
        ax.set_ylabel("Q-Value")
        ax.set_xlabel("Weeks")
        if ylim:
          ax.set_ylim(-4000, ylim)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                  ncol=3, mode="expand", borderaxespad=0.)
    f.set_size_inches(10, 3)
    f.tight_layout()

  def plot_rewards(self, rewards):
    """
    rewards: matrix of shape NR_AGENTSxNR_ACTIONSxNR_WEEKS
    """
    col_gen = cycle('bgrcmk')
    linestyles_gen = cycle(['-'])
    cols = [col_gen.next() for _ in xrange(NR_ACTIONS)]
    linestyles = [linestyles_gen.next() for _ in xrange(NR_ACTIONS)]

    x = np.linspace(0,NR_WEEKS-1,NR_WEEKS)
    f, axis = plt.subplots(ncols=rewards.shape[0])

    for agent, ax in enumerate(axis):
      for action in xrange(NR_ACTIONS):
        ax.scatter(x, rewards[agent][action], color=cols[action],
               label='stay' if action == 0 else 'bar%i'% action, s=1)
        ax.set_ylabel("Rewards")
        ax.set_xlabel("Weeks")
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                  ncol=3, mode="expand", borderaxespad=0.)
    f.set_size_inches(10, 3)
    f.tight_layout()
