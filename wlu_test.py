import wlu
import numpy as np
import pytest

def test_initialization():
    e = wlu.Experiment(init_q_value="rand", nr_agents=2)
    w = wlu.World()
    assert (w.agents[0].action_q_values != w.agents[1].action_q_values).any()
    e = wlu.Experiment(init_q_value="zeros", nr_agents=2)
    w = wlu.World()
    assert (w.agents[0].action_q_values == w.agents[1].action_q_values).all()
    e = wlu.Experiment(init_q_value="zeros", nr_agents=2, debug=False)
    w = wlu.World()
    w.agents[0].chose_action(.5)
    w.agents[0].update_utilities(10)
    assert (w.agents[0].action_q_values != w.agents[1].action_q_values).any()

def test_bar_result():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    w.bar_results = np.zeros(2)
    w.update_bar_results([.6,.1,.3])
    assert (w.bar_results == [1,1]).all()
    w.update_bar_results([.3,.4,.3])
    assert (w.bar_results == [0,1]).all()
    w.update_bar_results([.5,.1,.4])
    assert (w.bar_results == [1,0]).all()
    w.update_bar_results([.3,.3,.4])
    assert (w.bar_results == [0,0]).all()

def test_home_good():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    good = w.is_home_good([0, 1])
    assert not good
    good = w.is_home_good([1, 0])
    assert not good
    good = w.is_home_good([0, 0])
    assert good
    good = w.is_home_good([1, 1])
    assert not good

def test_low_reward():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    r = w.get_reward(.1, .3)
    assert r == 0

def test_high_reward():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    r = w.get_reward(.9, .3)
    assert r == 0

def test_point_five_treshold_reward():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    r = w.get_reward(0.485148514851, .5)
    assert abs(r - 952.51480391) < 1e-2

def test_point_three_treshold_reward():
    e = wlu.Experiment(thresholds=[0.2,0.3], nr_agents=101)
    w = wlu.World()
    r = w.get_reward(0.287128712871, .3)
    assert abs(r - 972.926674737) < 1e-2

def test_return_reward():
    e = wlu.Experiment(nr_weeks = 1)
    w = wlu.World()
    _, _, _, rewards, _ = e.run()
    assert rewards != None

def test_no_decay():
    e = wlu.Experiment(nr_agents=2, decay=None, p=.3)
    w = wlu.World()
    w.step()
    assert w.p == .3

def test_decay():
    e = wlu.Experiment(nr_agents=2, decay="exponential", p=1.0)
    w = wlu.World()
    w.step()
    assert w.p == 1.0*.999

def test_reward():
    e = wlu.Experiment(nr_agents=101)
    w = wlu.World()
    for x in range(26,36):
        rw = w.get_reward(x/101.0, 0.3)
        assert(rw > 0)

def test_discrete_reward_regular():
    e = wlu.Experiment(nr_agents=9, continuous=False)
    w = wlu.World()

    a0,a1,a2,a3,a4,a5,a6,a7,a8 = w.agents

    a8.action = 0
    a0.action, a1.action, a2.action, a3.action, a4.action = 2, 2, 2, 2, 2
    a5.action, a6.action, a7.action = 1, 1, 1

    w.calculate_world_utility(w.agents)
    assert(( w.attendances == [1 , 3 , 5] ).all())
    assert(( w.bar_results == [0 , 0] ).all())
    assert(( w.rewards == [500, 0, 0] ).all())

    w.update_rule()
    assert(( a0.action_q_values == [0, 0, 0] ).all())
    assert(( a1.action_q_values == [0, 0, 0] ).all())
    assert(( a8.action_q_values == [5, 0, 0] ).all())

def test_wlu_4_agents():
    e = wlu.Experiment(nr_agents=4, use_wlu=True, continuous=False)
    w = wlu.World()

    a0 = w.agents[0]
    a1 = w.agents[1]
    a2 = w.agents[2]
    a3 = w.agents[3]

    a0.action = 0
    a1.action = 1
    a2.action = 2
    a3.action = 2

    w.G = w.calculate_world_utility(w.agents)
    assert(w.G == np.mean([0, 679, 1000, 1000]))

    reserva = w.agents.pop(0)
    G = w.calculate_world_utility(w.agents)
    assert(( w.attendances == [0 , 1 , 2] ).all())
    assert(( w.bar_results == [0 , 0] ).all())
    assert(( w.rewards == [500, 0, 0] ).all())
    assert(G == np.mean([0, 0, 0]))

    w.agents.append(reserva)
    reserva = w.agents.pop(0)
    G = w.calculate_world_utility(w.agents)
    assert(( w.attendances == [1 , 0 , 2] ).all())
    assert(( w.bar_results == [1 , 0] ).all())
    assert(( w.rewards == [0, 0, 0] ).all())
    assert(G == np.mean([0, 0, 0]))

    w.agents.append(reserva)
    reserva = w.agents.pop(0)
    G = w.calculate_world_utility(w.agents)
    assert(( w.attendances == [1 , 1 , 1] ).all())
    assert(( w.bar_results == [0 , 1] ).all())
    assert(( w.rewards == [0, 0, 0] ).all())
    assert(G == np.mean([0, 0, 0]))

    w.agents.append(reserva)
    reserva = w.agents.pop(0)
    G = w.calculate_world_utility(w.agents)
    assert(( w.attendances == [1 , 1 , 1] ).all())
    assert(( w.bar_results == [0 , 1] ).all())
    assert(( w.rewards == [0, 0, 0] ).all())
    assert(G == np.mean([0, 0, 0]))

    w.agents.append(reserva)
    G = w.calculate_world_utility(w.agents)
    assert(( w.attendances == [1 , 1 , 2] ).all())
    assert(( w.bar_results == [1 , 1] ).all())
    assert(( w.rewards == [0, 679, 1000] ).all())
    assert(w.G == np.mean([0, 679, 1000, 1000]))

    w.update_rule()
    assert(w.G == np.mean([0, 679, 1000, 1000]))

def test_is_home_good_cont():
    e = wlu.Experiment(thresholds=[0.3,0.5], nr_agents=4, use_wlu=False, continuous=True)
    w = wlu.World()
    assert w.is_home_good([0,0]) == True
    assert w.is_home_good([1,0]) == False
    assert w.is_home_good([0,1]) == False
    assert w.is_home_good([1,1]) == False

def test_reward_continuous():
    e = wlu.Experiment(thresholds=[0.3,0.5], nr_agents=4, use_wlu=False, continuous=True)
    w = wlu.World()
    assert abs(w.reward_function(.2, .3) - 441.893666765) < 1e4
