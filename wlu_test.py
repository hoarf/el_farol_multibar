import wlu
import numpy as np
import pytest

def test_reward():
    e = wlu.Experiment(nr_agents=101)
    w = wlu.World()
    for x in range(26,36):
        rw = w.get_reward(x/101.0, 0.3)
        assert(rw > 0)

def test_wlu_4_agents():
    e = wlu.Experiment(nr_agents=4, use_wlu=True)
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
