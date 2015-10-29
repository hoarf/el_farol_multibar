import wlu
import numpy as np
import pytest

def test_reward():
    e = wlu.Experiment(nr_agents=101)
    w = wlu.World()
    for x in range(26,36):
        rw = w.get_reward(x/101.0, 0.3)
        assert(rw > 0)

def test_wlu():
    e = wlu.Experiment(nr_agents=51, use_wlu=True)
    w = wlu.World()
    wlu.NR_AGENTS = 2
    wlu.USE_WLU = True
    w.update_rule = w.update_agents_wlu

    a1 = wlu.Agent()
    a2 = wlu.Agent()

    a1.action = 1
    a2.action = 2

    a = [ a1, a2 ]
    w.agents = a
    w.G = w.calculate_world_utility(a)
    assert(w.G == 500)
    reserva = w.agents.pop(0)
    assert(w.calculate_world_utility(a) == 1000)
    w.agents.append(reserva)
    reserva = w.agents.pop(0)
    assert(w.calculate_world_utility(a) == 0)
    w.agents.append(reserva)
    assert(w.calculate_world_utility(a) == 500)

