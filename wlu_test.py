import wlu
import pytest

def test_reward():
    e = wlu.Experiment(nr_agents=101)
    w = wlu.World()
    for x in range(26,36):
        rw = w.get_reward(x/101.0, 0.3)
        assert(rw > 0)
