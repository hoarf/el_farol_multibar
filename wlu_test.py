import wlu
import pytest

def test_reward():
    w = wlu.World(nr_agents=101)
    for x in range(26,36):
        rw = w.get_reward(x/101.0, 0.3)
        assert(rw > 0)
