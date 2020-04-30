import argparse
from pprint import pprint

import rl
from env import wumpus


def run(algo_cls, H, gamma, wumpus_config):
    env = wumpus.WumpusEnvironment(**wumpus_config)
    iterator = algo_cls(env, gamma)
    v_or_pi = iterator.run(H)
    pprint(v_or_pi)
    return iterator, env


if __name__ == '__main__':
    wumpus_config = {
        'width': 4,
        'height': 4,
        'start': [3, 0],
        'wumpus_loc': [1, 0],
        'pit_locs': [[1, 1], [3, 2]],
        'goal':[1, 2],
        'noise': 0.2
    }

    iterator, env = run(rl.ValueIter, 1000, 0.9, wumpus_config)