import argparse
from pprint import pprint

import rl
from env import wumpus


def run(algo_cls, H, gamma, wumpus_config):
    env = wumpus.WumpusEnvironment(**wumpus_config)
    env.display_action_legends()
    iterator = algo_cls(env, gamma)
    v_or_pi = iterator.run(H)
    return iterator, env, v_or_pi


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

    iterator, env, v_or_pi = run(rl.PolicyIter, 20, 0.7, wumpus_config)
    import numpy as np
    pi = []
    for i in range(env.n_states):
        pi.append(np.argmax(v_or_pi[i]))
    
    print('\n\nFinal Policy')
    print('Reward: {}:'.format(env.calculate_total_reward(pi)))
    env.display_policy(pi)
