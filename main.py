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
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help="Algorithm to use - Value Iteration or Policy Iteration [p|V]",
                        default='P')
    parser.add_argument('--gamma',  type=float, help="Gamma Value, Default 0.9",
                        default=0.9)
    parser.add_argument('--noise',  type=float, help="Noise Value, Default 0.2",
                        default=0.2)
    parser.add_argument('--h',  type=int, help="Horizon Value, Default 20",
                        default=20)
    parser.add_argument('--width', type=int, help='Width of the world, default 4', default=4)
    parser.add_argument('--height', type=int, help='height of the world, default 4', default=4)
    parser.add_argument('--start', type=int, nargs='+', help="The start position of agent.")
    parser.add_argument('--goal', type=int, nargs='+', help="Location of gold")
    parser.add_argument('--wumpus', type=int, nargs='+', help="Wumpus location.")
    parser.add_argument('--pit', type=int, nargs='+', action='append',
                        help="Pit location to add multiple use it multiple times.")

    args = parser.parse_args()
    wumpus_config = {
        'width': args.width,
        'height': args.height,
        'start': args.start,
        'wumpus_loc': args.wumpus,
        'pit_locs': args.pit,
        'goal':args.goal,
        'noise': args.noise
    }

    if args.algo.upper() == 'P':
        algo = rl.PolicyIter
    else:
        algo = rl.ValueIter
    iterator, env, v_or_pi = run(algo, args.h, args.gamma, wumpus_config)
    import numpy as np
    pi = []
    for i in range(env.n_states):
        pi.append(np.argmax(v_or_pi[i]))
    
    print('\n\nFinal Policy')
    print('Reward: {}:'.format(env.calculate_total_reward(pi)))
    env.display_policy(pi)
