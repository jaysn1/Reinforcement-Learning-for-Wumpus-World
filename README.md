# Reinforcement Learning Agent in Wumpus World - Team Reinforced Learners
In this project, multiple different reinforcement learning algorithms including policy iteration and value iteration were run on a noisy Wumpus world and compared to a hybrid logic agent.

## Members
1. Ishan Chavan
2. Piyush Malhotra
3. Jay Sunil Nankani
4. Devan Pratt
## Topic Chosen: Topic 5 Reinforcement Learning Agent in Wumpus World

## Contributions: 
1. Ishan Chavan - Ishan worked on the environment set up
2. Piyush Malhotra - worked on the policy iteration and value iteration.
3. Jay Sunil Nankani - worked primarily on creating a wumpus world that would work with a reinforcement learning agent
4. Devan Pratt - Devan made a prototype of a Q-learning agent in the wumpus world for the group to use as a prototype

## Instructions:

### Dependencies
1. Python 3
2. NumPy


### How to run
To see commands associated with the file:

`python main.py --help`

```
usage: main.py [-h] [--algo ALGO] [--gamma GAMMA] [--noise NOISE] [--h H]
               [--width WIDTH] [--height HEIGHT] [--start START [START ...]]
               [--goal GOAL [GOAL ...]] [--wumpus WUMPUS [WUMPUS ...]]
               [--pit PIT [PIT ...]]

optional arguments:
  -h, --help            show this help message and exit
  --algo ALGO           Algorithm to use - Value Iteration or Policy Iteration
                        [p|V]
  --gamma GAMMA         Gamma Value, Default 0.9
  --noise NOISE         Noise Value, Default 0.2
  --h H                 Horizon Value, Default 20
  --width WIDTH         Width of the world, default 4
  --height HEIGHT       height of the world, default 4
  --start START [START ...]
                        The start position of agent.
  --goal GOAL [GOAL ...]
                        Location of gold
  --wumpus WUMPUS [WUMPUS ...]
                        Wumpus location.
  --pit PIT [PIT ...]   Pit location to add multiple use it multiple times.
```

To run a world:

`python main.py --algo P --width 4 --height 4 --start 3 0 --goal 1 2 --pit 1 1 --pit 3 2 --wumpus 1 0`

The environment associated with this:

```
      |  0  |  1  |  2  |  3  |
-------------------------------
|    0|     |     |     |     |
|    1|  W  |  P  |  G  |     |
|    2|     |     |     |     |
|    3|  S  |     |  P  |     |
```

Here,

```
Things and Abbreviations looking at Env:
S - Start Point
G - GOLD  P - PIT
W - WUMPUS X - STOP
```

Action Abbreviations:
```
Actions and Abbreviations looking at Policy:
U - UP    D - Down
L - LEFT  D - RIGHT
G - GRAB  S - SHOOT
C - CLIMB X - STOP
```