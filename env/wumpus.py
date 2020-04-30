from .__base import Environment


class WumpusEnvironment(Environment):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    SHOOT = 4
    GRAB = 5
    CLIMB = 6
    STOP = 7

    def __init__(self,
                 width,
                 height,
                 start,
                 wumpus_loc,
                 pit_locs,
                 goal,
                 noise):
        self.width = width
        self.height = height
        self.start = start
        self.wumpus_loc = wumpus_loc
        self.pit_locs = pit_locs
        self.goal = goal
        self.noise = noise
        self.__init_states()
        self.agent_loc = self.start
        self.wumpus_alive = True
        self.gold_collected = False
        self.has_arrow = True
        self.__reward_first_call_after_shoot = False
        self.n_states = len(self.states)
        self.n_actions = 8
        self.__wumpus_just_died = False
        self.calc_reward = False

    def __init_states(self):
        self.states = []
        for i in range(self.width):
            for j in range(self.height):
                self.states.append((i, j, False))
        for i in range(self.width):
            for j in range(self.height):
                self.states.append((i, j, True))

    def __get_neighbors_going_up(self, state):
        return [(-1, 0), (0, -1), (0, 1)]

    def __get_neighbors_going_down(self, state):
        return [(1, 0), (0, -1), (0, 1)]

    def __get_neighbors_going_right(self, state):
        return [(0, 1), (-1, 0), (1, 0)]

    def __get_neighbors_going_left(self, state):
        return [(0, -1), (-1, 0), (1, 0)]

    def __get_neighbors_going_forward(self, state):
        if self.agent_heading == WumpusEnvironment.NORTH:
            return [(-1, 0), (0, -1), (1, 0)]
        if self.agent_heading == WumpusEnvironment.SOUTH:
            return [(1, 0), (0, -1), (1, 0)]
        if self.agent_heading == WumpusEnvironment.EAST:
            return [(0, 1), (-1, 0), (1, 0)]
        return [(0, -1), (-1, 0), (1, 0)]

    def __shoot(self, state):
        self.has_arrow = False
        self.__reward_first_call_after_shoot = True
        last_wumpus_state = self.wumpus_alive
        wumpus_x, wumpus_y = self.wumpus_loc
        state_x, state_y, _ = state
        if wumpus_x < state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif wumpus_x > state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif wumpus_x == state_x and wumpus_y > state_y:
            self.wumpus_alive = False
        elif wumpus_x == state_x and wumpus_y < state_y:
            self.wumpus_alive = False
        self.__wumpus_just_died = last_wumpus_state and not self.wumpus_alive

    def move(self, state, action):
        state = self.states[state]
        if action == WumpusEnvironment.UP:
            next_states = []
            for neig in self.__get_neighbors_going_up(state):
                next_state = (state[0] + neig[0], state[1] + neig[1], state[-1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.DOWN:
            next_states = []
            for neig in self.__get_neighbors_going_down(state):
                next_state = (state[0] + neig[0], state[1] + neig[1], state[-1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.RIGHT:
            next_states = []
            for neig in self.__get_neighbors_going_right(state):
                next_state = (state[0] + neig[0], state[1] + neig[1], state[-1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.LEFT:
            next_states = []
            for neig in self.__get_neighbors_going_left(state):
                next_state = (state[0] + neig[0], state[1] + neig[1], state[-1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.CLIMB:
            if state == self.start:
                return []
        if action == WumpusEnvironment.SHOOT:
            self.__shoot(state)
            return [self.states.index(state)]
        if self.calc_reward and action == WumpusEnvironment.GRAB and list(state[:-1]) == self.goal:
            self.gold_collected = True
            next_state = (state[0], state[1], True)
            return [self.states.index(next_state)]
        return [self.states.index(state)]

    def prob(self, state, action, next_state):
        if action == WumpusEnvironment.UP:
            state_x, state_y, gold_collected = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_up(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1], gold_collected) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1], gold_collected) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1], gold_collected) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.DOWN:
            state_x, state_y, gold_collected = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_down(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1], gold_collected) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1], gold_collected) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1], gold_collected) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.LEFT:
            state_x, state_y, gold_collected = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_left(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1], gold_collected) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1], gold_collected) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1], gold_collected) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.RIGHT:
            state_x, state_y, gold_collected = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_right(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1], gold_collected) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1], gold_collected) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1], gold_collected) == next_state):
                return self.noise / 2
        if state == next_state:
            return 1.
        return 0.

    def reward(self, state, action, next_state):
        if list(self.states[state][:-1]) in self.pit_locs:
            return -100
        # if self.wumpus_alive and list(self.states[next_state][:-1]) == self.wumpus_loc:
        #     return -250
        if self.wumpus_alive and list(self.states[state][:-1]) == self.wumpus_loc:
            return -100
        if action == WumpusEnvironment.SHOOT:
            return -20
        if not self.wumpus_alive and list(self.states[state][:-1]) == self.wumpus_loc:
            return -15
        if action == WumpusEnvironment.CLIMB and not self.states[state][-1]:
            return -100
        if action == WumpusEnvironment.STOP:
            return -20
        if list(self.states[state][:-1]) == self.start and self.states[state][-1] and action == WumpusEnvironment.CLIMB:
                return 500
        if list(self.states[state][:-1]) == self.goal and not self.states[state][-1] and action == WumpusEnvironment.GRAB:
            return 500
        if action == WumpusEnvironment.GRAB or action == WumpusEnvironment.CLIMB:
            return -15
        return -15

    def display(self):
        print(' ' * 6, end='')
        for i in range(self.width):
            print('|{:^5s}'.format(str(i)), end='')
        print('|')
        print('-'*6*(self.width+1) + '-')
        for y in range(self.height):
            print('|{:>5s}'.format(str(y)), end='')
            for x in range(self.width):
                if [y, x] == self.start:
                    print('|{:^5s}'.format('S'), end='')
                elif [y, x] == self.wumpus_loc:
                    print('|{:^5s}'.format('W'), end='')
                elif [y, x] in self.pit_locs:
                    print('|{:^5s}'.format('P'), end='')
                elif [y, x] == self.goal:
                    print('|{:^5s}'.format('G'), end='')
                else:
                    print('|' + ' ' * 5, end='')
            print('|')

    def display_policy(self, pi):
        print('Before Collecting Gold:')
        self.__display_policy(pi)
        print('After Collecting Gold:')
        self.__display_policy(pi, True)

    def __display_policy(self, pi, gold_collected=False):
        actions = ['U', 'D', 'L', 'R', 'S' , 'G', 'C', 'X']
        print(' ' * 6, end='')
        for i in range(self.width):
            print('|{:^5s}'.format(str(i)), end='')
        print('|')
        print('-'*6*(self.width+1) + '-')
        for y in range(self.height):
            print('|{:>5s}'.format(str(y)), end='')
            for x in range(self.width):
                state_i = self.states.index((y, x, gold_collected))
                print('|{:^5s}'.format(actions[pi[state_i]]), end='')
            print('|')

    def display_action_legends(self):
        print('Actions and Abbreviations looking at Policy:')
        print('{:10s}{:10s}'.format('U - UP', 'D - Down'))
        print('{:10s}{:10s}'.format('L - LEFT', 'D - RIGHT'))
        print('{:10s}{:10s}'.format('G - GRAB', 'S - SHOOT'))
        print('{:10s}{:10s}'.format('C - CLIMB', 'X - STOP'))
        print('Things and Abbreviations looking at Env:')
        print('{:10s}'.format('S - Start Point'))
        print('{:10s}{:10s}'.format('G - GOLD', 'P - PIT'))
        print('{:10s}{:10s}'.format('W - WUMPUS', 'X - STOP'))

    def calculate_total_reward(self, pi, max_steps=100):
        agent_loc = (self.start[0], self.start[1], False)
        wumpus_alive = self.wumpus_alive
        r = 0
        ctr = 0
        self.calc_reward = True
        while True:
            action = pi[self.states.index(agent_loc)]
            next_states = self.move(self.states.index(agent_loc), action)
            r += self.reward(self.states.index(agent_loc), action, next_states[0])
            agent_loc = self.states[next_states[0]]
            if list(agent_loc[:-1]) == self.start and action == WumpusEnvironment.CLIMB:
                self.calc_reward = False
                return r 
            ctr += 1
            if ctr == max_steps:
                self.calc_reward = False
                return r
