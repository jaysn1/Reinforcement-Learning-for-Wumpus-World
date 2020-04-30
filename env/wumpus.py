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

    def __init_states(self):
        self.states = []
        for i in range(self.width):
            for j in range(self.height):
                self.states.append((i, j))

    def __get_neighbors_going_up(self, state):
        return [(-1, 0), (0, -1), (1, 0)]

    def __get_neighbors_going_down(self, state):
        return [(1, 0), (0, -1), (1, 0)]

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
        wumpus_x, wumpus_y = self.wumpus_loc
        state_x, state_y = state
        if wumpus_x < state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif wumpus_x > state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif wumpus_x == state_x and wumpus_y > state_y:
            self.wumpus_alive = False
        elif wumpus_x == state_x and wumpus_y < state_y:
            self.wumpus_alive = False

    def move(self, state, action):
        state = self.states[state]
        if action == WumpusEnvironment.UP:
            next_states = []
            for neig in self.__get_neighbors_going_up(state):
                next_state = (state[0] + neig[0], state[1] + neig[1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.DOWN:
            next_states = []
            for neig in self.__get_neighbors_going_down(state):
                next_state = (state[0] + neig[0], state[1] + neig[1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.RIGHT:
            next_states = []
            for neig in self.__get_neighbors_going_right(state):
                next_state = (state[0] + neig[0], state[1] + neig[1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.LEFT:
            next_states = []
            for neig in self.__get_neighbors_going_left(state):
                next_state = (state[0] + neig[0], state[1] + neig[1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.CLIMB:
            if state == self.start:
                return []
        if action == WumpusEnvironment.SHOOT:
            self.__shoot(state)
            return [self.states.index(state)]
        if action == WumpusEnvironment.GRAB and state == self.goal:
            self.gold_collected = True
            return [self.states.index(next_state)]
        return [self.states.index(state)]

    def prob(self, state, action, next_state):
        if action == WumpusEnvironment.UP:
            state_x, state_y = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_up(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1]) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1]) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1]) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.DOWN:
            state_x, state_y = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_down(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1]) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1]) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1]) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.LEFT:
            state_x, state_y = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_left(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1]) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1]) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1]) == next_state):
                return self.noise / 2
        if action == WumpusEnvironment.RIGHT:
            state_x, state_y = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_right(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1]) == next_state:
                return 1 - self.noise
            elif ((state_x + neighs[1][0], state_y + neighs[1][1]) == next_state or 
                  (state_x + neighs[2][0], state_y + neighs[2][1]) == next_state):
                return self.noise / 2
        if state == next_state:
            return 1.
        return 0.

    def reward(self, state, action, next_state):
        if self.states[state] in self.pit_locs:
            return -100
        if self.wumpus_alive and self.states[state] == self.wumpus_loc:
            return -100
        if action == WumpusEnvironment.SHOOT:
            return -20
        if not self.wumpus_alive and self.states[state] == self.wumpus_loc:
            return -10
        if self.states[state] == self.goal:
            return 10
        if self.states[state] == self.start and self.gold_collected and action == WumpusEnvironment.CLIMB:
            return 1000
        return -10
