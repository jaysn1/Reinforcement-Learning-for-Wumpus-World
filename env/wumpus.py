from .__base import Environment


class WumpusEnvironment(Environment):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    SHOOT = 3
    GRAB = 4
    CLIMB = 5
    STOP = 6

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
        self.agent_heading = WumpusEnvironment.NORTH
        self.wumpus_alive = True
        self.gold_collected = False

    def __init_states(self):
        self.states = []
        for i in range(width):
            for j in range(height):
                self.states.append((i, j))

    def __get_neighbors_going_forward(self, state):
        if self.agent_heading == WumpusEnvironment.NORTH:
            return [(-1, 0), (0, -1), (1, 0)]
        if self.agent_heading == WumpusEnvironment.SOUTH:
            return [(1, 0), (0, -1), (1, 0)]
        if self.agent_heading == WumpusEnvironment.EAST:
            return [(0, 1), (-1, 0), (1, 0)]
        return [(0, -1), (-1, 0), (1, 0)]

    def __turn_left(self):
        if self.agent_heading == WumpusEnvironment.NORTH:
            self.agent_heading = WumpusEnvironment.EAST
        elif self.agent_heading == WumpusEnvironment.SOUTH:
            self.agent_heading = WumpusEnvironment.WEST
        elif self.agent_heading == WumpusEnvironment.EAST:
            self.agent_heading = WumpusEnvironment.SOUTH
        else:
            self.agent_heading = WumpusEnvironment.NORTH

    def __turn_right(self):
        if self.agent_heading == WumpusEnvironment.NORTH:
            self.agent_heading = WumpusEnvironment.WEST
        elif self.agent_heading == WumpusEnvironment.SOUTH:
            self.agent_heading = WumpusEnvironment.EAST
        elif self.agent_heading == WumpusEnvironment.EAST:
            self.agent_heading = WumpusEnvironment.NORTH
        else:
            self.agent_heading = WumpusEnvironment.SOUTH

    def __shoot(self, state):
        wumpus_x, wumpus_y = self.wumpus_loc
        state_x, state_y = self.states[state]
        if self.agent_heading == WumpusEnvironment.NORTH and wumpus_x < state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif self.agent_heading == WumpusEnvironment.SOUTH and wumpus_x > state_x and wumpus_y == state_y:
            self.wumpus_alive = False
        elif self.agent_heading == WumpusEnvironment.EAST and wumpus_x == state_x and wumpus_y > state_y:
            self.wumpus_alive = False
        elif self.agent_heading == WumpusEnvironment.WEST and wumpus_x == state_x and wumpus_y < state_y:
            self.wumpus_alive = False

    def move(self, state, action):
        state = self.states[state]
        if action == WumpusEnvironment.FORWARD:
            next_states = []
            for neig in self.__get_neighbors_going_forward(state):
                next_state = (state[0] + neig[0], state[1] + neig[1])
                if next_state in self.states:
                    next_states.append(self.states.index(next_state))
            return next_states
        if action == WumpusEnvironment.TURN_LEFT:
            self.__turn_left()
            return [state]
        if action == WumpusEnvironment.TURN_RIGHT:
            self.__turn_right()
            return [state]
        if action == WumpusEnvironment.CLIMB:
            if self.states[state] == self.start:
                return []
        if action == WumpusEnvironment.SHOOT:
            self.__shoot(state)
            return [state]
        if action == WumpusEnvironment.GRAB and self.states[state] == self.goal:
            self.gold_collected = True
            return [state]
        return [state]

    def probs(self, state, action, next_state):
        if action == WumpusEnvironment.FORWARD:
            state_x, state_y = self.states[state]
            next_state = self.states[next_state]
            neighs = self.__get_neighbors_going_forward(state)
            if (state_x + neighs[0][0], state_y + neighs[0][1]) == next_state:
                return 0.8
            else:
                return 0.1
        if state == next_state:
            return 1.
        return 0.

    def reward(self, state, action, next_state):
        if self.states[state] in self.pit_locs:
            return -1000
        if self.wumpus_alive and self.states[state] == self.wumpus_loc:
            return -1000
        if not self.wumpus_alive and self.states[state] == self.wumpus_loc:
            return -1
        if self.states[state] == self.goal:
            return 1000
        if self.states[state] == self.start and self.gold_collected and action == WumpusEnvironment.CLIMB:
            return 1000
        return -1
