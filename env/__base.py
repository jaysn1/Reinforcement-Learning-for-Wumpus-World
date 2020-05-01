class Environment:
    def __init__(self):
        raise NotImplementedError()

    def move(self, state, action):
        raise NotImplementedError()

    def reward(self, state, action, next_state):
        raise NotImplementedError()

    def prob(self, state, action, next_state):
        raise NotImplementedError()
