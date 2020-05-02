class Environment:
    def __init__(self):
        raise NotImplementedError()

    def move(self, state, action):
        """
        Move given state and action
        """
        raise NotImplementedError()

    def reward(self, state, action, next_state):
        """
        Reward given state, action and next_state
        """
        raise NotImplementedError()

    def prob(self, state, action, next_state):
        """
        Probability given state, action and next_state
        """
        raise NotImplementedError()
