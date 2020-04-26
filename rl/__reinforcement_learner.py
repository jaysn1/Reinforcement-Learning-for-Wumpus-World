class ReinforcementLearnerBase:
    def __init__(self, env):
        self.env = env
        self.__is_ran = False

    def run(self):
        raise NotImplementedError()

    def next_action(self, state):
        raise NotImplementedError()
