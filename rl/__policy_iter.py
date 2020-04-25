from .__reinforcement_learner import ReinforcementLearnerBase

class PolicyIter(ReinforcementLearnerBase):
    def __init__(self, env):
        super().__init__(env)
        self.__pi_star

    def run(self):
        pass

    def next_action(self, state):
        pass