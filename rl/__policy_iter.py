import numpy as np

from .__reinforcement_learner import ReinforcementLearnerBase

class PolicyIter(ReinforcementLearnerBase):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        self.__is_ran = False
        self.__pi_star = np.ones(
            (self.env.n_states, self.env.n_actions)) / self.env.n_actions

    def run(self, H):
        for i in range(H):
            is_policy_good = False
            V = self.__eval_policy()
            for state in range(self.env.n_states):
                action = np.argmax(self.__pi_star[state])
                values = self.__1step_lookahead(state, V)
                best_action = np.argmax(values)
                if action != best_action:
                    is_policy_good = True
                    self.__pi_star[state] = np.eye(self.env.n_actions)[best_action]
            if is_policy_good:
                break
        self.__is_ran = True

    def next_action(self, state):
        assert self.__is_ran, "Can't get next action without fitting the model. First call PolicyIter().run()"
        return np.argmax(self.__pi_vals[state, :])
