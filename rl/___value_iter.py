import numpy as np

from .__reinforcement_learner import ReinforcementLearnerBase


class ValueIter(ReinforcementLearnerBase):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.__v_vals = np.zeros((env.n_states, env.n_actions))
        self.gamma = gamma

    def __compute_va_pairs(self, state):
        new_v = np.asarray(self.__v_vals)
        for action in self.env.get_actions(state):
            next_state = self.env.move(state, action)
            P = self.env.prob(state, action, next_state)
            R = self.env.reward(state, action, next_state)
            new_v[state, action] = P * (R + self.gamma * self.__v_vals[state, action])
        self.__v_vals[state, :] = new_v

    def run(self, H):
        for i in range(H):
            for state in range(self.env.n_states):
                self.__compute_va_pairs(state)

    def next_action(self, state):
        assert self.is_ran, "Can't get next action without fitting the model. First call ValueIter().run()"
        return np.argmax(self.__v_vals[state, :])