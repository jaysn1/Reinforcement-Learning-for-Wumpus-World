import numpy as np

from .__reinforcement_learner import ReinforcementLearnerBase


class ValueIter(ReinforcementLearnerBase):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.__v_vals = np.zeros((env.n_states, env.n_actions))
        self.gamma = gamma
        self.__is_ran = False

    def __compute_va_pairs(self, state):
        new_v = np.zeros(self.env.n_actions)
        for action in range(self.env.n_actions):
            next_states = self.env.move(state, action)
            for next_state in next_states:
                P = self.env.prob(state, action, next_state)
                R = self.env.reward(state, action, next_state)
                new_v[action] += P * (R + self.gamma * self.__v_vals[state, action])
        self.__v_vals[state, :] = new_v

    def run(self, H):
        for i in range(H):
            for state in range(self.env.n_states):
                self.__compute_va_pairs(state)
            if i % 5 == 0:
                pi = []
                for state in range(self.env.n_states):
                    pi.append(np.argmax(self.__v_vals[state]))
                print('\n{}/{}\t Reward: {}:'.format(i+1, H, self.env.calculate_total_reward(pi)))
                self.env.display_policy(pi)
        self.__is_ran = True
        return self.__v_vals

    def next_action(self, state):
        assert self.__is_ran, "Can't get next action without fitting the model. First call ValueIter().run()"
        return np.argmax(self.__v_vals[state, :])
