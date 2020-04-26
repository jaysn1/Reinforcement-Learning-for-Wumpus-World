import numpy as np

from .__reinforcement_learner import ReinforcementLearnerBase

class PolicyIter(ReinforcementLearnerBase):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
        self.__is_ran = False
        self.__pi_star = np.ones(
            (self.env.n_states, self.env.n_actions)) / self.env.n_actions

    def __1step_lookahead(self, state, V):
        values = np.zeros(self.env.n_states)
        for action in range(self.env.n_actions):
            next_states = self.env.move(state, action)
            for next_state in next_states:
                P = self.env.prob(state, action, next_state)
                R = self.env.reward(state, action, next_state)
                values[action] += P * (R + self.gamma * V[next_state])
        return values

    def __eval_policy(self, max_iter, threshold):
        V = np.zeros(self.env.n_states)
        for i in range(max_iter):
            for state in range(self.env.n_states):
                val = 0
                for action, prob in enumerate(self.__pi_star[state]):
                    next_states = self.env.move(state, action)
                    for next_state in next_states:
                        P = self.env.prob(state, action, next_state)
                        R = self.env.reward(state, action, next_state)
                        val += prob * P (R + self.gamma * V[next_state])
                diff = max(0, np.abs(V[state] - val))
                V[state] = val
            if diff < threshold:
                return V
        return V

    def run(self, H, threshold=1e-9):
        for i in range(H):
            is_policy_good = False
            V = self.__eval_policy(H, threshold)
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
