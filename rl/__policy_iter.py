import numpy as np

from .__reinforcement_learner import ReinforcementLearnerBase

class PolicyIter(ReinforcementLearnerBase):
    """
    Policy Iteration is an algorithm which is dependent on two steps
    - Policy Evaluation and
    - Policy Improvement
    """
    def __init__(self, env, gamma):
        """
        Make a PolicyIter Object
        
        @params
        env - Environment derived from env.Environment()
        gamma - discount factor
        """
        super().__init__(env)
        self.gamma = gamma
        self.__is_ran = False
        self.__pi_star = np.ones(
            (self.env.n_states, self.env.n_actions)) / self.env.n_actions

    def __1step_lookahead(self, state, V):
        values = np.zeros(self.env.n_actions)
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
                        val += prob * P * (R + self.gamma * V[next_state])
                diff = max(0, np.abs(V[state] - val))
                V[state] = val
            if diff < threshold:
                return V
        return V

    def run(self, H, threshold=1e-9):
        """
        Fit the model
        @params:
        H - horizon
        threshold - stopping criteria for policy improvement step.
        """
        for i in range(H):
            V = self.__eval_policy(H, threshold)
            for state in range(self.env.n_states):
                action = np.argmax(self.__pi_star[state])
                values = self.__1step_lookahead(state, V)
                best_action = np.argmax(values)
                if action != best_action:
                    self.__pi_star[state] = np.eye(self.env.n_actions)[best_action]
            if i % 5 == 0:
                pi = []
                for state in range(self.env.n_states):
                    pi.append(np.argmax(self.__pi_star[state]))
                print('\n{}/{}\t Reward: {}:'.format(i+1, H, self.env.calculate_total_reward(pi)))
                self.env.display_policy(pi)
        self.__is_ran = True
        return self.__pi_star

    def next_action(self, state):
        """
        Get the next action given state.
        """
        assert self.__is_ran, "Can't get next action without fitting the model. First call PolicyIter().run()"
        return np.argmax(self.__pi_star[state, :])
