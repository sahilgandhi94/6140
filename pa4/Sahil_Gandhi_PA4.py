import argparse
from copy import deepcopy

import numpy as np

import gridworld as gw


class RL:

    def __init__(self, alg, size, gamma, n_experiments,
                n_episodes, epsilon, alpha, sarsa_param):
        self.alg = alg
        self.grid_size = size
        self.gamma = gamma
        self.n_experiments = n_experiments
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.sarsa_param = sarsa_param
        
        self.n_actions = 4  # hard-coded 
        self._grid = self._initialize_grid()
        self.Q = None
        self.E = None  # Eligibility choice for lambda algorithms

    def run(self):
        """ Runs the experiments based on the set params """
        if self.alg == 'q':
            self._q_learning()
        elif self.alg == 's':
            self._sarsa()
        else:
            raise ValueError('Unknown value for `alg` found `{}`'.format(self.alg))

    def _initialize_grid(self):
        """ Initializes the grid based on the `grid_size` """
        return gw.GridworldEnv(shape=[self.grid_size, self.grid_size])

    def _get_initial_agent_state(self):
        """ Returns the initial position of agent in the grid """
        return self.grid_size**2 - self.grid_size

    def _initialize_q_values(self):
        """ Returns a Q-value data structure where all q-values are initialized to 0

            The Q-value data structure looks like:
            {
                <state>: {
                    <action#>: <q-value>,
                    ...
                },
                ...
            }
        """
        _actions = {actn: 0 for actn in range(self.n_actions)}
        return {state: deepcopy(_actions) for state in range(self.grid_size**2)}

    def _evaluate_action_with_env_probabilities(self, action):
        """ The possible action has a 0.8 probability of succeeding """
        assert action in [gw.UP, gw.DOWN, gw.LEFT, gw.RIGHT], \
        'Unsupported action `{}`'.format(action)
        return self._grid.get_action(action)

    def _take_action(self, state, action):
        """ Given the current position, this function outputs the position after the action """
        assert action in [gw.UP, gw.DOWN, gw.LEFT, gw.RIGHT], \
        'Unsupported action `{}`'.format(action)
        # added the 0th index because gridworld.move returns an array of one element
        next_state, reward, is_done = self._grid.move(state, action)[0]
        return (next_state, reward, is_done)

    def _choose_action_from_policy(self, state):
        """ Chooses the action based on 
            - (1-epsilon) probability from chosen policy, i.e. argmax(a, Q-values)
            - epsilon probability of random action """
        random_choice = np.random.uniform() < self.epsilon
        if random_choice:
            action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
        else:
            actions_q_values = self.Q[state]
            action = max(actions_q_values, key=actions_q_values.get)
        return action

    def _q_value_computation_q_learning(self, curr_state, next_state, reward, action):
        """ Compute running average Q(s,a) using Q-learning algorithm"""        
        self.Q[curr_state][action] = (((1 - self.alpha) * self.Q[curr_state][action]) + 
                (self.alpha * (reward + (self.gamma * max(self.Q[next_state].values())))))

    def _q_value_computation_sarsa(self, curr_state, next_state, reward, 
                                   curr_action, next_action):
        """ Compute running average Q(s,a) and E(s,a) using Sarsa(lambda) algorithm"""
        delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[curr_state][curr_action]
        self.E[curr_state][curr_action] += 1

        for s in range(self.grid_size**2):
            for a in range(self.n_actions):
                self.Q[s][a] += (self.alpha * delta * self.E[s][a])
                self.E[s][a] *= (self.gamma * self.sarsa_param)

    def _is_exit_from_terminal_state(self, curr_state, next_state, curr_is_done, next_is_done):
        """ Returns if the action is an exit-action from terminal state """
        return next_is_done and curr_is_done and (next_state == curr_state)

    def _q_learning(self):
        """ Executes 1 experiment of q-learning algorithm """
        self.Q = self._initialize_q_values()
        for i in range(self.n_episodes):
            curr_state = self._get_initial_agent_state()
            curr_is_done = False
            # print(i, '+'*15)
            step_count = 0
            while True:
                step_count += 1
                chosen_action = self._choose_action_from_policy(curr_state)
                action = self._evaluate_action_with_env_probabilities(chosen_action)
                next_state, reward, next_is_done = self._take_action(curr_state, action)
                self._q_value_computation_q_learning(curr_state,next_state, reward, action)
                # print(step_count, curr_state, action, reward, curr_is_done)
                if self._is_exit_from_terminal_state(curr_state, next_state, curr_is_done, next_is_done):
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done

    def _sarsa(self):
        """ Executes 1 experiment of sarsa(lambda) algorithm """
        self.Q = self._initialize_q_values()
        for i in range(self.n_episodes):
            self.E = self._initialize_q_values()
            curr_state = self._get_initial_agent_state()
            curr_is_done = False
            chosen_action = self._choose_action_from_policy(curr_state)
            action = self._evaluate_action_with_env_probabilities(chosen_action)
            # print(i, '+'*15)
            step_count = 0
            while True:
                step_count += 1
                # chosen_action = self._choose_action_from_policy(curr_state)
                # action = self._evaluate_action_with_env_probabilities(chosen_action)
                next_state, reward, next_is_done = self._take_action(curr_state, action)
                chosen_next_action = self._choose_action_from_policy(next_state)
                next_action = self._evaluate_action_with_env_probabilities(chosen_next_action)
                self._q_value_computation_sarsa(curr_state, next_state, reward, action, next_action)
                # print(step_count, curr_state, action, reward, curr_is_done)
                if self._is_exit_from_terminal_state(curr_state, next_state, curr_is_done, next_is_done):
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done
                    # need to still re-evaluate the success of the chosen action due to env probabilities
                    action = self._evaluate_action_with_env_probabilities(next_action)

    def _n_experiments_q_learning(self):
        """ Runs q-learning n times and returns per-experiment parameters """
        raise NotImplementedError

    def _n_experiments_sarsa(self):
        """ Runs sarsa(lambda) n times and returns per-experiment parameters """
        raise NotImplementedError

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning (PA4) - Sahil Gandhi.')
    parser.add_argument(
        '-alg',# default='q',
        help='The algorithm used. \n`q`: Q-Learning \n`s`: Sarsa Lambda'
    )
    parser.add_argument(
        '-size', default=5,
        help='Size of the gridworld'
    )
    parser.add_argument(
        '-gamma', default=0.99,
        help='The discount factor'
    )
    parser.add_argument(
        '-exps', default=500,
        help='Amount of experiments to run (default: 500)'
    )
    parser.add_argument(
        '-eps', default=500,
        help='# of learning episodes per experiment'
    )
    parser.add_argument(
        '-epsilon', default=0.1,
        help='epsilon for greedy policy execution'
    )
    parser.add_argument(
        '-alpha', default=0.1,
        help='step size'
    )
    parser.add_argument(
        '-lambda', default=0,
        help='the parameter for Sarsa alg'
    )
    cmd_args = parser.parse_args()
    assert cmd_args.alg in ['q', 's'], 'argument `alg` cannot take value `{}`'.format(
        cmd_args.alg
    )

    obj = RL(
        alg = cmd_args.alg,
        size = cmd_args.size,
        gamma = cmd_args.gamma,
        n_experiments = cmd_args.exps,
        n_episodes = cmd_args.eps,
        epsilon = cmd_args.epsilon,
        alpha = cmd_args.alpha,
        sarsa_param = vars(cmd_args)['lambda']
    )
    print('-'*15, 'Before run')
    print(vars(obj))
    obj.run()
    print('-'*15, 'After run')
    print(vars(obj))
