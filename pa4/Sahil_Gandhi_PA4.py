import argparse
from copy import deepcopy
from time import time

import numpy as np
from matplotlib import pyplot as plt

import gridworld as gw


class RL:

    def __init__(self, alg, size, gamma, n_experiments,
                n_episodes, epsilon, alpha, sarsa_param):
        self.alg = alg
        self.grid_size = int(size)
        self.gamma = float(gamma)
        self.n_experiments = int(n_experiments)
        self.n_episodes = int(n_episodes)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.sarsa_param = float(sarsa_param)
        
        self.n_actions = 4  # hard-coded 
        self._grid = self._initialize_grid()
        self.Q = None
        self.E = None  # Eligibility choice for lambda algorithms

    def learn(self):
        """ Learns 1 experiment based on the set params """
        if self.alg == 'q':
            self._q_learning()
        elif self.alg == 's':
            self._sarsa()
        else:
            raise ValueError('Unknown value for `alg` found `{}`'.format(self.alg))

    def run(self):
        """ Runs the experiments based on the set params """
        if self.alg == 'q':
            self._n_experiments_q_learning()
        elif self.alg == 's':
            self._n_experiments_sarsa()
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
        return next_is_done and curr_is_done# and (next_state == curr_state)

    def _q_learning(self):
        """ Executes 1 experiment of q-learning algorithm """
        step_counts = []
        q_values_per_episode = []
        log_q = []
        
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
                # if next_is_done:
                    log_q.append(deepcopy(self.Q))
                    # print(step_count, next_state, action, reward, next_is_done)
                    step_counts.append(step_count)
                    q_values_per_episode.append(deepcopy(self.Q))
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done
        # print(log_q)
        return step_counts, q_values_per_episode

    def _sarsa(self):
        """ Executes 1 experiment of sarsa(lambda) algorithm """
        step_counts = []
        q_values_per_episode = []

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
                    step_counts.append(step_count)
                    q_values_per_episode.append(deepcopy(self.Q))
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done
                    # need to still re-evaluate the success of the chosen action due to env probabilities
                    action = self._evaluate_action_with_env_probabilities(next_action)
        return step_counts, q_values_per_episode

    def _n_experiments_q_learning(self):
        """ Runs q-learning n times and returns per-experiment parameters """
        step_counts_per_episode_per_exp = []
        q_values_per_episode_per_exp = []
        t = time()
        start = t
        for i in range(self.n_experiments):
            step_counts, q_values = self._q_learning()
            step_counts_per_episode_per_exp.append(step_counts)
            q_values_per_episode_per_exp.append(q_values)
            t_ = time()
            print('{} experiment completed in {} secs'.format(i+1, round(t_-t, 3)))
            t = t_
        print('Total time taken {} mins'.format(round((time()-start)/60, 3)))
        
        self.plot_time_to_goal_vs_episodes(step_counts_per_episode_per_exp)
        self.plot_maximum_q_value_vs_episodes(q_values_per_episode_per_exp)

    def _n_experiments_sarsa(self):
        """ Runs sarsa(lambda) n times and returns per-experiment parameters """
        step_counts_per_episode_per_exp = []
        q_values_per_episode_per_exp = []
        t = time()
        start = t
        for i in range(self.n_experiments):
            step_counts, q_values = self._sarsa()
            step_counts_per_episode_per_exp.append(step_counts)
            q_values_per_episode_per_exp.append(q_values)
            t_ = time()
            print('{} experiment completed in {} secs'.format(i+1, round(t_-t, 3)))
            t = t_
        print('Total time taken {} mins'.format(round((time()-start)/60, 3)))
        
        self.plot_time_to_goal_vs_episodes(step_counts_per_episode_per_exp)
        self.plot_maximum_q_value_vs_episodes(q_values_per_episode_per_exp)

    def plot_time_to_goal_vs_episodes(self, step_counts_per_episode_per_exp):
        steps = np.array(step_counts_per_episode_per_exp).T  # each ith row is the num of steps in ith episode for all experiments
        averages = np.average(steps, axis=1)
        plt.plot(range(self.n_episodes), averages, label='Average time steps required to reach goal')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Average time steps')
        title = r'Alg:{}; Exp:{}; Eps:{}; Size:{}; $ \lambda $:{}; $ \gamma $:{}; $ \epsilon $:{}; $ \alpha $:{}'.format(
            self.alg, self.n_experiments, self.n_episodes, self.grid_size, self.sarsa_param, self.gamma, self.epsilon, self.alpha
        )
        plt.title(title)
        plt.savefig('Average time steps required to reach goal')
        plt.clf()

    def plot_maximum_q_value_vs_episodes(self, q_values_per_episode_per_exp):
        # we are only looking at the start state - so only look at q-values for that state
        start_state = self._get_initial_agent_state()
        max_q_values = [[max(qval[start_state].values()) for qval in exp_res] for exp_res in q_values_per_episode_per_exp]
        q_values = np.array(max_q_values).T  # each ith row is the max q value in the ith episode for all experiments
        vals = np.average(q_values, axis=1)
        plt.plot(range(self.n_episodes), vals, label='Max q-value for the start state')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Max q-value for start state')
        title = r'Alg:{}; Exp:{}; Eps:{}; Size:{}; $ \lambda $:{}; $ \gamma $:{}; $ \epsilon $:{}; $ \alpha $:{}'.format(
            self.alg, self.n_experiments, self.n_episodes, self.grid_size, self.sarsa_param, self.gamma, self.epsilon, self.alpha
        )
        plt.title(title)
        plt.savefig('Max q-value for the start state')
        plt.clf()
        

    def print_policy(self):
        _actions = ['U', 'R', 'D', 'L']
        for i in range(self.grid_size):
            line = []
            for j in range(self.grid_size):
                state = (i*self.grid_size) + j
                line.append(_actions[max(self.Q[state], key=self.Q[state].get)])
            print(line)


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
    print('-'*15, 'Optimal policy')
    obj.print_policy()
