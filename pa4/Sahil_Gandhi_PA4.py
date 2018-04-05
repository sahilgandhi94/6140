import argparse
from copy import deepcopy
from multiprocessing import Pool
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

    def _q_value_computation_q_learning(self, curr_state, next_state, reward, action, is_done=False):
        """ Compute running average Q(s,a) using Q-learning algorithm"""
        if not is_done:
            self.Q[curr_state][action] = (((1 - self.alpha) * self.Q[curr_state][action]) + 
                    (self.alpha * (reward + (self.gamma * max(self.Q[next_state].values())))))
        else:
            self.Q[curr_state][action] = (((1 - self.alpha) * self.Q[curr_state][action]) + 
                    (self.alpha * (reward)))

    def _q_value_computation_sarsa(self, curr_state, next_state, reward, 
                                   curr_action, next_action, is_done=False):
        """ Compute running average Q(s,a) and E(s,a) using Sarsa(lambda) algorithm"""
        if not is_done:
            delta = reward + self.gamma * self.Q[next_state][next_action] - self.Q[curr_state][curr_action]
        else: 
            delta = reward - self.Q[curr_state][curr_action]
        
        self.E[curr_state][curr_action] += 1
        for s in range(self.grid_size**2):
            for a in range(self.n_actions):
                self.Q[s][a] += (self.alpha * delta * self.E[s][a])
                self.E[s][a] *= (self.gamma * self.sarsa_param)

    def _is_exit_from_terminal_state(self, curr_state, next_state, curr_is_done, next_is_done):
        """ Returns if the action is an exit-action from terminal state """
        return next_is_done and curr_is_done# and (next_state == curr_state)

    def _q_learning(self, exp_no=None):
        """ Executes 1 experiment of q-learning algorithm """
        step_counts = []
        q_values_per_episode = []
        t = time()
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
                self._q_value_computation_q_learning(curr_state,next_state, reward, action,
                                                    is_done=(curr_is_done and next_is_done))
                # print(step_count, curr_state, action, reward, curr_is_done)
                if self._is_exit_from_terminal_state(curr_state, next_state, curr_is_done, next_is_done):
                    step_counts.append(step_count)
                    q_values_per_episode.append(max(self.Q[self._get_initial_agent_state()].values()))
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done
        print('{} experiment completed in {} secs'.format(exp_no, round(time()-t, 3)))
        return step_counts, q_values_per_episode

    def _sarsa(self, exp_no=None):
        """ Executes 1 experiment of sarsa(lambda) algorithm """
        step_counts = []
        q_values_per_episode = []
        t = time()
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
                next_state, reward, next_is_done = self._take_action(curr_state, action)
                chosen_next_action = self._choose_action_from_policy(next_state)
                next_action = self._evaluate_action_with_env_probabilities(chosen_next_action)
                self._q_value_computation_sarsa(curr_state, next_state, reward, action, next_action,
                                                is_done=(curr_is_done and next_is_done))
                if self._is_exit_from_terminal_state(curr_state, next_state, curr_is_done, next_is_done):
                    step_counts.append(step_count)
                    q_values_per_episode.append(max(self.Q[self._get_initial_agent_state()].values()))
                    break
                else:
                    curr_state = next_state
                    curr_is_done = next_is_done
                    action = self._evaluate_action_with_env_probabilities(next_action)
        print('{} experiment completed in {} secs'.format(exp_no, round(time()-t, 3)))
        return step_counts, q_values_per_episode

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
        q_values = np.array(q_values_per_episode_per_exp).T  # each ith row is the max q value in the ith episode for all experiments
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

    def plot(self, values, variable, var_values, y_label, title):
        values = [np.average(np.array(val).T, axis=1) for val in values]
        for i, val in enumerate(values):
            # print('plotting value for {}-{}-{} var, values {}'.format(variable, i, var_values[i], val[:10]))
            plt.plot(range(self.n_episodes), val, label='{}: {}'.format(variable, var_values[i]))
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel(y_label)
        # $ \lambda $:{}; $ \gamma $:{}; $ \epsilon $:{}; $ \alpha $:{}
        title += ' Alg:{}; Exp:{}; Eps:{}; Size:{} '.format(
            self.alg, self.n_experiments, self.n_episodes, self.grid_size #, self.sarsa_param, self.gamma, self.epsilon, self.alpha
        )
        plt.title(title)
        plt.savefig('images/'+variable+'-'+title)
        plt.clf()

    def print_policy(self):
        _actions = ['U', 'R', 'D', 'L']
        for i in range(self.grid_size):
            line = []
            for j in range(self.grid_size):
                state = (i*self.grid_size) + j
                line.append(_actions[max(self.Q[state], key=self.Q[state].get)])
            print(line)

    @staticmethod
    def new_obj(cmd_args):
        return RL(
            alg = cmd_args.alg,
            size = cmd_args.size,
            gamma = cmd_args.gamma,
            n_experiments = cmd_args.exps,
            n_episodes = cmd_args.eps,
            epsilon = cmd_args.epsilon,
            alpha = cmd_args.alpha,
            sarsa_param = vars(cmd_args)['lambda']
        )

    @staticmethod
    def run_experiments(*args):
        cmd_args, exp_num = args[0]
        obj = RL.new_obj(cmd_args)
        if cmd_args.alg == 'q':
            return obj._q_learning(exp_num)
        else:
            return obj._sarsa(exp_num)

def eval_n_experiments(cmd_args, variable, var_values):
    """ Runs algorithm n times and plots per-experiment parameters """
    t = time()
    pool = Pool()
    step_count_per_value = []
    q_value_per_value = []
    for val in var_values:
        step_counts_per_episode_per_exp = []
        q_values_per_episode_per_exp = []
        values = pool.map(RL.run_experiments, [(cmd_args, i+1) for i in range(int(cmd_args.exps))])
        for val in values:
            step_counts_per_episode_per_exp.append(val[0])
            q_values_per_episode_per_exp.append(val[1])
        step_count_per_value.append(step_counts_per_episode_per_exp)
        q_value_per_value.append(q_values_per_episode_per_exp)
    print('Total time taken to run n-experiments for `{}` variable is {} mins'.format(variable, round((time()-t)/60, 3)))

    obj = RL.new_obj(cmd_args)
    obj.plot(step_count_per_value, variable, var_values, y_label='Average time steps', title='Average time steps')
    obj.plot(q_value_per_value, variable, var_values, y_label='Max q-value for start state', title='Max q-value for start state')

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

    alphas = [0, 0.01, .05, .1, .5, 1]
    epsilons = [0, .1, .25, .5, 1]
    lambdas = [0, .25, .5, .75, 1]
    
    eval_n_experiments(cmd_args, 'alpha', alphas)
    eval_n_experiments(cmd_args, 'epsilon', epsilons)
    if cmd_args.alg == 's':
        eval_n_experiments(cmd_args, 'lambda', lambdas)
