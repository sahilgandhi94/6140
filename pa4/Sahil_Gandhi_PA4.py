import argparse
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
        
        self._grid = self._initialize_grid()

    def run(self):
        """ Runs the experiments based on the set params """
        print('-'*15)
        print(vars(self))
        raise NotImplementedError

    
    def _initialize_grid(self):
        """ Initializes the grid based on the `grid_size` """
        return gw.GridworldEnv(shape=[self.grid_size, self.grid_size])

    def get_action(self, action):
        """ The possible action has a 0.8 probability of succeeding """
        assert action in [gw.UP, gw.DOWN, gw.LEFT, gw.RIGHT], \
        'Unsupported action `{}`'.format(action)
        return self._grid.get_action(action)

    def move(self, state, action):
        """ Given the current position, this function outputs the position after the action """
        assert action in [gw.UP, gw.DOWN, gw.LEFT, gw.RIGHT], \
        'Unsupported action `{}`'.format(action)
        return self._grid.move(state, action)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning - Sahil Gandhi.')
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

    RL(
        alg = cmd_args.alg,
        size = cmd_args.size,
        gamma = cmd_args.gamma,
        n_experiments = cmd_args.exps,
        n_episodes = cmd_args.eps,
        epsilon = cmd_args.epsilon,
        alpha = cmd_args.alpha,
        sarsa_param = vars(cmd_args)['lambda']
    ).run()