"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters

from gym import spaces
from tensorforce.agents import PPOAgent

class TensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo'):
        super(TensorForceAgent, self).__init__(character)
        self.algorithm = algorithm

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return None

    def initialize(self, env, logpath):
        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': 'int',
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
                #n = 0
                #for space in env.action_space.spaces:
                #    n += space.n

                #actions = dict(type='int', num_actions=n)
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            return PPOAgent(
                #states=dict(type='float', shape=env.observation_space.shape),
                states=dict(type='float', shape=(env.env._board_size, env.env._board_size, 24)),
                actions=actions,
                network=[
                    dict(type='conv2d', size=24),
                    dict(type='conv2d', size=12),
                    dict(type='conv2d', size=3),
                    dict(type='conv2d', size=1),
                    dict(type='conv2d', size=1),
                    dict(type='flatten'),
                    dict(type='dense', size=128),
                    dict(type='dense', size=64),
                    dict(type='dense', size=64),
                    dict(type='dense', size=32),
                    dict(type='dense', size=env.action_space.n),  # TODO
                    #dict(type='conv2d', size=32),
                    #dict(type='conv2d', size=64),
                    #dict(type='conv2d', size=32),
                    #dict(type='flatten'),
                    #dict(type='dense', size=64),
                    #dict(type='dense', size=32),
                ],
                batching_capacity=1000,
                step_optimizer=dict(type='adam', learning_rate=1e-4),
                summarizer=dict(directory=logpath,
                            labels=['configuration',
                                'gradients_scalar',
                                'regularization',
                                'inputs',
                                'losses',
                                'variables']
                        ), )
        return None
