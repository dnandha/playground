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
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            bs = 11  # TODO: get boardsize from env
            return PPOAgent(
                states=dict(type='float', shape=(bs, bs, 24)),
                actions=actions,
                network=[
                    #dict(type='linear_normalization'),
                    dict(type='conv2d', size=32),
                    dict(type='conv2d', size=64),
                    dict(type='conv2d', size=12),
                    dict(type='flatten'),
                    dict(type='dense', size=64),
                    dict(type='dense', size=32),
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
