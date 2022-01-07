"""
The Pommerman environment returns observations and expects actions for multiple agents,
but the action and observation space is set to one agent only
"""

from gym import spaces
import numpy as np

from .. import constants
from .. import utility
from . import v0


class Pomme(v0.Pomme):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'rgb_pixel'],
        'video.frames_per_second': constants.RENDER_FPS
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observations(self):
        """Observations for first agent."""
        obs = super().get_observations()
        return Pomme.featurize(obs[0])

    def step(self, actions):
        """Give one reward for first agent."""
        obs, rewards, done, info = super().step(actions)
        reward = rewards[0]
        return obs, reward, done, info
