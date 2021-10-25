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

    def __init__(self, num_agents=4, *args, **kwargs):
        self.num_agents = num_agents
        super().__init__(*args, **kwargs)

    def _set_action_space(self):
        self.action_space = spaces.MultiDiscrete([6] * self.num_agents)

    def _set_observation_space(self):
        """The Observation Space for ALL agents. """
        # .. same as in v0
        bss = self._board_size**2
        min_obs = [0] * 3 * bss + [0] * 5 + [constants.Item.AgentDummy.value] * 4
        max_obs = [len(constants.Item)] * bss + [self._board_size] * bss + [25] * bss
        max_obs += [self._board_size] * 2 + [self._num_items] * 2 + [1]
        max_obs += [constants.Item.Agent3.value] * 4
        # ... same as in v0

        # repeat space for ALL agents
        min_obs = np.repeat(np.array(min_obs).reshape(1, -1), self.num_agents, axis=0)
        max_obs = np.repeat(np.array(max_obs).reshape(1, -1), self.num_agents, axis=0)

        self.observation_space = spaces.Box(
            np.array(min_obs), np.array(max_obs))

    def get_observations(self):
        """Observations for ALL agents."""
        self.observations = super().get_observations()
        return np.vstack([Pomme.featurize(obs) for obs in self.observations])

    def step(self, actions):
        """Give one cumulative reward for this step."""
        obs, rewards, done, info = super().step(actions)
        reward = np.sum(rewards)
        return obs, reward, done, info
