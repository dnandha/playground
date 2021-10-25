from stable_baselines3 import PPO
import pommerman
from pommerman import agents
from pommerman.envs.v0 import Pomme
from pommerman.configs import ffa_v0_fast_env

agent_list = [
    agents.RandomAgent(),
    agents.RandomAgent(),
    agents.RandomAgent(),
    agents.RandomAgent(),
]

#args = ffa_v0_fast_env()['env_kwargs']
#env = Pomme(**args, agent_list)
env = pommerman.make('PommeFFACompetition-v03', agent_list)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
