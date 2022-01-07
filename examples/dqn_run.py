from stable_baselines3 import DQN
import pommerman
from pommerman import agents

agent_list = [
    agents.RandomAgent(),
    agents.RandomAgent(),
    agents.RandomAgent(),
    agents.RandomAgent(),
]

env = pommerman.make('PommeFFACompetition-v03', agent_list)

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
