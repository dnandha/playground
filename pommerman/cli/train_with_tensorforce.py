"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym

from pommerman import helpers, make
from pommerman.agents import TensorForceAgent

from tensorboardX import SummaryWriter

from pommerman.agents import action_prune
from pommerman.constants import Action

CLIENT = docker.from_env()


def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def reward_proc(self, action, obs, next_obs, reward):
        reward = reward

        # check if proposed action is valid
        valid_actions = action_prune.get_filtered_actions(obs,
                                                          prev_two_obs=[None, None])  # TODO: try out

        # if not, give negative reward
        if valid_actions and action not in valid_actions:
            reward = -1.0

        # TODO: check next_obs for powerup and give small reward

        return reward

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        # get current state
        obs = self.gym.get_observations()

        # get other player actions
        all_actions = self.gym.act(obs)
        # insert own action
        actions = self.unflatten_action(action=action)
        all_actions.insert(self.gym.training_agent, actions)

        # step the env
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])

        # get next state
        next_obs = self.gym.get_observations()

        # shape reward
        agent_reward = self.reward_proc(action,
                                        obs[self.gym.training_agent],
                                        next_obs[self.gym.training_agent],
                                        reward[self.gym.training_agent])

        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[self.gym.training_agent])
        return agent_obs




def main():
    #writer = SummaryWriter('runs/exp-1')

    #def eps_finished(r):
    #    writer.add_scalar('reward', r.episode_rewards[-1], r.episode)
    #    writer.add_scalar('timesteps', r.timestep, r.episode)
    #    return True

    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    parser.add_argument(
        "--logpath",
        default="./runs",
        help="Log path")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Load checkpoint")
    args = parser.parse_args()

    config = args.config
    #record_pngs_dir = args.record_pngs_dir
    #record_json_dir = args.record_json_dir
    #agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env, args.logpath)

    if args.checkpoint:
        print("Restoring model")
        agent.restore_model(directory="./", file=args.checkpoint)

    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=10000, max_episode_timesteps=2000, testing=args.checkpoint)#, episode_finished=eps_finished)
    if not args.checkpoint:
        print("Saving model")
        agent.save_model("./checkpoints")

    print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
          runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
