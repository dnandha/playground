'''An example to show how to set up an pommerman game programmatically'''
import pommerman
from pommerman import agents

import numpy as np
def matrix_obs(obs):
    mobs = np.zeros((obs['board'].shape[0], obs['board'].shape[1], 24))

    # each type of board item gets own channel
    for i in range(0, 14):
        mobs[:, :, i] = np.array(obs['board'] == i, dtype=np.float)

    mobs[:, :, 14] = obs['bomb_blast_strength'] / 10.  # max bs = 10
    mobs[:, :, 15] = obs['bomb_life'] / 10.  # default bl = 9
    mobs[:, :, 16] = obs['flame_life'] / 2.  # default = 2

    # only relevant for kicks, but anyways ... four different directions
    for i in range(1, 5):
        mobs[:, :, 17+i] = np.array(obs['bomb_moving_direction'] == i,
                                    dtype=np.float)


    # ammo, blast strength, kick at agent pos as extra channels
    pos = obs['position']
    mobs[pos[1], pos[0], 21] = np.float(obs['can_kick'])
    mobs[pos[1], pos[0], 22] = obs['ammo']/10.  # max = 10
    mobs[pos[1], pos[0], 23] = obs['blast_strength']/10.

    # center board around agent position
    board_center = np.array(obs['board'].shape) // 2
    pos_shift = board_center - pos
    mobs = np.roll(mobs, pos_shift[0], axis=1)
    mobs = np.roll(mobs, pos_shift[1], axis=0)

    # TODO: teammates and enemies for radio

    return mobs


def main():
    '''Simple function to bootstrap a game.
       
       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        #agents.RandomAgent(),
        agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        import pdb; pdb.set_trace()
        matrix_obs(state[0])
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
