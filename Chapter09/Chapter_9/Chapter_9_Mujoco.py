import gym # openAi gym
from gym import envs

env = gym.make('FetchReach-v1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()