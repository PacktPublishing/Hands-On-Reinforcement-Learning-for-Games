from os import system, name
from time import sleep
import numpy as np
import gym
import random
from tqdm import tqdm

env = gym.make("FrozenLake-v0")
env.render()

action_size = env.action_space.n
print("Action size ", action_size)
state_size = env.observation_space.n
print("State size ", state_size)

qtable = np.ones((state_size, action_size))/action_size
print(qtable)

total_episodes = 50000 
total_test_episodes = 100 
play_game_test_episode = 5000
max_steps = 99 
learning_rate = 0.7 
gamma = 0.618 

epsilon = 1.0 
max_epsilon = 1.0 
min_epsilon = 0.1 
decay_rate = 0.01 

rewards = []

def clear():
    if name == 'nt': 
        _ = system('cls')    
    else: 
        _ = system('clear') 

def play_game(render_game):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0    
    for step in range(max_steps):  
        if render_game:
            env.render()
            print("****************************************************")
            print("EPISODE ", episode)
            sleep(.5)
            clear()
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            if render_game:
                print ("Score", total_rewards)
            break
        state = new_state
    return done, state, step, total_rewards

for episode in tqdm(range(total_episodes)):
    state = env.reset()
    step = 0
    done = False
    if episode % play_game_test_episode == 0:
        play_game(True)
    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
        np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state
        if done == True:
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

env.reset()
print(qtable)

for episode in range(total_test_episodes):
    done, state, step, total_rewards = play_game(False)

env.close()
print ("Score over time: " + str(sum(rewards)/total_test_episodes))