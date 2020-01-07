import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make('MountainCar-v0')
Q_table = np.zeros((65,65,3))
alpha=0.3
buckets=[65, 65]
gamma=0.99
rewards=[]
episodes=2000
lambdaa=0.8

def to_discrete_states(observation):
	interval=[0 for i in range(len(observation))]
	max_range=[1.2,0.07]	

	for i in range(len(observation)):
		data = observation[i]
		inter = int(math.floor((data + max_range[i])/(2*max_range[i]/buckets[i])))
		if inter>=buckets[i]:
			interval[i]=buckets[i]-1
		elif inter<0:
			interval[i]=0
		else:
			interval[i]=inter
	return interval

def expect_epsilon(t):
    return min(0.015, 1.0 - math.log10((t+1)/220.))

def get_action(observation,t):
	if np.random.random()<max(0.001, expect_epsilon(t)):
		return env.action_space.sample()
	interval = to_discrete_states(observation)
	return np.argmax(np.array(Q_table[tuple(interval)]))

def expect_alpha(t):
    return min(0.1, 1.0 - math.log10((t+1)/125.))

def updateQ_SARSA(observation,reward,action,ini_obs,next_action,t,eligibility):
	interval = to_discrete_states(observation)
	Q_next = Q_table[tuple(interval)][next_action]
	ini_interval = to_discrete_states(ini_obs)
	lr=max(0.4, expect_alpha(t))
	td_error=(reward + gamma*(Q_next) - Q_table[tuple(ini_interval)][action])
	Q_table[:,:,action]+=lr*td_error*(eligibility[:,:,action])

for episode in range(episodes):
    observation = env.reset()
    t=0
    eligibility = np.zeros((65,65,3))
    done=False
    while (done==False):
        env.render()
        action = get_action(observation,episode)
        next_obs, reward, done, info = env.step(action)
        interval = to_discrete_states(observation)
        eligibility *= lambdaa * gamma
        eligibility[tuple(interval)][action]+=1
        
        next_action = get_action(next_obs,episode)
        updateQ_SARSA(next_obs,reward,action,observation,next_action,episode,eligibility)
        observation=next_obs
        action = next_action
        t+=1
    rewards.append(t+1)
    
plt.plot(rewards)
plt.show()