from os import system, name
import time
import gym
import numpy as np
env = gym.make('FrozenLake-v0')
env.reset()

def clear():
    if name == 'nt': 
        _ = system('cls')    
    else: 
        _ = system('clear') 

def act(V, env, gamma, policy, state, v):
    for action, action_prob in enumerate(policy[state]):                
        for state_prob, next_state, reward, end in env.P[state][action]:                                        
            v += action_prob * state_prob * (reward + gamma * V[next_state])                    
            V[state] = v
            
def eval_policy(policy, env, gamma=1.0, theta=1e-9, terms=1e9):     
    V = np.zeros(env.nS)  
    delta = 0
    for i in range(int(terms)): 
        for state in range(env.nS):            
            act(V, env, gamma, policy, state, v=0.0)         
        clear()
        print(V)
        time.sleep(1) 
        v = np.sum(V)
        if v - delta < theta:
            return V
        else:
            delta = v
    return V

policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA
V = eval_policy(policy, env.env) 

print(policy, V)

