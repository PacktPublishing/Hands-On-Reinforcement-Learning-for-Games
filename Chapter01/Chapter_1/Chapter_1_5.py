import random

arms = 7
bandits = 7
learning_rate = .1
gamma = .9
episodes = 10000
reward = []
for i in range(bandits):  
    reward.append([])       
    for j in range(arms): 
        reward[i].append(random.uniform(-1,1))
print(reward)

Q = []
for i in range(bandits):  
    Q.append([])       
    for j in range(arms): 
        Q[i].append(10.0)
print(Q)

def greedy(values):
    return values.index(max(values))

def learn(state, action, reward, next_state):
    #q = gamma * max(Q[next_state])
    q = 0
    q += reward
    q -= Q[state][action]
    q *= learning_rate
    q += Q[state][action]
    Q[state][action] = q

# agent learns
bandit = random.randint(0,bandits-1)
for i in range(0, episodes):
    last_bandit = bandit
    bandit = random.randint(0,bandits-1)
    action = greedy(Q[bandit]) 
    r = reward[last_bandit][action]
    learn(last_bandit, action, r, bandit)

print(Q)
