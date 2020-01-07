import random

reward = [1.0, 0.5, 0.2, 0.5, 0.6, 0.1, -.5]
arms = len(reward)
learning_rate = .1
episodes = 10000
Value = [5.0] * arms
print(Value)

def greedy(values):
    return values.index(max(values))

# agent learns
for i in range(0, episodes):
    action = greedy(Value)
    Value[action] = Value[action] + learning_rate * (
        reward[action] - Value[action])

print(Value)