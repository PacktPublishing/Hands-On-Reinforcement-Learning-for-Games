import random

reward = [1.0, 0.5, 0.2, 0.5, 0.6, 0.1, -.5]
arms = len(reward)
learning_rate = .1
episodes = 100
Value = [0.0] * arms
print(Value)

#agent learns
for i in range(0, episodes):
    action = random.randint(0,arms-1)
    Value[action] = Value[action] + learning_rate * (
        reward[action] - Value[action])

print(Value)