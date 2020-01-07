import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class REINFORCE(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(REINFORCE, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def act(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + gamma * R
            loss = -log_prob * R
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.data = []

env = gym.make('LunarLander-v2')
pi = REINFORCE(env.observation_space.shape[0], env.action_space.n)
score = 0.0
print_interval = 100
iterations = 10000
min_play_reward = 20

def play_game():
    done = False
    state = env.reset()
    its = 500
    while(not done and its > 0):
        its -= 1
        prob = pi.act(torch.from_numpy(state).float())
        m = Categorical(prob)
        action = m.sample()
        next_state, reward, done, _ = env.step(action.item())
        env.render()
        state = next_state

for iteration in range(iterations):
    s = env.reset()
    for t in range(501): # CartPole-v1 forced to terminates at 500 step.
        prob = pi.act(torch.from_numpy(s).float())
        m = Categorical(prob)
        action = m.sample()
        s_prime, r, done, info = env.step(action.item())
        pi.put_data((r,torch.log(prob[action])))
            
        s = s_prime
        score += r
        if done:
            if score/print_interval > min_play_reward:
                play_game()
            break

    pi.train_net()
        
    if iteration%print_interval==0 and iteration!=0:
        print("# of episode :{}, avg score : {}".format(iteration, score/print_interval))
        score = 0.0
        
    
env.close()