import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from common.replay_buffer import PrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter

env_id = "LunarLander-v2"
env = gym.make(env_id)
writer = SummaryWriter()

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(autograd.Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(autograd.Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NoisyDQN, self).__init__()
        
        self.linear =  nn.Linear(env.observation_space.shape[0], 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, env.action_space.n)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    
    def act(self, state):
        state   = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].item()
        return action
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

current_model = NoisyDQN(env.observation_space.shape[0], env.action_space.n)
target_model  = NoisyDQN(env.observation_space.shape[0], env.action_space.n)
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

beta_start = 0.4
beta_iterations = 50000 
beta_by_iteration = lambda iteration: min(1.0, beta_start + iteration * (1.0 - beta_start) / beta_iterations)

replay_buffer = PrioritizedReplayBuffer(25000, alpha=0.6)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)

def compute_td_loss(batch_size, beta):
    state, action, reward, next_state, done, weights, indices = replay_buffer.sample(batch_size, beta) 

    state      = autograd.Variable(torch.FloatTensor(np.float32(state)))
    next_state = autograd.Variable(torch.FloatTensor(np.float32(next_state)))
    action     = autograd.Variable(torch.LongTensor(action))
    reward     = autograd.Variable(torch.FloatTensor(reward))
    done       = autograd.Variable(torch.FloatTensor(np.float32(done)))
    weights    = autograd.Variable(torch.FloatTensor(weights))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

def plot(iteration, rewards, losses, ep_reward): 
    print("Outputing Iteration " + str(iteration))
    writer.add_scalar('Train/Rewards', rewards[-1], iteration)
    writer.add_scalar('Train/Losses', losses[-1], iteration)     
    writer.add_scalar('Train/Episode', ep_reward, iteration)
    writer.add_scalar('Train/Beta', beta_by_iteration(iteration), iteration)
    writer.flush()

iterations = 1000000
batch_size = 64
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for iteration in range(1, iterations + 1):
    action = current_model.act(state)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        beta = beta_by_iteration(iteration)
        loss = compute_td_loss(batch_size, beta)
        losses.append(loss.item())
        
    if iteration % 200 == 0 and len(all_rewards)>0 and len(losses)>0:
        plot(iteration, all_rewards, losses, episode_reward)
        
    if iteration % 1000 == 0:
        update_target(current_model, target_model)