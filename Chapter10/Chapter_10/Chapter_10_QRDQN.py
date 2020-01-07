import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

#from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

env_id = "LunarLander-v2"
env = gym.make(env_id)
writer = SummaryWriter()

class QRDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_quants):
        super(QRDQN, self).__init__()
        
        self.num_inputs  = num_inputs
        self.num_actions = num_actions
        self.num_quants  = num_quants
        
        self.features = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions * self.num_quants)
        )
        
        #self.noisy_value1 = NoisyLinear(64, 128, use_cuda=USE_CUDA)
        #self.noisy_value2 = NoisyLinear(128, self.num_actions * self.num_quants, use_cuda=USE_CUDA)
        
    def forward(self, x):
        batch_size = x.size(0)

        x = self.features(x)
        
        #x = self.noisy_value1(x)
        #x = F.relu(x)
        #x = self.noisy_value2(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        
        return x
    
    def q_values(self, x):
        x = self.forward(x)
        return x.mean(2)
    
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise() 
        
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = autograd.Variable(torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0), volatile=True)
            qvalues = self.forward(state).mean(2)
            action  = qvalues.max(1)[1]
            action  = action.data.cpu().numpy()[0]
        else:
            action = random.randrange(self.num_actions)
        return action

def projection_distribution(dist, next_state, reward, done):
    next_dist = target_model(next_state)
    next_action = next_dist.mean(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    next_dist = next_dist.gather(1, next_action).squeeze(1).cpu().data

    expected_quant = reward.unsqueeze(1) + 0.99 * next_dist * (1 - done.unsqueeze(1))
    expected_quant = autograd.Variable(expected_quant)

    quant_idx = torch.sort(dist, 1, descending=False)[1]

    tau_hat = torch.linspace(0.0, 1.0 - 1./num_quant, num_quant) + 0.5 / num_quant
    tau_hat = tau_hat.unsqueeze(0).repeat(batch_size, 1)
    quant_idx = quant_idx.cpu().data
    batch_idx = np.arange(batch_size)
    tau = tau_hat[:, quant_idx][batch_idx, batch_idx]
        
    return tau, expected_quant

num_quant = 51
Vmin = -10
Vmax = 10

current_model = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)
target_model  = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)
    
optimizer = optim.Adam(current_model.parameters())

replay_buffer = ReplayBuffer(10000)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 

    state      = autograd.Variable(torch.FloatTensor(np.float32(state)))
    next_state = autograd.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = autograd.Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))

    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
    dist = dist.gather(1, action).squeeze(1)
    
    tau, expected_quant = projection_distribution(dist, next_state, reward, done)
    k = 1
    
    huber_loss = 0.5 * tau.abs().clamp(min=0.0, max=k).pow(2)
    huber_loss += k * (tau.abs() -  tau.abs().clamp(min=0.0, max=k))
    quantile_loss = (tau - (tau < 0).float()).abs() * huber_loss
    loss = torch.tensor(quantile_loss.sum() / num_quant, requires_grad=True)
        
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(current_model.parameters(), 0.5)
    optimizer.step()
    
    return loss


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 50000

epsilon_by_frame = lambda iteration: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * iteration / epsilon_decay)

def plot(iteration, rewards, losses, ep_reward):    
    print("Outputing Iteration " + str(iteration))
    writer.add_scalar('Train/Rewards', rewards[-1], iteration)
    writer.add_scalar('Train/Losses', losses[-1], iteration) 
    writer.add_scalar('Train/Exploration', epsilon_by_frame(iteration), iteration)
    writer.add_scalar('Train/Episode', ep_reward, iteration)
    writer.flush()

iterations = 1000000
batch_size = 32
gamma      = 0.98

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()
for iteration in range(1, iterations + 1):
    action = current_model.act(state, epsilon_by_frame(iteration))
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if iteration % 200 == 0:
        plot(iteration, all_rewards, losses, episode_reward)
        
    if iteration % 1000 == 0:
        update_target(current_model, target_model)