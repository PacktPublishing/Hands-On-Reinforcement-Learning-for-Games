import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman

import matplotlib.pyplot as plt

from tqdm import tqdm

#USE_CUDA = torch.cuda.is_available()
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def act(self, x, deterministic=False):
        logit, value = self.forward(x)
        probs = F.softmax(logit)
        
        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(1)
        
        return action
    
    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)
        
        probs     = F.softmax(logit)
        log_probs = F.log_softmax(logit)
        
        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()
        
        return logit, action_log_probs, value, entropy

class ActorCritic(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()
        
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
        )
        
        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

# @ikostrikov style

class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.states  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.rewards = torch.zeros(num_steps,     num_envs, 1)
        self.masks   = torch.ones(num_steps  + 1, num_envs, 1)
        self.actions = torch.zeros(num_steps,     num_envs, 1).long()
        #self.use_cuda = False
            
    def cuda(self):
        #self.use_cuda  = True
        self.states    = self.states.cuda()
        self.rewards   = self.rewards.cuda()
        self.masks     = self.masks.cuda()
        self.actions   = self.actions.cuda()
        
    def insert(self, step, state, action, reward, mask):
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        
    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        
    def compute_returns(self, next_value, gamma):
        returns   = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        #if self.use_cuda:
        #    returns = returns.cuda()
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]

def main():
    mode = "regular"
    num_envs = 16

    def make_env():
        def _thunk():
            env = MiniPacman(mode, 1000)
            return env

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape

    #a2c hyperparams:
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(10e3)

    #rmsprop hyperparams:
    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99

    #Init a2c and rmsprop
    actor_critic = ActorCritic(envs.observation_space.shape, envs.action_space.n)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
    
    #if USE_CUDA:
    #    actor_critic = actor_critic.cuda()

    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    #rollout.cuda()

    all_rewards = []
    all_losses  = []

    state = envs.reset()
    state = torch.FloatTensor(np.float32(state))

    rollout.states[0].copy_(state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)

    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            action = actor_critic.act(autograd.Variable(state))

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())

            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            #if USE_CUDA:
            #    masks = masks.cuda()

            state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, state, action.data, reward, masks)


        _, next_value = actor_critic(autograd.Variable(rollout.states[-1], volatile=True))
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            autograd.Variable(rollout.states[:-1]).view(-1, *state_shape),
            autograd.Variable(rollout.actions).view(-1, 1)
        )

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = autograd.Variable(returns) - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(autograd.Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm(actor_critic.parameters(), max_grad_norm)
        optimizer.step()
    
        if i_update % num_frames == 0:
            all_rewards.append(final_rewards.mean())
            all_losses.append(loss.item())
        
            #clear_output(True)
            plt.figure(figsize=(20,5))
            plt.subplot(131)
            plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
            plt.plot(all_rewards)
            plt.subplot(132)
            plt.title('loss %s' % all_losses[-1])
            plt.plot(all_losses)
            plt.show()
        
        rollout.after_update()

    torch.save(actor_critic.state_dict(), "actor_critic_" + mode)

    import time 

    def displayImage(image, step, reward):
        #clear_output(True)
        s = "step: " + str(step) + " reward: " + str(reward)
        plt.figure(figsize=(10,3))
        plt.title(s)
        plt.imshow(image)
        plt.show()
        time.sleep(0.1)

    env = MiniPacman(mode, 1000)

    done = False
    state = env.reset()
    total_reward = 0
    step   = 1

    while not done:
        current_state = torch.FloatTensor(state).unsqueeze(0)
        #if USE_CUDA:
        #    current_state = current_state.cuda()
        
        action = actor_critic.act(autograd.Variable(current_state))
    
        next_state, reward, done, _ = env.step(action.data[0, 0])
        total_reward += reward
        state = next_state
    
        image = torch.FloatTensor(state).permute(1, 2, 0).cpu().numpy()
        displayImage(image, step, total_reward)
        step += 1



if __name__ == '__main__':
    main()
