# TD3 -> BipedalWalker-v3
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

# ========================
# Hyperparameters
# ========================
ENV_NAME = "BipedalWalker-v3"
MAX_EPISODES = 3000
MAX_STEPS = 1000
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1_000_000
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
EXPLORATION_NOISE = 0.1
LEARNING_STARTS = 10000
UPDATE_FREQ = 10

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================
# Network Structures
# ========================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2_1 = nn.Linear(256, 256)
        self.layer3_1 = nn.Linear(256, 1)

        self.layer1_2 = nn.Linear(state_dim + action_dim, 256)
        self.layer2_2 = nn.Linear(256, 256)
        self.layer3_2 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1_1(sa))
        q1 = F.relu(self.layer2_1(q1))
        q1 = self.layer3_1(q1)

        q2 = F.relu(self.layer1_2(sa))
        q2 = F.relu(self.layer2_2(q2))
        q2 = self.layer3_2(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.layer1_1(sa))
        q1 = F.relu(self.layer2_1(q1))
        q1 = self.layer3_1(q1)
        
        return q1

# ========================
# Replay Buffer
# ========================

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)

        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

# ========================
# TD3 Agent
# ========================

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.max_action = max_action
        self.total_it = 0

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.tau = 0.005
        self.discount = 0.99
        self.total_it = 0

    def select_action(self, state, noise=EXPLORATION_NOISE):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action += np.random.normal(0, noise * self.max_action, size=action.shape)
        
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)
        
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ========================
# Training Loop
# ========================

agent = TD3Agent(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

print("경험 수집 중...")
state, _ = env.reset()
for t in range(LEARNING_STARTS):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.add(state, action, reward, next_state, float(done))
    state = next_state if not done else env.reset()[0]
    if (t + 1) % 1000 == 0:
        print(f"경험 수집: {t+1}/{LEARNING_STARTS}")

print("학습 시작...")
for ep in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    step_count = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward
        step_count += 1

        if step % UPDATE_FREQ == 0 and replay_buffer.size() >= BATCH_SIZE:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            break

    print(f"에피소드 {ep} | 보상: {episode_reward:.2f} | 스텝: {step_count}")
    
    if ep % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()