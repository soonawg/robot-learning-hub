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
ENV_NAME = "HalfCheetah-v4"
MAX_EPISODES = 5000
MAX_STEPS = 1000
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1_000_000

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4

GAMMA = 0.99
TAU = 0.005
TARGET_ENTROPY = None  # default: -action_dim

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Network Structures
# ========================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        # TODO: Mean, Log Std 출력하는 네트워크 구성
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # TODO: mean, log_std 계산
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        # TODO: Reparameterization Trick + Squashed Gaussian Policy
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

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
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

# ========================
# SAC Agent (미완성 템플릿)
# ========================

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        # TODO: Actor, Critic, Target Critic, alpha 자동 조정 등 구성
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 자동 엔트로피 조정
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.target_entropy = -action_dim
        self.max_action = max_action
        self.discount = GAMMA
        self.tau = TAU

    def select_action(self, state, evaluate=False):
        # TODO: 평가 모드에서는 noise 제거
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grade():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean) * self.max_action
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy().flatten()


    def train(self, replay_buffer, batch_size):
        # TODO: SAC 학습 로직 구현
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ---- 1. Critic Update ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - torch.exp(self.log_alpha) * next_log_probs
            target_Q = rewards + (1 - dones) * self.discount * target_Q
        
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backwarD()
        self.critic_optimizer.step()

        # ---- 2. Actor Update ----
        new_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (torch.exp(self.log_alpha) * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- 3. Actor (Entropy) Update ----
        alpha_loss = -self.log_alpha * (log_probs + self.target_entropy).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- 4. Target Critic Update ----
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )



# ========================
# Training Loop (템플릿)
# ========================

agent = SACAgent(state_dim, action_dim, max_action)
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


# 결과:
# 에피소드 2995 | 보상: 298.56 | 스텝: 792
# 에피소드 2996 | 보상: 303.63 | 스텝: 760
# 에피소드 2997 | 보상: 301.11 | 스텝: 767
# 에피소드 2998 | 보상: 301.62 | 스텝: 765
# 에피소드 2999 | 보상: 300.43 | 스텝: 772