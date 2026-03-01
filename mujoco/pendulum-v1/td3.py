# TD3의 주요 특징
# DDPG의 문제점을 해결하기 위해 만들어진 알고리즘이기에 DDPG의 기본 틀은 그대로 가져옴
# Q-Value 2개 -> min사용
# Target policy smoothing
# Actor delayed update
# 위 3가지가 TD3만의 주요 특징들

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import copy

# 시드 설정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 하이퍼파라미터
BUFFER_SIZE = 100000     # 리플레이 버퍼 크기
BATCH_SIZE = 64          # 미니배치 크기
GAMMA = 0.99             # 할인율
TAU = 0.005              # 타겟 네트워크 소프트 업데이트 비율
LR_ACTOR = 0.0001        # Actor 학습률
LR_CRITIC = 0.001        # Critic 학습률
POLICY_NOISE = 0.2       # 타겟 정책 스무딩에 사용되는 노이즈 크기
NOISE_CLIP = 0.5         # 노이즈 클리핑 범위
POLICY_DELAY = 2         # 정책 업데이트 지연 스텝

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 네트워크
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

# Critic 네트워크 (Q1, Q2)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 아키텍처
        self.layer1_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2_1 = nn.Linear(400, 300)
        self.layer3_1 = nn.Linear(300, 1)
        
        # Q2 아키텍처
        self.layer1_2 = nn.Linear(state_dim + action_dim, 400)
        self.layer2_2 = nn.Linear(400, 300)
        self.layer3_2 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.layer1_1(sa))
        q1 = F.relu(self.layer2_1(q1))
        q1 = self.layer3_1(q1)
        
        # Q2
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

# 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
    
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

# TD3 에이전트
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.max_action = max_action
        self.total_it = 0
    
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # 탐색을 위한 노이즈 추가
        if noise != 0:
            action = action + np.random.normal(0, noise * self.max_action, size=action.shape)
            
        return np.clip(action, -self.max_action, self.max_action)
    
    def train(self, replay_buffer, batch_size=BATCH_SIZE):
        self.total_it += 1
        
        # 리플레이 버퍼에서 배치 샘플링
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 타겟 액션 계산 (타겟 정책 스무딩 포함)
        with torch.no_grad():
            # 타겟 정책 스무딩: 노이즈 추가 후 클리핑
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # 타겟 Q 값 계산 (두 Q 네트워크 중 작은 값 사용)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * GAMMA * target_Q
        
        # 현재 Q 값 계산
        current_Q1, current_Q2 = self.critic(states, actions)
        
        # Critic 손실 계산 및 업데이트
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 지연된 정책 업데이트
        if self.total_it % POLICY_DELAY == 0:
            # Actor 손실 계산 및 업데이트 (Q1 값 최대화)
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 타겟 네트워크 소프트 업데이트
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

# 학습 함수
def train_td3(env_name, num_episodes=1000, max_steps=500, render=False, save_interval=100):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(1, num_episodes+1):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        
        for t in range(max_steps):
            # 환경과 상호작용
            if replay_buffer.size() < BATCH_SIZE:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=0.1)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 리플레이 버퍼에 경험 저장
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            state = next_state
            episode_reward += reward
            
            # 에이전트 학습
            if replay_buffer.size() >= BATCH_SIZE:
                agent.train(replay_buffer)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        
        # 모델 저장
        if episode % save_interval == 0:
            torch.save(agent.actor.state_dict(), f"td3_actor_{env_name}_{episode}.pth")
            torch.save(agent.critic.state_dict(), f"td3_critic_{env_name}_{episode}.pth")
    
    # 학습 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='100-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'TD3 on {env_name}')
    plt.legend()
    plt.savefig(f'td3_{env_name}_training.png')
    plt.show()
    
    env.close()
    return agent

# 학습된 에이전트 테스트
def test_agent(agent, env_name, num_episodes=10, render=True):
    env = gym.make(env_name, render_mode='human' if render else None)
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
        
        print(f"Test Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    env_name = "Pendulum-v1"
    
    # TD3 에이전트 학습
    agent = train_td3(env_name, num_episodes=500)
    
    # 학습된 에이전트 테스트
    test_agent(agent, env_name, render=True)
