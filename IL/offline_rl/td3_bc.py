import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from copy import deepcopy

# 하이퍼파라미터
ENV_NAME = "Pendulum-v1"  # 예시 환경
SEED = 42
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
POLICY_FREQ = 2
ALPHA = 2.5  # BC loss와 Q loss의 trade-off
MAX_ACTION = 2.0  # Pendulum의 경우
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 네트워크 정의
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = torch.relu(self.l1(xu))
        x1 = torch.relu(self.l2(x1))
        return self.l3(x1)

# 리플레이 버퍼 (오프라인 데이터셋)
class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = deque(maxlen=max_size)

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(DEVICE),
            torch.FloatTensor(action).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor(reward).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(done).unsqueeze(1).to(DEVICE),
        )

# 오프라인 데이터셋 생성 함수 (expert + random)
def generate_offline_data(env, policy, num_transitions, random_ratio=0.0):
    buffer = ReplayBuffer()
    num_random = int(num_transitions * random_ratio)
    num_expert = num_transitions - num_random

    # Random policy 데이터
    state, _ = env.reset(seed=SEED)
    for _ in range(num_random):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add((state, action, next_state, reward, float(done)))
        state = next_state if not done else env.reset()[0]

    # Expert policy 데이터
    state, _ = env.reset(seed=SEED+1)
    for _ in range(num_expert):
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add((state, action, next_state, reward, float(done)))
        state = next_state if not done else env.reset()[0]

    return buffer

# 간단한 expert policy (Pendulum 기준)
def expert_policy(state):
    # 실제 실험에서는 사전 학습된 policy 사용 권장
    return np.clip(np.array([2.0 * state[2]]), -MAX_ACTION, MAX_ACTION)

# TD3+BC 에이전트
class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic1_target = deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=CRITIC_LR)

        self.critic2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=CRITIC_LR)

        self.max_action = max_action
        self.total_it = 0

    def train(self, replay_buffer, iterations):
        for it in range(iterations):
            self.total_it += 1
            state, action, next_state, reward, done = replay_buffer.sample(BATCH_SIZE)

            # Critic update
            with torch.no_grad():
                noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
                target_q1 = self.critic1_target(next_state, next_action)
                target_q2 = self.critic2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * GAMMA * target_q

            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)
            critic1_loss = nn.MSELoss()(current_q1, target_q)
            critic2_loss = nn.MSELoss()(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # Actor update (policy_freq마다)
            if it % POLICY_FREQ == 0:
                pi = self.actor(state)
                q = self.critic1(state, pi)
                lmbda = ALPHA / q.abs().mean().detach()
                # TD3+BC Loss
                actor_loss = -lmbda * q.mean() + nn.MSELoss()(pi, action)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Target network soft update
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

# 평가 함수
def evaluate_policy(env, policy, episodes=5):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return np.mean(scores)

# 메인 실행
if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.action_space.seed(SEED)

    # 오프라인 데이터셋 생성 (예: expert 80%, random 20%)
    buffer = generate_offline_data(env, expert_policy, num_transitions=100_000, random_ratio=0.2)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = TD3_BC(state_dim, action_dim, MAX_ACTION)

    # 학습
    for epoch in range(10):  # 에폭 수는 실험에 맞게 조정
        agent.train(buffer, iterations=1000)
        score = evaluate_policy(env, agent.select_action)
        print(f"Epoch {epoch+1}, Eval Score: {score:.2f}")

    print("학습 완료!")
