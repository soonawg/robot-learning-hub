"""
MaxEnt IRL 실습 (LunarLander-v3)
- Expert trajectory로부터 보상 함수를 추정하고, 그 보상으로 RL agent를 학습
- 성능이 잘 나오도록 feature engineering, 안정적 학습, 충분한 데이터 사용
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

# =========================
# 1. Expert Trajectory 수집
# =========================

def collect_expert_trajectories(env, policy_fn, num_episodes=200, min_reward=100):
    """
    Expert 정책으로 trajecory를 수집한다.
    - policy_fn: state -> action
    - min_reward: 일정 보상 이상만 expert만 간주
    """
    trajectories = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        traj = []
        total_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            action = policy_fn(state)
            next_state, reward, done, truncated, info = env.step(action)
            traj.append((state, action))
            state = next_state
            total_reward += reward
        if total_reward > min_reward:
            trajectories.append(traj)
        if ep % 20 == 0:
            print(f"Expert episode {ep}, reward: {total_reward:.2f}, saved: {total_reward > min_reward}")
    print(f"Collected {len(trajectories)} expert trajectories")
    return trajectories

def simple_expert_policy(state):
    """
    LunarLander-v3용 간단한 expert 정책
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
    if left_contact and right_contact: return 0
    if y < 0.1: return 1
    if abs(angle) > 0.5: return 2 if angle > 0 else 3
    if abs(angular_vel) > 0.5: return 2 if angular_vel > 0 else 3
    if abs(vx) > 0.5: return 2 if vx > 0 else 3
    if vy < -0.5: return 1
    if y > 0.3 and abs(angle) < 0.1 and abs(vx) < 0.1: return 1
    return 0

# =========================
# 2. Feature Engineering
# =========================

def featurize_state(state):
    """
    상태를 feature vector로 변환 (여기선 원본 상태 그대로 사용)
    - 필요시 다항식 확장, 정규화 등 추가 가능
    """
    return np.array(state, dtype=np.float32)

def featurize_trajectory(traj):
    """
    trajectory의 모든 state를 feature로 변환
    """
    return np.array([featurize_state[s] for s, a in traj])

# =========================
# 3. MaxEnt IRL 핵심 함수
# =========================

class RewardNet(nn.Module):
    """
    상태 feature를 입력받아 보상값을 출력하는 신경망
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        r = self.fc3(x)
        return r.squeeze(-1)

def compute_state_visitation_freq(env, policy, gamma, traj_length, num_trajs=100):
    """
    현재 policy로부터 상태 방문 빈도 추정
    - policy: state -> action
    - 반환: (state_dim, ) 평균 방문 feature
    """
    state_dim = env.observation_space.shape[0]
    feat_sum = np.zeros(state_dim)
    for _ in range(num_trajs):
        state, _ = env.reset()
        for t in range(traj_length):
            feat_sum += featurize_state(state) * (gamma ** t)
            action = policy(state)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            if done or truncated:
                break
    return feat_sum / num_trajs

def compute_expert_feature_expectation(expert_trajs, gamma):
    """
    Expert trajectory에서 feature expectation 계산
    """
    state_dim = len(featurize_state(expert_trajs[0][0][0]))
    feat_sum = np.zeros(state_dim)
    for traj in expert_trajs:
        for t, (state, action) in enumerate(traj):
            feat_sum += featurize_state(state) * (gamma ** t)
    return feat_sum / len(expert_trajs)

def sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=1e-2, epochs=100):
    """
    추정된 reward_net을 이용해 policy를 REINFORCE로 학습
    - 간단한 policy network 사용 (softmax policy)
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_dim)
            self.relu = nn.ReLU()
        def forward(self, state):
            x = self.relu(self.fc1(state))
            x = self.relu(self.fc2(x))
            return self.fc3(x)  # logits

    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for epoch in range(epochs):
        states, actions, rewards, log_probs = [], [], [], []
        state, _ = env.reset()
        for t in range(traj_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, _, done, truncated, info = env.step(action.item())
            # reward는 reward_net에서 추정
            reward = reward_net(torch.FloatTensor(state).unsqueeze(0)).item()
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            if done or truncated:
                break
        # REINFORCE: discounted return 계산
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        # Policy gradient loss (maximize expected return)
        loss = -torch.mean(log_probs * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Policy RL epoch {epoch}, loss: {loss.item():.4f}, return: {np.sum(rewards):.2f}")
    # 최종 policy 함수 반환
    def policy_fn(state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = policy_net(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
        return action
    return policy_fn

def maxent_irl(env, expert_trajs, gamma=0.99, lr=1e-2, epochs=200, traj_length=200):
    """
    MaxEnt IRL 메인 함수
    - reward_net을 학습하여 expert와 유사한 feature expectation을 갖도록 함
    """
    state_dim = env.observation_space.shape[0]
    reward_net = RewardNet(state_dim)
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    # 1. Expert feature expectation 계산
    expert_feat_exp = compute_expert_feature_expectation(expert_trajs, gamma)
    print("Expert feature expectation:", expert_feat_exp.round(2))

    # 2. 초기 policy: 랜덤 정책
    def random_policy(state): return env.action_space.sample()

    # 3. IRL 학습 루프
    for epoch in range(epochs):
        # 3-1. 현재 reward_net으로 policy 학습 (inner loop)
        policy_fn = sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=1e-2, epochs=10)
        # 3-2. 현재 policy의 feature expectation 계산
        policy_feat_exp = compute_state_visitation_freq(env, policy_fn, gamma, traj_length, num_trajs=30)
        # 3-3. IRL loss: expert와 policy의 feature expectation 차이 최소화
        loss = torch.sum((torch.FloatTensor(policy_feat_exp) - torch.FloatTensor(expert_feat_exp)) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"IRL epoch {epoch}, loss: {loss.item():.4f}")
    return reward_net

# =========================
# 4. 전체 파이프라인 실행
# =========================

def main():
    # 1. 환경 생성
    env = gym.make('LunarLander-v3')
    gamma = 0.99
    traj_length = 200

    # 2. Expert trajectory 수집
    print("Collecting expert trajectories...")
    expert_trajs = collect_expert_trajectories(env, simple_expert_policy, num_episodes=300, min_reward=100)

    # 3. MaxEnt IRL로 보상 함수 학습
    print("Running MaxEnt IRL...")
    reward_net = maxent_irl(env, expert_trajs, gamma=gamma, lr=1e-2, epochs=50, traj_length=traj_length)

    # 4. 학습된 보상 함수로 RL policy 학습
    print("Training RL agent with learned reward...")
    learned_policy = sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=1e-2, epochs=200)

    # 5. 평가
    print("Evaluating learned policy...")
    rewards = []
    for ep in range(20):
        state, _ = env.reset()
        total_reward = 0
        done, truncated = False, False
        while not (done or truncated):
            action = learned_policy(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
        print(f"Test episode {ep}, reward: {total_reward:.2f}")
    print(f"Learned policy average reward: {np.mean(rewards):.2f}")
    
    # 6. 결과 시각화
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o')
    plt.title("Learned Policy Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.axhline(y=200, color='r', linestyle='--', label='Success threshold')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()