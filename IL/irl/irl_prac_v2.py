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
# 1. Expert Trajectory 수집 (v1에서 개선된 버전)
# =========================

def collect_expert_trajectories(env, policy_fn, num_episodes=500, min_reward=-300):
    """
    Expert 정책으로 trajecory를 수집한다.
    - min_reward를 낮춰서 더 많은 데이터 수집
    - 상위 성능 에피소드만 선별
    """
    all_trajectories = []
    all_rewards = []
    
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
        
        all_trajectories.append(traj)
        all_rewards.append(total_reward)
        
        if ep % 50 == 0:
            print(f"Expert episode {ep}, reward: {total_reward:.2f}")
    
    # 상위 30% 에피소드만 expert로 선별
    if len(all_rewards) > 10:
        threshold = np.percentile(all_rewards, 70)  # 상위 30%
        expert_trajs = [traj for traj, reward in zip(all_trajectories, all_rewards) if reward >= threshold]
        print(f"Collected {len(expert_trajs)} expert trajectories (top 30%, threshold: {threshold:.2f})")
    else:
        # 데이터가 적으면 모든 것을 사용
        expert_trajs = all_trajectories
        print(f"Collected {len(expert_trajs)} expert trajectories (all episodes)")
    
    return expert_trajs

def improved_expert_policy(state):
    """
    LunarLander-v3용 개선된 expert 정책
    - 더 정교한 제어 로직으로 성능 향상
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state

    # 착륙 성공 시
    if left_contact and right_contact and abs(vx) < 0.1 and abs(vy) < 0.1:
        return 0

    # 위험한 상황에서는 메인 엔진 우선
    if y < 0.2 and vy < -0.3:
        return 1
    
    # 각도 제어 (더 세밀하게)
    if abs(angle) > 0.3:
        if angle > 0.05:
            return 2 # 왼쪽 엔진
        elif angle < -0.05:
            return 3 # 오른쪽 엔진
    
    # 각속도 제어
    if abs(angular_vel) > 0.3:
        if angular_vel > 0.1:
            return 2
        elif angular_vel < -0.1:
            return 3
    
    # 수평 속도 제어 (더 보수적으로)
    if abs(vx) > 0.3:
        if vx > 0.1:
            return 2
        elif vx < -0.1:
            return 3
    
    # 수직 속도가 너무 빠르면 메인 엔진
    if vy < -0.4:
        return 1
    
    # 안정적인 하강
    if y > 0.5 and abs(angle) < 0.1 and abs(vx) < 0.2:
        if vy < -0.2:  # 너무 빠르게 떨어지지 않도록
            return 1
    
    return 0

# =========================
# 2. Feature Engineering
# =========================

def featurize_state(state):
    """
    상태를 feature vector로 변환
    - 기본 상태 + 추가 feature로 확장
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state

    # 기본 feature
    features = [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]

    # 추가 feature: 속도 크기, 각도 절댓값 등
    features.extend([
        np.sqrt(vx**2 + vy**2), # 속도 크기
        abs(angle),             # 각도 절댓값
        abs(angular_vel),       # 각속도 절댓값
        x * vx,                 # 위치-속도 상호작용
        y * vy,                 
        angle * angular_vel
    ])

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
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, state):
        x = self.dropout(self.relu(self.fc1(state)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        r = self.fc4(x)
        return r.squeeze(-1)

def compute_state_visitation_freq(env, policy, gamma, traj_length, num_trajs=50):
    """
    현재 policy로부터 상태 방문 빈도 추정
    """
    state_dim = len(featurize_state(env.observation_space.sample()))
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
    if not expert_trajs:
        raise ValueError("Expert trajectories가 비어있습니다!")
    
    state_dim = len(featurize_state(expert_trajs[0][0][0]))
    feat_sum = np.zeros(state_dim)
    
    for traj in expert_trajs:
        for t, (state, action) in enumerate(traj):
            feat_sum += featurize_state(state) * (gamma ** t)
    
    return feat_sum / len(expert_trajs)

def sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=5e-3, epochs=50):
    """
    추정된 reward_net을 이용해 policy를 REINFORCE로 학습
    """
    state_dim = len(featurize_state(env.observation_space.sample()))
    action_dim = env.action_space.n

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, action_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)

        def forward(self, state):
            x = self.dropout(self.relu(self.fc1(state)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.dropout(self.relu(self.fc3(x)))
            return self.fc4(x)

    policy_net = PolicyNet()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for epoch in range(epochs):
        states, actions, rewards, log_probs = [], [], [], []
        state, _ = env.reset()
        
        for t in range(traj_length):
            state_feat = featurize_state(state)
            state_tensor = torch.FloatTensor(state_feat).unsqueeze(0)
            
            logits = policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, _, done, truncated, info = env.step(action.item())
            
            # reward는 reward_net에서 추정
            reward = reward_net(state_tensor).item()
            
            states.append(state_feat)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            
            if done or truncated:
                break
        
        if len(rewards) > 0:
            # REINFORCE: discounted return 계산
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns)
            log_probs = torch.stack(log_probs)
            
            # Policy gradient loss
            loss = -torch.mean(log_probs * returns)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Policy RL epoch {epoch}, loss: {loss.item():.4f}, return: {np.sum(rewards):.2f}")
    
    # 최종 policy 함수 반환
    def policy_fn(state):
        with torch.no_grad():
            state_feat = featurize_state(state)
            state_tensor = torch.FloatTensor(state_feat).unsqueeze(0)
            logits = policy_net(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
        return action
    
    return policy_fn

# def maxent_irl(env, expert_trajs, gamma=0.99, lr=1e-3, epochs=30, traj_length=200):
#     """
#     MaxEnt IRL 메인 함수 (개선된 버전)
#     """
#     state_dim = len(featurize_state(env.observation_space.sample()))
#     reward_net = RewardNet(state_dim)
#     optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=1e-4)

#     # Expert feature expectation 계산
#     expert_feat_exp = compute_expert_feature_expectation(expert_trajs, gamma)
#     print("Expert feature expectation shape:", expert_feat_exp.shape)
#     print("Expert feature expectation (first 5):", expert_feat_exp[:5].round(3))

#     # IRL 학습 루프
#     for epoch in range(epochs):
#         # 현재 reward_net으로 policy 학습
#         policy_fn = sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=5e-3, epochs=20)
        
#         # 현재 policy의 feature expectation 계산
#         policy_feat_exp = compute_state_visitation_freq(env, policy_fn, gamma, traj_length, num_trajs=20)
        
#         # IRL loss: expert와 policy의 feature expectation 차이 최소화
#         loss = torch.sum((torch.FloatTensor(policy_feat_exp) - torch.FloatTensor(expert_feat_exp)) ** 2)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 5 == 0:
#             print(f"IRL epoch {epoch}, loss: {loss.item():.4f}")
    
#     return reward_net

def maxent_irl(env, expert_trajs, gamma=0.99, lr=1e-3, epochs=30, traj_length=200):
    state_dim = len(featurize_state(env.observation_space.sample()))
    reward_net = RewardNet(state_dim)
    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=1e-4)

    # Expert feature expectation 계산
    expert_feat_exp = compute_expert_feature_expectation(expert_trajs, gamma)
    print("Expert feature expectation shape:", expert_feat_exp.shape)
    print("Expert feature expectation (first 5):", expert_feat_exp[:5].round(3))

    for epoch in range(epochs):
        # 1. Expert trajectory에서 reward_net의 출력값 합산
        expert_rewards = []
        for traj in expert_trajs:
            traj_reward = 0
            for t, (state, action) in enumerate(traj):
                state_feat = torch.FloatTensor(featurize_state(state)).unsqueeze(0)
                r = reward_net(state_feat)
                traj_reward += (gamma ** t) * r
            expert_rewards.append(traj_reward)
        expert_reward_mean = torch.stack(expert_rewards).mean()

        # 2. 현재 reward_net으로 policy 학습
        policy_fn = sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=5e-3, epochs=20)

        # 3. 현재 policy로 trajectory 샘플링 및 reward_net의 출력값 합산
        policy_rewards = []
        for _ in range(len(expert_trajs)):
            state, _ = env.reset()
            traj_reward = 0
            for t in range(traj_length):
                state_feat = torch.FloatTensor(featurize_state(state)).unsqueeze(0)
                r = reward_net(state_feat)
                traj_reward += (gamma ** t) * r
                action = policy_fn(state)
                next_state, reward, done, truncated, info = env.step(action)
                state = next_state
                if done or truncated:
                    break
            policy_rewards.append(traj_reward)
        policy_reward_mean = torch.stack(policy_rewards).mean()

        # 4. IRL loss: policy와 expert의 reward expectation 차이 최소화
        loss = policy_reward_mean - expert_reward_mean  # MaxEnt IRL의 목적함수
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
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

    # 2. Expert trajectory 수집 (개선된 정책 사용)
    print("Collecting expert trajectories with improved policy...")
    expert_trajs = collect_expert_trajectories(env, improved_expert_policy, num_episodes=500)

    if len(expert_trajs) == 0:
        print("Error: No expert trajectories collected!")
        return

    # 3. MaxEnt IRL로 보상 함수 학습
    print("Running MaxEnt IRL...")
    reward_net = maxent_irl(env, expert_trajs, gamma=gamma, lr=1e-3, epochs=30, traj_length=traj_length)

    # 4. 학습된 보상 함수로 RL policy 학습
    print("Training RL agent with learned reward...")
    learned_policy = sample_policy_from_reward(env, reward_net, gamma, traj_length, lr=5e-3, epochs=100)

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
    
    avg_reward = np.mean(rewards)
    print(f"Learned policy average reward: {avg_reward:.2f}")
    
    # 6. 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, marker='o', alpha=0.7)
    plt.title("Learned Policy Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.axhline(y=200, color='r', linestyle='--', label='Success threshold')
    plt.axhline(y=avg_reward, color='g', linestyle='-', label=f'Average: {avg_reward:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.axvline(x=200, color='r', linestyle='--', label='Success threshold')
    plt.axvline(x=avg_reward, color='g', linestyle='-', label=f'Average: {avg_reward:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()