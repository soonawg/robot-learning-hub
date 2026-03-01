import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        # state와 action을 concatenate해서 사용
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        # action을 one-hot encoding
        if len(action.shape) == 1:
            action_onehot = torch.zeros(action.shape[0], 2)
            action_onehot[range(action.shape[0]), action] = 1
        else:
            action_onehot = action
        
        x = torch.cat([state, action_onehot], dim=-1)
        return self.network(x)

def collect_expert_trajectories(env, num_trajectories=10):
    """전문가 궤적 수집 - 더 나은 휴리스틱 사용"""
    trajectories = []
    
    for _ in range(num_trajectories * 10):  # 더 많이 시도
        trajectory = []
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            # 개선된 휴리스틱 정책
            # state: [position, velocity, angle, angular_velocity]
            position, velocity, angle, angular_velocity = state
            
            # 더 복잡한 제어 로직
            if abs(angle) < 0.03:  # 거의 수직일 때
                action = 1 if velocity > 0 else 0  # 속도 방향으로
            else:
                # 각도와 각속도를 모두 고려
                if angle > 0:
                    action = 1 if angular_velocity > -0.5 else 0
                else:
                    action = 0 if angular_velocity < 0.5 else 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state.copy(), action, reward))
            
            total_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # 더 낮은 기준으로 설정
        if total_reward > 50:  # 100 -> 50으로 낮춤
            trajectories.append(trajectory)
            if len(trajectories) >= num_trajectories:
                break
    
    # 만약 여전히 부족하면 랜덤 시드 바꿔서 더 시도
    if len(trajectories) < num_trajectories:
        print(f"Only collected {len(trajectories)} trajectories, trying with random actions...")
        for _ in range(num_trajectories * 20):
            trajectory = []
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(500):
                # 완전 랜덤도 시도
                action = env.action_space.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                trajectory.append((state.copy(), action, reward))
                
                total_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            if total_reward > 20:  # 매우 낮은 기준
                trajectories.append(trajectory)
                if len(trajectories) >= num_trajectories:
                    break
    
    return trajectories

def policy_gradient_step(policy_net, value_net, reward_net, trajectories, lr=0.001):
    """정책 그래디언트 업데이트 - 안정화된 버전"""
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)
    
    if not trajectories:
        return 0.0, 0.0
    
    policy_loss = 0
    value_loss = 0
    
    for trajectory in trajectories:
        if not trajectory:
            continue
            
        states = torch.FloatTensor([t[0] for t in trajectory])
        actions = torch.LongTensor([t[1] for t in trajectory])
        
        # 학습된 reward로 return 계산
        returns = []
        G = 0
        for t in reversed(trajectory):
            state_tensor = torch.FloatTensor(t[0]).unsqueeze(0)
            action_tensor = torch.LongTensor([t[1]])
            
            with torch.no_grad():
                learned_reward = reward_net(state_tensor, action_tensor).item()
                # reward clipping for stability
                learned_reward = max(min(learned_reward, 10), -10)
            
            G = learned_reward + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # Value function 학습 with gradient clipping
        values = value_net(states).squeeze()
        if values.dim() == 0:
            values = values.unsqueeze(0)
            
        value_loss += nn.MSELoss()(values, returns)
        
        # Policy 학습 with advantage normalization
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        log_probs = torch.log(policy_net(states).gather(1, actions.unsqueeze(1)) + 1e-8).squeeze()
        policy_loss += -(log_probs * advantages).mean()
    
    # 업데이트 with gradient clipping
    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    policy_optimizer.step()
    
    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    value_optimizer.step()
    
    return policy_loss.item(), value_loss.item()

def maxent_irl_step(reward_net, expert_trajectories, policy_trajectories, lr=0.01):
    """MaxEnt IRL 리워드 업데이트"""
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=lr)
    
    expert_loss = 0
    policy_loss = 0
    
    # Expert trajectories - maximize reward
    for trajectory in expert_trajectories:
        for state, action, _ in trajectory:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.LongTensor([action])
            
            reward = reward_net(state_tensor, action_tensor)
            expert_loss -= reward.mean()  # maximize (so negative)
    
    # Policy trajectories - minimize reward (regularization)
    for trajectory in policy_trajectories:
        for state, action, _ in trajectory:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.LongTensor([action])
            
            reward = reward_net(state_tensor, action_tensor)
            policy_loss += reward.mean()
    
    total_loss = expert_loss + 0.1 * policy_loss
    
    reward_optimizer.zero_grad()
    total_loss.backward()
    reward_optimizer.step()
    
    return total_loss.item()

def collect_policy_trajectories(env, policy_net, num_trajectories=5):
    """현재 정책으로 궤적 수집"""
    trajectories = []
    
    for _ in range(num_trajectories):
        trajectory = []
        state, _ = env.reset()
        
        for step in range(500):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            
            state = next_state
            
            if terminated or truncated:
                break
        
        trajectories.append(trajectory)
    
    return trajectories

def evaluate_policy(env, policy_net, num_episodes=10):
    """정책 평가"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs).item()  # greedy
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)

def main():
    # 환경 설정
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 네트워크 초기화
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    reward_net = RewardNetwork(state_dim, action_dim)
    
    # 전문가 궤적 수집
    print("Collecting expert trajectories...")
    expert_trajectories = collect_expert_trajectories(env, num_trajectories=20)
    print(f"Collected {len(expert_trajectories)} expert trajectories")
    
    # 학습
    num_iterations = 100
    evaluation_scores = []
    
    print("Starting MaxEnt IRL training...")
    
    for iteration in range(num_iterations):
        # 1. 현재 정책으로 궤적 수집
        policy_trajectories = collect_policy_trajectories(env, policy_net, num_trajectories=10)
        
        # 2. IRL 단계: 리워드 함수 업데이트
        irl_loss = maxent_irl_step(reward_net, expert_trajectories, policy_trajectories)
        
        # 3. RL 단계: 정책 업데이트
        policy_loss, value_loss = policy_gradient_step(
            policy_net, value_net, reward_net, policy_trajectories
        )
        
        # 4. 평가
        if iteration % 10 == 0:
            mean_reward, std_reward = evaluate_policy(env, policy_net)
            evaluation_scores.append(mean_reward)
            print(f"Iteration {iteration}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  IRL Loss: {irl_loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
    
    # 최종 평가
    final_mean, final_std = evaluate_policy(env, policy_net, num_episodes=100)
    print(f"\nFinal Performance: {final_mean:.2f} ± {final_std:.2f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_iterations, 10), evaluation_scores)
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title('MaxEnt IRL Training Progress on CartPole')
    plt.grid(True)
    plt.show()
    
    # 학습된 정책으로 몇 번 실행해보기
    print("\nRunning learned policy...")
    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()