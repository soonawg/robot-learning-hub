import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 전문가 정책(Expert Policy) 정의 (CartPole은 간단하므로 rule-based)
def expert_policy(state):
    # 카트 위치와 속도, 막대 각도와 각속도
    x, x_dot, theta, theta_dot = state
    # 막대가 오른쪽으로 기울면 오른쪽으로, 왼쪽이면 왼쪽으로
    return 1 if theta > 0 else 0

# 2. 전문가 데모 데이터 수집
def collect_expert_trajectories(env, n_trajectories=20, max_steps=200):
    trajectories = []
    for _ in range(n_trajectories):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = expert_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action))
            state = next_state
            if terminated or truncated:
                break
        trajectories.append(episode)
    return trajectories

# 3. 상태 특성 추출 함수 (여기서는 상태 그대로 사용)
def featurize_state(state):
    return np.array(state)

# 4. MaxEnt IRL 핵심 함수
class RewardNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.fc(state)

def compute_state_visitation_freq(env, reward_net, gamma=0.99, n_trajs=100, max_steps=200):
    # 정책을 softmax로 근사하여 상태 방문 빈도 추정
    state_dim = env.observation_space.shape[0]
    state_counts = np.zeros(state_dim)
    for _ in range(n_trajs):
        state, _ = env.reset()
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            reward = reward_net(state_tensor).item()
            # Softmax 정책: 보상이 높을수록 선택 확률 증가
            action_probs = [np.exp(reward), np.exp(-reward)]
            action_probs = np.array(action_probs) / np.sum(action_probs)
            action = np.random.choice([0, 1], p=action_probs)
            state_counts += state
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            if terminated or truncated:
                break
    return state_counts / n_trajs

def maxent_irl(env, expert_trajs, state_dim, lr=1e-2, n_iters=100, gamma=0.99):
    reward_net = RewardNet(state_dim)
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    losses = []
    for it in range(n_iters):
        # 1. 전문가 trajectory에서 reward_net의 출력 합
        expert_rewards = []
        for traj in expert_trajs:
            for s, _ in traj:
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                expert_rewards.append(reward_net(s_tensor))
        expert_reward_sum = torch.cat(expert_rewards).mean()

        # 2. 샘플 trajectory에서 reward_net의 출력 합
        sampled_trajs = collect_expert_trajectories(env, n_trajectories=20)
        sample_rewards = []
        for traj in sampled_trajs:
            for s, _ in traj:
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                sample_rewards.append(reward_net(s_tensor))
        sample_reward_sum = torch.cat(sample_rewards).mean()

        # 3. loss 계산 (최대 엔트로피 IRL: 전문가 보상 - 샘플 보상)
        loss = -(expert_reward_sum - sample_reward_sum)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (it+1) % 10 == 0:
            print(f"Iter {it+1}/{n_iters} | Loss: {loss.item():.4f}")

    return reward_net, losses

# 5. 실행 및 시각화
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]

    print("전문가 데모 수집 중...")
    expert_trajs = collect_expert_trajectories(env, n_trajectories=20)
    print("MaxEnt IRL 학습 시작...")
    reward_net, losses = maxent_irl(env, expert_trajs, state_dim, n_iters=50)

    # 학습된 보상 함수 시각화 (각 상태 차원별로)
    test_states = np.linspace(-2.4, 2.4, 100)
    rewards = []
    for x in test_states:
        state = np.array([x, 0, 0, 0], dtype=np.float32)
        reward = reward_net(torch.FloatTensor(state)).item()
        rewards.append(reward)
    plt.plot(test_states, rewards)
    plt.xlabel("Cart Position")
    plt.ylabel("Learned Reward")
    plt.title("MaxEnt IRL reward function")
    plt.show()

    # 학습 곡선
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MaxEnt IRL reward curve")
    plt.show()

        # 학습된 reward_net으로 평가 정책(softmax policy)로 한 에피소드의 reward 합계 출력
    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 200:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        reward = reward_net(state_tensor).item()
        total_reward += reward
        # softmax policy로 행동 선택
        action_probs = [np.exp(reward), np.exp(-reward)]
        action_probs = np.array(action_probs) / np.sum(action_probs)
        action = np.random.choice([0, 1], p=action_probs)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        steps += 1
    print(f"최종 평가 에피소드에서 reward_net의 누적 reward 합계: {total_reward:.4f}")