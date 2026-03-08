import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def expert_policy(state):
    x, x_dot, theta, theta_dot = state
    return 1 if theta > 0 else 0

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

def featurize_state(state):
    return np.array(state)

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
    state_dim = env.observation_space.shape[0]
    state_counts = np.zeros(state_dim)
    for _ in range(n_trajs):
        state, _ = env.reset()
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            reward = reward_net(state_tensor).item()
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
        expert_rewards = []
        for traj in expert_trajs:
            for s, _ in traj:
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                expert_rewards.append(reward_net(s_tensor))
        expert_reward_sum = torch.cat(expert_rewards).mean()

        sampled_trajs = collect_expert_trajectories(env, n_trajectories=20)
        sample_rewards = []
        for traj in sampled_trajs:
            for s, _ in traj:
                s_tensor = torch.FloatTensor(s).unsqueeze(0)
                sample_rewards.append(reward_net(s_tensor))
        sample_reward_sum = torch.cat(sample_rewards).mean()

        loss = -(expert_reward_sum - sample_reward_sum)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (it+1) % 10 == 0:
            print(f"Iter {it+1}/{n_iters} | Loss: {loss.item():.4f}")

    return reward_net, losses

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]

    print("전문가 데모 수집 중...")
    expert_trajs = collect_expert_trajectories(env, n_trajectories=20)
    print("MaxEnt IRL 학습 시작...")
    reward_net, losses = maxent_irl(env, expert_trajs, state_dim, n_iters=50)

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

    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MaxEnt IRL reward curve")
    plt.show()

    state, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 200:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        reward = reward_net(state_tensor).item()
        total_reward += reward
        action_probs = [np.exp(reward), np.exp(-reward)]
        action_probs = np.array(action_probs) / np.sum(action_probs)
        action = np.random.choice([0, 1], p=action_probs)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        steps += 1
    print(f"최종 평가 에피소드에서 reward_net의 누적 reward 합계: {total_reward:.4f}")
