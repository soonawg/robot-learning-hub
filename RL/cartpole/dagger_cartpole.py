import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 전문가 정책 (angle + angular velocity)
def expert_policy(state):
    angle = state[2]
    angle_dot = state[3]
    return 0 if angle + angle_dot < 0 else 1

# 환경 및 데이터 수집 세팅
env = gym.make("CartPole-v1", max_episode_steps=500)  # max 스텝 제한
states, actions = [], []
max_steps = 500  # 안전장치

for episode in range(200):
    state, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = expert_policy(state)
        states.append(state)
        actions.append(action)
        state, _, done, _, _ = env.step(action)
        steps += 1
    print(f"Episode {episode+1} finished after {steps} steps.")

states = np.array(states)
actions = np.array(actions)
np.save("states.npy", states)
np.save("actions.npy", actions)
print("✅ 전문가 데이터 저장 완료")

# 액션 분포 확인
plt.hist(actions, bins=2)
plt.title("Action Distribution in Expert Data")
plt.xticks([0,1])
plt.show()

# BC 모델 정의 및 학습
X = torch.tensor(states, dtype=torch.float32)
y = torch.tensor(actions, dtype=torch.long)

class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.model(x)

policy = BCPolicy()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

for epoch in range(20):
    logits = policy(X)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# 평균 성능 평가 함수
def evaluate_policy(policy, env_name="CartPole-v1", episodes=10, render=True):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []
    max_steps = 500

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = torch.argmax(policy(state_tensor)).item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        if render:
            print(f"Episode {ep+1}: Total reward = {total_reward}")

    env.close()
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\n✅ 평균 리턴: {mean_reward:.2f} ± {std_reward:.2f} (over {episodes} episodes)")
    return mean_reward, std_reward

# 평균 성능 테스트
evaluate_policy(policy, episodes=10, render=True)
