import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 하이퍼파라미터
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 1000

# Q-네트워크 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 환경 설정
env = gym.make('Taxi-v3', render_mode="human")
input_size = env.observation_space.n
output_size = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# 훈련 루프
for episode in range(EPISODES):
    state = env.reset()
    state = torch.tensor([state], dtype=torch.float32, device=device)
    done = False
    total_reward = 0

    while not done:
        # 엡실론 탐욕 정책
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor([next_state], dtype=torch.float32, device=device)
        total_reward += reward

        # Q-값 업데이트
        q_value = policy_net(state)[0, action]
        with torch.no_grad():
            target_q_value = reward + GAMMA * target_net(next_state).max().item()
        loss = criterion(q_value, torch.tensor([target_q_value], device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    # 타겟 네트워크 업데이트
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 엡실론 감소
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()