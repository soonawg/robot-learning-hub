import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque

# 재현 가능성을 위한 시드 설정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 환경 생성
env = gym.make("LunarLander-v3")
env.reset(seed=seed)

# 상태 및 액션 공간 정보
state_dim = env.observation_space.shape[0]  # 8
action_dim = env.action_space.n             # 4

# Q-Network 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, 
                 gamma=0.99, buffer_size=100000, batch_size=64, 
                 tau=0.001, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, device="cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 할인 계수
        self.tau = tau      # 타겟 네트워크 소프트 업데이트 계수
        self.batch_size = batch_size
        
        # epsilon 값 초기화 (탐험-활용 균형)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 장치 설정
        self.device = device
        
        # Q-Network 및 타겟 네트워크
        self.qnetwork_local = QNetwork(state_dim, action_dim).to(self.device)
        self.qnetwork_target = QNetwork(state_dim, action_dim).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # 리플레이 버퍼
        self.memory = ReplayBuffer(buffer_size, batch_size, self.device)
        
        # 학습 스텝 카운터
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # 경험을 리플레이 버퍼에 저장
        self.memory.add(state, action, reward, next_state, done)
        
        # 메모리에 충분한 샘플이 쌓이면 학습 수행
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def act(self, state, eval_mode=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Epsilon-greedy 액션 선택
        if not eval_mode and random.random() < self.epsilon:
            return np.array([random.randrange(self.action_dim)])
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # 최대 Q 값을 가진 액션 선택
        return np.argmax(action_values.cpu().data.numpy(), axis=1)
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # 현재 Q 값 계산
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # 타겟 Q 값 계산
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Q 네트워크 업데이트
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # Epsilon 감소
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# 훈련 함수
def train_dqn(env, agent, n_episodes=2000, max_t=1000, 
              epsilon_decay=0.995, print_every=100):
    
    scores = []
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action.item())
            agent.step(state, action, reward, next_state, done or truncated)
            state = next_state
            score += reward
            
            if done or truncated:
                break
                
        scores_window.append(score)
        scores.append(score)
        
        # Epsilon 감소
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * epsilon_decay)
        
        # 진행 상황 출력
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {agent.epsilon:.2f}')
        
        # 환경이 해결되었는지 확인 (LunarLander-v2의 경우 200점 이상)
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoint.pth')
            break
            
    return scores

# 테스트 함수
def test_agent(env, agent, n_episodes=10, render=False):
    if render:
        env = gym.make("LunarLander-v3", render_mode="human")  # v2에서 v3로 변경
    scores = []
    
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        
        while True:
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action.item())
            state = next_state
            score += reward
            
            if done or truncated:
                break
                
        scores.append(score)
        print(f'Episode {i+1}\tScore: {score:.2f}')
        
    print(f'Average Score: {np.mean(scores):.2f}')
    return scores

# 결과 시각화 함수
def plot_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('DQN Training Progress')
    plt.savefig('dqn_training.png')
    plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    learning_rate = 0.001
    gamma = 0.99
    buffer_size = 100000
    batch_size = 64
    tau = 0.001
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    # 장치 설정 (GPU가 있으면 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # DQN 에이전트 초기화
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, 
                    learning_rate=learning_rate, gamma=gamma,
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                    device=device)
    
    # 훈련 실행
    scores = train_dqn(env, agent, n_episodes=1000, print_every=10)
    
    # 결과 시각화
    plot_scores(scores)
    
    # 학습된 에이전트 테스트
    test_scores = test_agent(env, agent, n_episodes=5, render=True)
    
    # 환경 닫기
    env.close()