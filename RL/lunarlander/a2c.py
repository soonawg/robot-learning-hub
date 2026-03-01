# 필요한 라이브러리 가져오기
import numpy as np                  # 수치 계산을 위한 라이브러리
import torch                        # 딥러닝 프레임워크
import torch.nn as nn               # 신경망 모듈
import torch.optim as optim         # 최적화 알고리즘
import torch.nn.functional as F     # 활성화 함수 등 기능 함수
from torch.distributions import Categorical  # 이산 확률 분포 클래스
import gymnasium as gym             # 강화학습 환경
import matplotlib.pyplot as plt     # 시각화 도구

# 학습 가속화를 위해 GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Advantage Actor-Critic(A2C) 네트워크 - Actor와 Critic을 하나의 네트워크로 구현
class A2CNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=128):
        # input_dim: 상태 공간의 차원, n_actions: 행동 공간의 크기, hidden_dim: 은닉층의 뉴런 수
        super(A2CNetwork, self).__init__()
        
        # 상태를 처리하는 공통 특성 추출 레이어
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 입력 차원 -> 은닉층 차원
            nn.ReLU()                          # 비선형성 추가
        )
        
        # Actor 네트워크 - 정책(각 행동의 확률)을 출력
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 첫 번째 은닉층
            nn.ReLU(),                          # 활성화 함수
            nn.Linear(hidden_dim, n_actions)    # 출력층 - 각 행동의 로짓값 출력
        )
        
        # Critic 네트워크 - 현재 상태의 가치를 평가
        self.critic_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 첫 번째 은닉층
            nn.ReLU(),                          # 활성화 함수
            nn.Linear(hidden_dim, 1)            # 출력층 - 상태 가치 출력
        )
    
    def forward(self, x):
        # 공통 특성 추출
        features = self.feature_layer(x)
        
        # Actor: 행동 확률 분포 계산 (소프트맥스 적용)
        action_probs = F.softmax(self.actor_layer(features), dim=-1)
        
        # Critic: 상태 가치 계산
        state_values = self.critic_layer(features)
        
        return action_probs, state_values
    
    def get_action(self, state):
        # 상태를 텐서로 변환하고 디바이스로 이동
        state = torch.FloatTensor(state).to(device)
        # 상태를 네트워크에 통과시켜 행동 확률 얻기
        action_probs, _ = self.forward(state)
        # 확률 분포 생성
        dist = Categorical(action_probs)
        # 분포에서 행동 샘플링
        action = dist.sample()
        
        # 행동의 로그 확률 계산 (훈련에 사용)
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

# A2C 에이전트 클래스 - 학습 및 행동 결정 담당
class A2CAgent:
    def __init__(self, env, gamma=0.99, lr=0.001):
        # env: 강화학습 환경, gamma: 할인율, lr: 학습률
        self.env = env
        self.gamma = gamma  # 미래 보상 할인율
        
        # 환경의 상태 및 행동 공간 정보 저장
        self.input_dim = env.observation_space.shape[0]  # 상태 공간 차원
        self.n_actions = env.action_space.n              # 행동 공간 크기
        
        # A2C 네트워크 및 옵티마이저 초기화
        self.network = A2CNetwork(self.input_dim, self.n_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 학습 데이터 저장 버퍼
        self.rewards = []      # 받은 보상 저장
        self.log_probs = []    # 행동의 로그 확률 저장
        self.state_values = [] # 상태 가치 저장
        self.entropies = []    # 엔트로피 저장 (현재 사용되지 않음)
    
    def train_episode(self, max_steps=1000):
        # 한 에피소드 동안 학습 진행
        state, _ = self.env.reset()  # 환경 초기화
        done = False                 # 종료 여부
        total_reward = 0             # 총 보상
        steps = 0                    # 스텝 수
        
        # 에피소드가 끝나거나 최대 스텝에 도달할 때까지 반복
        while not done and steps < max_steps:
            # 현재 정책에 따라 행동 선택
            action, log_prob = self.network.get_action(state)
            
            # 선택한 행동을 환경에 적용하고 결과 관찰
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated  # 종료 조건 통합
            
            # 트랜지션 저장 (상태, 로그 확률, 보상)
            self.store_transition(state, log_prob, reward)
            
            # 다음 상태로 이동
            state = next_state
            total_reward += reward
            steps += 1
            
            # 에피소드 종료 또는 일정 스텝마다 네트워크 업데이트
            if done or steps % 5 == 0:
                self.update()
        
        return total_reward, steps
    
    def store_transition(self, state, log_prob, reward):
        # 경험 데이터 저장
        # 상태 텐서로 변환
        state_tensor = torch.FloatTensor(state).to(device)
        # 현재 상태의 가치 계산
        _, state_value = self.network(state_tensor)
        
        # 버퍼에 데이터 추가
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.state_values.append(state_value)
    
    def update(self):
        # 네트워크 가중치 업데이트 (정책 최적화)
        # 데이터가 없으면 종료
        if len(self.rewards) == 0:
            return
        
        # 누적 보상(Return) 계산
        R = 0  # 마지막 상태부터의 누적 보상
        returns = []  # 각 상태의 누적 보상 저장
        
        # 역순으로 미래 보상에 대한 할인된 누적 보상 계산
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R  # 벨만 방정식
            returns.insert(0, R)  # 리스트 앞에 삽입하여 순서 유지
        
        # 텐서 변환 및 디바이스 이동
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(self.log_probs).to(device)
        state_values = torch.cat(self.state_values).to(device)
        
        # 어드밴티지(Advantage) 계산 - 실제 Return과 예측 가치의 차이
        advantages = returns - state_values.detach()
        
        # Actor (정책) 손실 계산: -log(π(a|s)) * A(s,a)
        # 어드밴티지가 양수면 해당 행동의 확률을 높이고, 음수면 낮춤
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic (가치) 손실 계산: MSE(V(s), R)
        # 예측 가치와 실제 Return의 차이 최소화
        critic_loss = F.mse_loss(state_values, returns)
        
        # 총 손실 계산 (가치 손실에는 가중치 0.5 적용)
        loss = actor_loss + 0.5 * critic_loss
        
        # 역전파 및 가중치 업데이트
        self.optimizer.zero_grad()  # 기존 그래디언트 초기화
        loss.backward()             # 역전파
        self.optimizer.step()       # 가중치 업데이트
        
        # 버퍼 초기화
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.entropies = []

# A2C 알고리즘 학습 함수
def train_a2c(env_name, num_episodes=1000, max_steps=1000, gamma=0.99, lr=0.001):
    # 환경 생성
    env = gym.make(env_name)
    # 에이전트 초기화
    agent = A2CAgent(env, gamma, lr)
    
    # 각 에피소드의 보상 기록
    episode_rewards = []
    
    # 지정된 에피소드 수만큼 학습
    for episode in range(num_episodes):
        # 한 에피소드 실행 및 결과 저장
        reward, steps = agent.train_episode(max_steps)
        episode_rewards.append(reward)
        
        # 10 에피소드마다 진행 상황 출력
        if (episode+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])  # 최근 10개 에피소드 평균 보상
            print(f"Episode {episode+1}, Steps: {steps}, Reward: {reward:.2f}, Avg Reward (10): {avg_reward:.2f}")
        
        # 100 에피소드마다 학습 결과 그래프 저장
        if (episode+1) % 100 == 0:
            plt.figure(figsize=(10, 5))
            plt.plot(episode_rewards)
            plt.title(f'A2C on {env_name} - Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig(f'a2c_{env_name}_rewards_{episode+1}.png')
            plt.close()
    
    # 최종 학습 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title(f'A2C on {env_name} - Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'a2c_{env_name}_final_rewards.png')
    plt.show()
    
    # 학습된 모델 파라미터 저장
    torch.save(agent.network.state_dict(), f'a2c_{env_name}_model.pth')
    
    # 환경 종료 및 결과 반환
    env.close()
    return agent, episode_rewards

# 학습된 에이전트 평가 함수
def evaluate_agent(agent, env_name, num_episodes=5, render=True):
    # 시각화 옵션으로 환경 생성
    env = gym.make(env_name, render_mode='human' if render else None)
    
    # 지정된 에피소드 수만큼 평가
    for episode in range(num_episodes):
        state, _ = env.reset()  # 환경 초기화
        done = False            # 종료 여부
        total_reward = 0        # 총 보상
        steps = 0               # 스텝 수
        
        # 에피소드가 끝날 때까지 반복
        while not done:
            # 학습된 정책으로 행동 선택 (탐색 없음)
            action, _ = agent.network.get_action(state)
            
            # 선택한 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 종료 조건 통합
            
            # 다음 상태로 이동
            state = next_state
            total_reward += reward
            steps += 1
            
            # 렌더링 옵션이 켜져 있으면 화면에 표시
            if render:
                env.render()
        
        # 에피소드 결과 출력
        print(f"Evaluation Episode {episode+1}, Steps: {steps}, Reward: {total_reward:.2f}")
    
    # 환경 종료
    env.close()

# 메인 실행 부분
if __name__ == "__main__":
    # 학습할 환경 설정
    env_name = "LunarLander-v3"
    
    # A2C 에이전트 학습 실행
    print(f"Training A2C agent on {env_name}...")
    agent, rewards = train_a2c(env_name, num_episodes=500, max_steps=1000)
    
    # 학습된 에이전트 성능 평가
    print(f"\nEvaluating trained agent on {env_name}...")
    evaluate_agent(agent, env_name, num_episodes=3)