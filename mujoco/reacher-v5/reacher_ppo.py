import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import wandb


# 시드 설정
torch.manual_seed(0)
np.random.seed(0)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor 네트워크 (정책 네트워크)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # ReLU 활성화 함수는 기울기 소실 문제를 줄이고 학습 속도를 향상시킴
        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(), # Tanh 대신 ReLU로 변경
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU()
        # )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), # Tanh 대신 ReLU로 변경
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 행동의 평균과 표준편차 출력
        # Tanh를 사용하여 행동값을 [-1, 1] 범위로 제한 (Hopper 환경의 행동 공간에 맞춤)
        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # 표준편차의 로그값을 학습 가능한 파라미터로 설정 (안정성을 위해 로그 스케일 사용)
        # self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.log_std = nn.Parameter(torch.zeros(1, action_dim) * -0.5)
        
    def forward(self, state):
        # 네트워크를 통과시켜 행동의 평균과 표준편차를 계산
        # 표준편차는 항상 양수가 되도록 지수함수 적용
        x = self.actor(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        # 상태에 따른 행동을 샘플링 혹은 결정적으로 선택
        # deterministic=True: 평가 시 사용, 평균값만 반환
        # deterministic=False: 학습 시 사용, 정규분포에서 샘플링하여 탐색 보장
        mean, std = self.forward(state)
        
        if deterministic:
            return mean # 평가 시에는 평균값만 사용 (결정적 정책)
        
        dist = Normal(mean, std) # 정규분포 생성
        action = dist.sample() # 분포에서 행동 샘플링 (탐색촉진)
        return action
    
    def evaluate(self, state, action):
        # 주어진 상태와 행동에 대한 로그 확률과 엔트로피 계산
        # PPO 알고리즘의 목적함수 계산에 필요
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # 모든 행동 차원의 로그 확률 합
        entropy = dist.entropy().sum(dim=-1, keepdim=True)  # 정책의 불확실성 측정, 탐색을 촉진
        return log_prob, entropy

# Critic 네트워크 (가치 함수 네트워크)
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        # Critic은 상태의 가치를 추정하는 네트워크
        # Actor와 달리 Tanh 활성화 함수 사용 (가치 추정에 적합)
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), # 가치 추정에는 Tanh가 안정적인 결과를 제공
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # 상태 가치 (스칼라 값)을 제공
        )
    
    def forward(self, state):
        # 상태의 가치를 계산
        return self.critic(state)

# 메모리 버퍼
class RolloutBuffer:
    def __init__(self):
        # 경험 데이터를 저장하는 버퍼
        # PPO는 온라인 알고리즘이므로 현재 정책으로 수집한 데이터만 사용함
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def add(self, state, action, reward, next_state, done, log_prob):
        # 경험 데이터를 버퍼에 추가함
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        
    def clear(self):
        # 버퍼를 비움 (PPO는 온라인 알고리즘이므로 업데이트 후 데이터를 재사용하지 않음)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def get_batches(self, batch_size):
        # 미니배치 학습을 위해 데이터를 배치 단위로 제공하는 제너레이터
        # 무작위로 샘플링하여 데이터 간 상관관계 감소
        data_len = len(self.states)
        if data_len < batch_size:  # 데이터가 배치 크기보다 작으면 전체 데이터 사용
            batch_size = data_len
        
        indices = np.arange(data_len)
        np.random.shuffle(indices) # 데이터 순서를 무작위로 섞음
        
        for i in range(0, data_len, batch_size):
            batch_indices = indices[i:min(i + batch_size, data_len)]
            yield (  # 배치 데이터를 텐서로 변환하여 반환
                torch.FloatTensor(np.array(self.states)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.actions)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.rewards)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.next_states)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.dones)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.log_probs)[batch_indices]).to(device)
            )

# PPO 에이전트
class PPO:
    def __init__(self, state_dim, action_dim,
                 lr_actor=3e-4,        # Actor 학습률 # 현재 적절
                 lr_critic=3e-4,       # Critic 학습률 # 현재 적절
                 gamma=0.99,           # discount factor (미래 보상의 중요도 결정) # 0.99 -> 0.995로 변경함 (오히려 성능 감소) -> 다시 0.99로 복귀
                 gae_lambda=0.95,      # GAE 파라미터 (Advantage 추정의 편향-분산 균형 조절 # 현재 적절
                 clip_ratio=0.2,       # PPO 클리핑 파라미터 # 0.2에서 0.1로 감소함 (오히려 성능 감소) -> 다시 0.2로 복귀
                 update_epochs=10,     # 각 업데이트마다 반복 횟수 # 10에서 15로 증가시킴 (오히려 성능 감소) -> 다시 10으로 복귀
                 batch_size=256):      # 미니배치 크기 # 현재 적절
        # PPO 알고리즘 구현, Actor-Critic 구조 사용
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer() # 경험 데이터 저장소
        
    def select_action(self, state, evaluation=False):
        # 현재 정책을 기반으로 행동을 선택
        # evaluation=True: 결정적 행동 (평가용)
        # evaluation=False: 확률적 행동 (학습용, 탐색 포함)
        with torch.no_grad(): # 그래디언트 계산 비활성화 (학습 아닌 추론 단계)
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor.get_action(state, deterministic=evaluation)
            action = action.cpu().numpy().flatten()
            
            if not evaluation:
                # 학습 시에는 행동의 로그 확률도 계산 (PPO 업데이트에 필요)
                mean, std = self.actor(state)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(torch.FloatTensor(action).to(device)).sum(dim=-1)
                return action, log_prob.item()
            
            return action
    
    def compute_advantages(self, states, rewards, next_states, dones):
        # Generalized Advantge Estimation (GAE) 계산
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            # 차원 확인 및 조정 (텐서 연산의 일관성 유지)
            if len(values.shape) == 1:
                values = values.unsqueeze(1)
            if len(next_values.shape) == 1:
                next_values = next_values.unsqueeze(1)
            if len(rewards.shape) == 1:
                rewards = rewards.unsqueeze(1)
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(1)
            
            # GAE(Generalized Advantage Estimation) 계산
            # λ-weighted 시간차 오차의 합으로 advantage 추정
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(len(rewards))): # 역순으로 추정 (재귀적 특성)
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                
                # 시간차 오차 계산
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                # GAE 업데이트
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            # 반환값 계산 (advantage + 가치)
            returns = advantages + values
            
            return advantages.squeeze(), returns.squeeze()
    
    def update(self):
        # PPO 알고리즘의 핵심 업데이트 로직
        # 수집한 데이터로 정책과 가치 함수를 개선
        # 모든 데이터를 텐서로 변환
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(device)
        next_states = torch.FloatTensor(np.array(self.buffer.next_states)).to(device)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(device)
        
        # PPO 업데이트 수행 (여러 epoch 반복)
        # 같은 데이터를 여러 번 사용하여 효율성 증가
        for _ in range(self.update_epochs):
            for state_batch, action_batch, reward_batch, next_state_batch, done_batch, old_log_prob_batch in self.buffer.get_batches(self.batch_size):
                # 현재 정책에서의 로그 확률 및 엔트로피 계산
                log_probs, entropy = self.actor.evaluate(state_batch, action_batch)
                
                # 현재 배치에 대한 advantage와 returns 계산
                advantages_batch, returns_batch = self.compute_advantages(
                    state_batch, 
                    reward_batch,
                    next_state_batch, 
                    done_batch
                )
                
                # 차원 맞추기 (텐서 연산 준비)
                advantages_batch = advantages_batch.reshape(-1)
                returns_batch = returns_batch.reshape(-1)
                
                # 정규화 (배치가 충분히 크지 않을 때의 문제 해결 | 학습 안정성 향상)
                if advantages_batch.numel() > 1:  # 배치 크기가 1보다 큰 경우에만 정규화
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # 비율 계산 (현재 정책 / 이전 정책)
                # PPO의 핵심 아이디어 : 현재 정책과 데이터 수집 정책의 차이 제한
                ratio = torch.exp(log_probs - old_log_prob_batch)
                
                # advantages_batch를 ratio와 같은 shape로 맞추기
                advantages_batch = advantages_batch.unsqueeze(-1)
                
                # PPO 클리핑 목적함수 계산
                # 정책 업데이트를 제한하여 급격한 변화 방지
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                loss_actor = -torch.min(ratio * advantages_batch, clip_adv).mean()
                
                # 가치 함수 손실 계산 (MSE)
                values = self.critic(state_batch).squeeze()
                loss_critic = nn.MSELoss()(values, returns_batch)
                
                # 엔트로피 보너스 (탐색 촉진)
                loss_entropy = -0.01 * entropy.mean()
                
                # 전체 손실 계산
                loss = loss_actor + 0.5 * loss_critic + loss_entropy
                
                # 네트워크 업데이트
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        # 버퍼 비우기 (온라인 업데이트이므로 데이터 재사용 안함)
        self.buffer.clear()

# 학습 함수
def train(env_name, num_episodes=1000, max_timesteps=1000, render_every=100):
    # wandb 초기화
    wandb.init(project="ppo-hopper", config={
        "env_name": env_name,
        "num_episodes": num_episodes,
        "max_timesteps": max_timesteps,
        "render_every": render_every,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "update_epochs": 10,
        "batch_size": 256
    })
    
    # 전체 학습 과정을 관리하는 함수
    # 초기 환경 생성
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPO(state_dim, action_dim)
    
    # 학습 기록 저장용 리스트
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(1, num_episodes + 1):
        # 렌더링이 필요한 경우에만 환경을 새로 생성
        if episode % render_every == 0:
            env.close()  # 이전 환경 닫기
            env = gym.make(env_name, render_mode="human")
        
        state, _ = env.reset()  # 각 에피소드 시작 시 환경 초기화
        episode_reward = 0
        
        for t in range(max_timesteps):
            # 행동 선택
            action, log_prob = agent.select_action(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 버퍼에 추가
            agent.buffer.add(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward
            
            # 에피소드가 끝나거나 버퍼가 충분히 차면 업데이트
            if done or t == max_timesteps - 1:
                agent.update()
                break
        
        # 결과 기록
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) # 최근 100개 에피소드의 평균
        avg_rewards.append(avg_reward)
        
        # 진행 상황 출력
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        
        # 평가 (특정 에피소드마다)
        if episode % render_every == 0:
            evaluate(env_name, agent)
        
        # wandb에 메트릭 로깅
        wandb.log({"episode_reward": episode_reward, "avg_reward": avg_reward})
    
    env.close()
    
    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'PPO on {env_name}')
    plt.legend()
    plt.savefig(f'{env_name}_rewards.png')
    plt.show()
    
    return agent

# 평가 함수
def evaluate(env_name, agent, num_episodes=5):
    # 학습된 에이전트의 성능을 평가하는 함수
    # 결정적 행동 선택과 시각화를 포함
    env = gym.make(env_name, render_mode="human")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 결정적 행동 선택 (노이즈 없음)
            action = agent.select_action(state, evaluation=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    env.close()

# 메인 함수
if __name__ == "__main__":
    # Hopper-v5 환경에서 PPO 학습
    agent = train("Reacher-v5", num_episodes=500, render_every=50)
    
    # 학습된 에이전트 평가
    evaluate("Reacher-v5", agent, num_episodes=3)