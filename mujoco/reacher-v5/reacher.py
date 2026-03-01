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
    def __init__(self, state_dim, action_dim, hidden_dim=128):  # Reacher는 더 작은 hidden_dim으로도 충분
        super(Actor, self).__init__()
        # Reacher에서는 위치 관련 정보가 중요하므로 ReLU 대신 Tanh 사용
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),  # ReLU에서 Tanh로 변경 (위치 제어에 더 적합)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 마지막 레이어에서 Tanh 제거 (mean에서 처리)
        )
        
        # 행동의 평균과 표준편차 출력
        # 표준편차의 로그값을 학습 가능한 파라미터로 설정 (더 낮은 초기값으로 설정하여 탐색 줄임)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)  # 초기값 조정
        
    def forward(self, state):
        # 네트워크를 통과시켜 행동의 평균을 계산하고 Tanh 적용
        x = self.actor(state)
        mean = torch.tanh(x)  # 별도 레이어 대신 직접 tanh 적용
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            return mean
        
        dist = Normal(mean, std)
        action = dist.sample()
        # Reacher는 행동 범위가 제한되어 있으므로 clamp 적용
        action = torch.clamp(action, -1.0, 1.0)  # 행동을 [-1, 1] 범위로 제한
        return action
    
    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

# Critic 네트워크 (가치 함수 네트워크)
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):  # Reacher는 더 작은 hidden_dim으로도 충분
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

# 메모리 버퍼 (원래 코드와 동일)
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def add(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        
    def get_batches(self, batch_size):
        data_len = len(self.states)
        if data_len < batch_size:
            batch_size = data_len
        
        indices = np.arange(data_len)
        np.random.shuffle(indices)
        
        for i in range(0, data_len, batch_size):
            batch_indices = indices[i:min(i + batch_size, data_len)]
            yield (
                torch.FloatTensor(np.array(self.states)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.actions)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.rewards)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.next_states)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.dones)[batch_indices]).to(device),
                torch.FloatTensor(np.array(self.log_probs)[batch_indices]).to(device)
            )

# Reacher 환경에 맞게 보상 스케일링 함수 추가
def scale_reward(reward):
    """Reacher의 보상은 주로 음수이므로 적절히 스케일링"""
    return reward * 10  # 보상 스케일링으로 학습 촉진

# PPO 에이전트
class PPO:
    def __init__(self, state_dim, action_dim,
                 lr_actor=3e-4,
                 lr_critic=1e-3,       # Critic 학습률 증가 (Reacher에서 가치 함수 빠른 학습 유도)
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 update_epochs=10,
                 batch_size=64):       # 배치 크기 감소 (Reacher는 더 작은 배치로도 안정적)
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.buffer = RolloutBuffer()
        
    def select_action(self, state, evaluation=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor.get_action(state, deterministic=evaluation)
            action = action.cpu().numpy().flatten()
            
            if not evaluation:
                mean, std = self.actor(state)
                dist = Normal(mean, std)
                log_prob = dist.log_prob(torch.FloatTensor(action).to(device)).sum(dim=-1)
                return action, log_prob.item()
            
            return action
    
    def compute_advantages(self, states, rewards, next_states, dones):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            if len(values.shape) == 1:
                values = values.unsqueeze(1)
            if len(next_values.shape) == 1:
                next_values = next_values.unsqueeze(1)
            if len(rewards.shape) == 1:
                rewards = rewards.unsqueeze(1)
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values
            
            return advantages.squeeze(), returns.squeeze()
    
    def update(self):
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(device)
        next_states = torch.FloatTensor(np.array(self.buffer.next_states)).to(device)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(device)
        
        for _ in range(self.update_epochs):
            for state_batch, action_batch, reward_batch, next_state_batch, done_batch, old_log_prob_batch in self.buffer.get_batches(self.batch_size):
                log_probs, entropy = self.actor.evaluate(state_batch, action_batch)
                
                advantages_batch, returns_batch = self.compute_advantages(
                    state_batch, 
                    reward_batch,
                    next_state_batch, 
                    done_batch
                )
                
                advantages_batch = advantages_batch.reshape(-1)
                returns_batch = returns_batch.reshape(-1)
                
                if advantages_batch.numel() > 1:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                ratio = torch.exp(log_probs - old_log_prob_batch)
                
                advantages_batch = advantages_batch.unsqueeze(-1)
                
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                loss_actor = -torch.min(ratio * advantages_batch, clip_adv).mean()
                
                values = self.critic(state_batch).squeeze()
                loss_critic = nn.MSELoss()(values, returns_batch)
                
                # 엔트로피 계수 증가 (Reacher는 탐색이 중요)
                loss_entropy = -0.02 * entropy.mean()  # 엔트로피 계수 증가
                
                loss = loss_actor + 0.5 * loss_critic + loss_entropy
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # 그래디언트 클리핑 추가 (안정성)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.buffer.clear()

# 학습 함수 (Reacher에 최적화)
def train(env_name, num_episodes=1000, max_timesteps=1000, update_frequency=50, render_every=100):
    # wandb 초기화
    wandb.init(project="Reacher", config={
        "env_name": env_name,
        "num_episodes": num_episodes,
        "max_timesteps": max_timesteps,
        "update_frequency": update_frequency,  # 새 파라미터 추가
        "render_every": render_every,
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,  # 증가된 critic 학습률
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "update_epochs": 10,
        "batch_size": 64  # 감소된 배치 크기
    })
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPO(state_dim, action_dim)
    
    episode_rewards = []
    avg_rewards = []
    
    # 목표 위치 시각화를 위한 변수 추가
    target_positions = []
    achieved_distances = []
    
    for episode in range(1, num_episodes + 1):
        if episode % render_every == 0:
            env.close()
            env = gym.make(env_name, render_mode="human")
        
        state, _ = env.reset()
        episode_reward = 0
        
        # Reacher에서 목표 위치 추출 (마지막 2개 요소)
        if len(state) >= 11:  # Reacher-v5의 관측 크기 확인 (실제 환경에 맞게 조정 필요)
            target_pos = state[-2:]  # 마지막 2개 요소가 목표 위치
            target_positions.append(target_pos)
        
        steps_since_update = 0  # 업데이트 빈도를 추적하기 위한 카운터
        
        for t in range(max_timesteps):
            action, log_prob = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Reacher를 위한 보상 스케일링
            scaled_reward = scale_reward(reward)
            
            agent.buffer.add(state, action, scaled_reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward  # 원래 보상을 기록에 사용
            
            steps_since_update += 1
            
            # 주기적인 업데이트 (Reacher는 더 자주 업데이트하면 효과적)
            if steps_since_update >= update_frequency or done or t == max_timesteps - 1:
                if len(agent.buffer.states) >= agent.batch_size:  # 배치 크기 이상일 때만 업데이트
                    agent.update()
                steps_since_update = 0
            
            if done:
                break
        
        # 목표까지 최종 거리 기록 (있을 경우)
        if len(state) >= 11:
            target = state[-2:]
            finger_pos = state[0:2]  # 실제 인덱스는 환경에 따라 다를 수 있음
            final_distance = np.linalg.norm(finger_pos - target)
            achieved_distances.append(final_distance)
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            if len(achieved_distances) > 0:
                print(f"Latest distance to target: {achieved_distances[-1]:.4f}")
        
        if episode % render_every == 0:
            evaluate(env_name, agent)
        
        # wandb에 메트릭 로깅 (거리 정보 추가)
        log_data = {"episode_reward": episode_reward, "avg_reward": avg_reward}
        if len(achieved_distances) > 0:
            log_data["distance_to_target"] = achieved_distances[-1]
        wandb.log(log_data)
    
    env.close()
    
    # 결과 시각화 (보상 및 목표 거리)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'PPO on {env_name} - Rewards')
    plt.legend()
    
    if len(achieved_distances) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(achieved_distances, label='Distance to Target')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.title('Distance to Target')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{env_name}_training_results.png')
    plt.show()
    
    return agent

# 평가 함수 (원래 코드와 동일)
def evaluate(env_name, agent, num_episodes=3):
    env = gym.make(env_name, render_mode="human")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluation=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    env.close()

# 메인 함수
if __name__ == "__main__":
    # Reacher-v5 환경에서 PPO 학습
    agent = train("Reacher-v5", num_episodes=500, max_timesteps=50, update_frequency=25, render_every=50)
    
    # 학습된 에이전트 평가
    evaluate("Reacher-v5", agent, num_episodes=5)