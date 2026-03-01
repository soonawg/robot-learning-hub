import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import wandb
import os
from datetime import datetime


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
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # 표준편차를 제한하여 안정적인 학습 보장
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        self.log_std_min, self.log_std_max = -20, 2

    def forward(self, state):
        x = self.actor(state)
        mean = self.mean(x)
        # 표준편차 클리핑으로 안정성 향상
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            return mean
        
        dist = Normal(mean, std)
        action = dist.rsample()  # reparameterization trick 사용
        return torch.clamp(action, -1.0, 1.0)  # 행동 클리핑
    
    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

# Critic 네트워크 (가치 함수 네트워크) - 2개의 크리틱으로 학습 안정성 향상
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # 첫 번째 Q 네트워크
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 두 번째 Q 네트워크
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        # 두 Q값의 평균 반환
        value1 = self.critic1(state)
        value2 = self.critic2(state)
        return (value1 + value2) / 2.0
    
    def both_values(self, state):
        # 두 Q값 모두 반환
        value1 = self.critic1(state)
        value2 = self.critic2(state)
        return value1, value2

class RolloutBuffer:
    def __init__(self, buffer_size=2048):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.buffer_size = buffer_size
        self.size = 0
        
    def add(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.size += 1
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.size = 0
        
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
    
    def is_full(self):
        return self.size >= self.buffer_size


class PPO:
    def __init__(self, state_dim, action_dim,
                lr_actor=3e-4,
                lr_critic=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_ratio=0.2,
                update_epochs=10,
                batch_size=64,
                entropy_coef=0.01,
                value_clip_ratio=0.2,
                buffer_size=2048,
                target_kl=0.01
                ):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_clip_ratio = value_clip_ratio
        self.target_kl = target_kl
        
        self.buffer = RolloutBuffer(buffer_size)
        
        # 학습률 스케줄러 추가
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)
        
        # 모델 저장 관련
        self.best_reward = -float('inf')
        self.save_dir = f"models/walker2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
            
            # 보상 클리핑 추가 (학습 안정성 향상)
            rewards = torch.clamp(rewards, -10.0, 10.0)
            
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
        if len(self.buffer.states) == 0:
            return
            
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(device)
        next_states = torch.FloatTensor(np.array(self.buffer.next_states)).to(device)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(device)
        
        # 전체 데이터에 대한 가치 계산
        with torch.no_grad():
            old_values = self.critic(states)
            
        advantages, returns = self.compute_advantages(states, rewards, next_states, dones)
        
        # 여러 epoch 반복 학습
        for epoch in range(self.update_epochs):
            # 데이터셋을 미니배치로 분할
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
                
                # 차원 맞추기
                advantages_batch = advantages_batch.reshape(-1)
                returns_batch = returns_batch.reshape(-1)
                
                # advantage 정규화
                if advantages_batch.numel() > 1:
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                
                # PPO 비율 계산
                ratio = torch.exp(log_probs - old_log_prob_batch)
                
                # advantages_batch를 ratio와 같은 shape로 맞추기
                advantages_batch = advantages_batch.unsqueeze(-1)
                
                # PPO 클리핑 목적함수
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch
                loss_actor = -torch.min(surr1, surr2).mean()
                
                # 가치 함수 손실 계산 - 클리핑 추가
                values = self.critic(state_batch).squeeze()
                
                # 가치 함수 클리핑 (안정성 향상)
                value_pred_clipped = old_values + torch.clamp(
                    values - old_values, -self.value_clip_ratio, self.value_clip_ratio
                )
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                loss_critic = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # 엔트로피 보너스
                loss_entropy = -self.entropy_coef * entropy.mean()
                
                # 전체 손실 계산
                loss = loss_actor + loss_critic + loss_entropy
                
                # KL divergence 계산 (early stopping 용)
                approx_kl = ((old_log_prob_batch - log_probs)**2).mean()
                
                # 네트워크 업데이트
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 그라디언트 클리핑 (gradient explosion 방지)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # KL divergence가 너무 커지면 업데이트 중단
                if approx_kl > self.target_kl:
                    break
        
        # 학습률 스케줄러 단계 진행
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 버퍼 비우기
        self.buffer.clear()
    
    def save_model(self, episode_reward):
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'reward': episode_reward
            }, f"{self.save_dir}/best_model.pt")
            print(f"모델 저장됨! 최고 보상: {episode_reward:.2f}")
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.best_reward = checkpoint['reward']
        print(f"모델 로드됨! 최고 보상: {self.best_reward:.2f}")


def normalize_reward(rewards, epsilon=1e-8):
    # 보상 정규화 함수 추가
    return (rewards - rewards.mean()) / (rewards.std() + epsilon)


def train(env_name, num_episodes=1000, max_timesteps=1000, render_every=100):
    # wandb 초기화
    run_name = f"Walker2d_PPO_{datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb.init(project="walker2d", name=run_name, config={
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
        "batch_size": 64,
        "entropy_coef": 0.01,
        "buffer_size": 2048
    })
    
    # 환경 생성 및 시드 설정
    env = gym.make(env_name)
    env.action_space.seed(0)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPO(state_dim, action_dim, 
                batch_size=64, 
                buffer_size=2048, 
                entropy_coef=0.01)
    
    episode_rewards = []
    avg_rewards = []
    
    update_freq = 2048  # 몇 개의 타임스텝마다 업데이트할지
    step_count = 0
    
    for episode in range(1, num_episodes + 1):
        if episode % render_every == 0:
            env.close()
            env = gym.make(env_name, render_mode="human")
        
        state, _ = env.reset(seed=episode)  # 매 에피소드마다 다른 시드 사용
        episode_reward = 0
        
        for t in range(max_timesteps):
            action, log_prob = agent.select_action(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 버퍼에 추가
            agent.buffer.add(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # 정해진 타임스텝마다 학습 수행
            if step_count >= update_freq:
                agent.update()
                step_count = 0
            
            if done:
                break
        
        # 에피소드가 끝났을 때 버퍼에 데이터가 있으면 업데이트
        if len(agent.buffer.states) > 0:
            agent.update()
        
        # 결과 기록
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # 모델 저장
        agent.save_model(episode_reward)
        
        # 진행 상황 출력
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
        
        # 평가
        if episode % render_every == 0:
            evaluate(env_name, agent)
        
        # wandb에 메트릭 로깅
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward, 
            "avg_reward": avg_reward,
            "actor_lr": agent.actor_optimizer.param_groups[0]['lr'],
            "critic_lr": agent.critic_optimizer.param_groups[0]['lr']
        })
    
    env.close()
    
    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Improved PPO on {env_name}')
    plt.legend()
    plt.savefig(f'{env_name}_rewards_improved.png')
    plt.show()
    
    return agent


def evaluate(env_name, agent, num_episodes=3):
    env = gym.make(env_name, render_mode="human")
    
    eval_rewards = []
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
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    avg_eval_reward = np.mean(eval_rewards)
    print(f"평균 평가 보상: {avg_eval_reward:.2f}")
    wandb.log({"eval_reward": avg_eval_reward})
    
    env.close()
    return avg_eval_reward


if __name__ == "__main__":
    agent = train("Walker2d-v5", num_episodes=1000, render_every=50)
    
    evaluate("Walker2d-v5", agent, num_episodes=5)