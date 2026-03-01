import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x

# Critic Network (TD3 uses double Q)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

# Replay Buffer (offline dataset)
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size=1000000):
        self.states = np.zeros((size, state_dim))
        self.actions = np.zeros((size, action_dim))
        self.next_states = np.zeros((size, state_dim))
        self.rewards = np.zeros((size, 1))
        self.dones = np.zeros((size, 1))
        self.ptr, self.max_size = 0, size
    
    def add(self, s, a, r, s_, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
    
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        return dict(
            s=torch.FloatTensor(self.states[idxs]),
            a=torch.FloatTensor(self.actions[idxs]),
            r=torch.FloatTensor(self.rewards[idxs]),
            s_=torch.FloatTensor(self.next_states[idxs]),
            d=torch.FloatTensor(self.dones[idxs])
        )

# TD3+BC Trainer
class TD3_BC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.alpha = 2.5  # BC loss weight
    
    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5):
        data = replay_buffer.sample(batch_size)
        s, a, r, s_, d = data["s"], data["a"], data["r"], data["s_"], data["d"]

        with torch.no_grad():
            noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            next_a = self.actor_target(s_) + noise
            next_a = next_a.clamp(-self.max_action, self.max_action)
            q1_target, q2_target = self.critic_target(s_, next_a)
            q_target = torch.min(q1_target, q2_target)
            target = r + discount * (1 - d) * q_target

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update with BC loss
        pi = self.actor(s)
        q1_pi, _ = self.critic(s, pi)
        bc_loss = F.mse_loss(pi, a)
        actor_loss = -q1_pi.mean() + self.alpha * bc_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # target networks update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

# 사용 예시
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3_BC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # 예를 들어 offline 데이터셋으로 채우는 부분
    for _ in range(10000):
        s, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            replay_buffer.add(s, a, r, s_, done)
            s = s_

    # 간단한 학습 루프
    for step in range(1000):
        critic_loss, actor_loss = agent.train(replay_buffer)
        if step % 100 == 0:
            print(f"Step {step}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")

