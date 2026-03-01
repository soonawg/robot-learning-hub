import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

# 하이퍼파라미터
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
LAMBDA = 0.3
BATCH_SIZE = 256

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-2, 2)

# Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = []
        self.capacity = capacity
    
    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

# AWAC Agent
class AWACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = ReplayBuffer()
    
    def update(self):
        if len(self.buffer.buffer) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # critic update
        with torch.no_grad():
            next_action = self.actor.sample(next_states)
            target_q = self.target_critic(next_states, next_action)
            target = rewards + GAMMA * (1 - dones) * target_q
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor update (AWAC)
        with torch.no_grad():
            q_values = self.critic(states, actions)
            v_values = self.critic(states, self.actor.sample(states))
            advantages = q_values - v_values
        weights = torch.exp(advantages / LAMBDA).clamp(max=20)  # stabilize
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        actor_loss = -(weights * log_prob).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # soft target update
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
        
        return critic_loss.item(), actor_loss.item()

# 간단한 학습 루프 예시
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    agent = AWACAgent(env.observation_space.shape[0], env.action_space.shape[0])
    
    for episode in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.actor.sample(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push((state, action, reward, next_state, done))
            state = next_state
            agent.update()
        if episode % 10 == 0:
            print(f"Episode {episode} complete")
    env.close()
