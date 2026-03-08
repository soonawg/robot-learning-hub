import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque


class ExpertDataCollector:
    def __init__(self, env):
        self.env = env
        self.expert_data = []
    
    def collect_expert_data(self, num_episodes=500):
        all_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            episode_data = []

            while not (done or truncated):
                action = self.simple_policy(state)

                next_state, reward, done, truncated, info = self.env.step(action)

                episode_data.append((state, action))

                state = next_state
                total_reward += reward
        
            all_rewards.append(total_reward)
            
            if episode >= 100:
                threshold = np.percentile(all_rewards, 50)
                if total_reward >= threshold:
                    self.expert_data.extend(episode_data)
            else:
                if total_reward > -200:
                    self.expert_data.extend(episode_data)
        
            if episode % 100 == 0:
                if episode >= 100:
                    threshold = np.percentile(all_rewards, 50)
                    print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}, Threshold: {threshold:.2f}")
                else:
                    print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}")

    def simple_policy(self, state):
        x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
        if left_contact and right_contact:
            return 0
        
        if abs(angle) > 0.5:
            return 2 if angle > 0 else 3
        
        if abs(angular_vel) > 0.5:
            return 2 if angular_vel > 0 else 3
        
        if abs(vx) > 0.5:
            return 2 if vx > 0 else 3
        
        if vy < -0.5:
            return 1
        
        if y > 0.3 and abs(angle) < 0.1 and abs(vx) < 0.1:
            return 1
        
        return 0

    def get_training_data(self):
        if not self.expert_data:
            raise ValueError("Expert 데이터가 수집되지 않았습니다!")
        
        states = [data[0] for data in self.expert_data]
        actions = [data[1] for data in self.expert_data]

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))

        return states, actions


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    
class BehavioralCloning:
    def __init__(self, env, policy_network, learning_rate=0.001):
        self.env = env
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, expert_states, expert_actions, epochs=1000, batch_size=32):
        data_size = len(expert_states)
        if data_size < batch_size:
            print(f"Warning: 데이터 크기({data_size})가 배치 크기({batch_size})보다 작습니다. 배치 크기를 데이터 크기로 조정합니다.")
            batch_size = data_size
        
        losses = []

        for epoch in range(epochs):
            indices = random.sample(range(len(expert_states)), batch_size)
            batch_states = expert_states[indices]
            batch_actions = expert_actions[indices]

            action_probs = self.policy(batch_states)

            loss = self.criterion(action_probs, batch_actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return losses

    def evaluate(self, num_episodes=10):
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor)
                action = torch.argmax(action_probs).item()

                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                state = next_state
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        return avg_reward, total_rewards


class ImprovedExpertDataCollector(ExpertDataCollector):
    def __init__(self, env):
        super().__init__(env)

    def collect_expert_data(self, num_episodes=500):
        successful_episodes = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done, truncated, total_reward = False, False, 0
            episode_data = []

            while not (done or truncated):
                action = self.improved_policy(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_data.append((state, action))
                state = next_state
                total_reward += reward
            
            successful_episodes.append((total_reward, episode_data))
            
            if episode % 100 == 0 and len(successful_episodes) > 10:
                rewards = [r for r, _ in successful_episodes]
                top_30_threshold = np.percentile(rewards, 70)
                print(f"Episode {episode}, Top 30% threshold: {top_30_threshold:.2f}")
        
        rewards = [r for r, _ in successful_episodes]
        threshold = np.percentile(rewards, 70)
        
        for reward, episode_data in successful_episodes:
            if reward >= threshold:
                self.expert_data.extend(episode_data)
        
        print(f"Final expert data: {len(self.expert_data)} samples from episodes with reward >= {threshold:.2f}")

    def improved_policy(self, state):
        x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
        if left_contact and right_contact: return 0
        if y < 0.2 and vy < -0.3: return 1
        if abs(angle) > 0.3: return 2 if angle > 0.05 else 3
        if abs(angular_vel) > 0.3: return 2 if angular_vel > 0.1 else 3
        if abs(vx) > 0.3: return 2 if vx > 0.1 else 3
        if vy < -0.4: return 1
        if y > 0.5 and abs(angle) < 0.1 and abs(vx) < 0.2 and vy < -0.2: return 1
        
        return 0


def main():
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    print("Collecting expert data (v1)...")
    collector = ExpertDataCollector(env)
    collector.collect_expert_data(num_episodes=1000)
    
    try:
        expert_states, expert_actions = collector.get_training_data()
        print(f"Collected {len(expert_states)} expert samples")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
    print("Training BC model (v1)...")
    bc = BehavioralCloning(env, policy, learning_rate=0.0001)
    losses = bc.train(expert_states, expert_actions, epochs=2000, batch_size=64)
    
    print("Evaluating policy (v1)...")
    avg_reward, rewards = bc.evaluate(num_episodes=20)
    print(f"Average reward (v1): {avg_reward:.2f}")
    
    print("\n=== Trying with improved expert policy (v2) ===")
    collector_v2 = ImprovedExpertDataCollector(env)
    collector_v2.collect_expert_data(num_episodes=500)
    
    try:
        expert_states_v2, expert_actions_v2 = collector_v2.get_training_data()
        print(f"Collected {len(expert_states_v2)} improved expert samples")
        
        if len(expert_states_v2) > 1000:
            policy_v2 = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
            bc_v2 = BehavioralCloning(env, policy_v2, learning_rate=0.0001)
            losses_v2 = bc_v2.train(expert_states_v2, expert_actions_v2, epochs=2000)
            
            avg_reward_v2, rewards_v2 = bc_v2.evaluate(num_episodes=20)
            print(f"Improved BC Average reward (v2): {avg_reward_v2:.2f}")
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(losses, label='Original BC')
            if 'losses_v2' in locals(): plt.plot(losses_v2, label='Improved BC')
            plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(rewards, label='Original BC')
            if 'rewards_v2' in locals(): plt.plot(rewards_v2, label='Improved BC')
            plt.title('Evaluation Rewards'); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
            
            plt.subplot(1, 3, 3)
            methods = ['Original BC', 'Improved BC'] if 'avg_reward_v2' in locals() else ['Original BC']
            avg_rewards = [avg_reward, avg_reward_v2] if 'avg_reward_v2' in locals() else [avg_reward]
            plt.bar(methods, avg_rewards)
            plt.title('Average Performance'); plt.ylabel('Average Reward')
            plt.axhline(y=200, color='r', linestyle='--', label='Success threshold')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
    except ValueError as e:
        print(f"Error with improved expert: {e}")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses); plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(rewards); plt.title('Evaluation Rewards'); plt.xlabel('Episode'); plt.ylabel('Reward')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
