import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import pickle
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probs = torch.softmax(logits, dim=1)
        
        if deterministic:
            return torch.argmax(probs, dim=1).item()
        else:
            action = torch.multinomial(probs, 1).item()
        
        return action


class ExpertPolicy:
    def __init__(self, env_name):
        self.env_name = env_name
    
    def get_action(self, state):
        if "CartPole" in self.env_name:
            return self._cartpole_expert(state)
        elif "LunarLander" in self.env_name:
            return self._lunar_lander_expert(state)
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
    
    def _cartpole_expert(self, state):
        angle = state[2]
        angle_velocity = state[3]

        if angle > 0.1 or (angle > -0.1 and angle_velocity > 0):
            return 1
        else:
            return 0
    
    def _lunar_lander_expert(self, state):
        x_pos, y_pos = state[0], state[1]
        x_vel, y_vel = state[2], state[3]
        angle = state[4]

        if y_pos < 0.5:
            if abs(x_vel) > 0.1:
                return 1 if x_vel < 0 else 2
            elif y_vel < -0.1:
                return 0
            else:
                return 3
        
        if abs(angle) > 0.1:
            return 1 if angle > 0 else 2
        elif y_vel < -0.5:
            return 0
        else:
            return 3

class DataCollector:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.expert_policy = ExpertPolicy(env_name)
        self.data = []
    
    def collect_expert_rollouts(self, num_episodes):
        data = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_data = []

            while True:
                expert_action = self.expert_policy.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(expert_action)
                done = terminated or truncated

                episode_data.append({
                    'state': state.copy(),
                    'action': expert_action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })

                state = next_state
                if done:
                    break
            
            data.extend(episode_data)
        
        return data
    
    def collect_dagger_rollouts(self, student_policy, num_episodes):
        data = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_data = []
            
            while True:
                student_action = student_policy.get_action(state)
                expert_action = self.expert_policy.get_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(student_action)
                done = terminated or truncated
                
                episode_data.append({
                    'state': state.copy(),
                    'action': expert_action,
                    'student_action': student_action,
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
                
                state = next_state
                if done:
                    break
            
            data.extend(episode_data)
        
        return data


class BC:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, states, actions):
        self.optimizer.zero_grad()
        
        logits = self.policy(states)
        loss = self.criterion(logits, actions)

        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, dataset, epochs=100, batch_size=32):
        states = torch.FloatTensor([d['state'] for d in dataset])
        actions = torch.LongTensor([d['action'] for d in dataset])

        n_samples = len(dataset)
        losses = []

        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            total_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]

                loss = self.train_step(batch_states, batch_actions)
                total_loss += loss
            
            avg_loss = total_loss / (n_samples // batch_size)
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses

    def evaluate(self, env, num_episodes=10):
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            while True:
                action = self.policy.get_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'rewards': total_rewards
        }

class DAgger:
    def __init__(self, bc_trainer, data_collector):
        self.bc = bc_trainer
        self.data_collector = data_collector
        self.dataset = []
        self.results = []
    
    def dagger_iteration(self, iteration, num_episodes=100):
        print(f"DAgger Iteration {iteration}")
        
        new_data = self.data_collector.collect_dagger_rollouts(
            self.bc.policy, num_episodes
        )
        
        self.dataset.extend(new_data)
        
        losses = self.bc.train(self.dataset, epochs=50, batch_size=32)
        
        eval_results = self.bc.evaluate(self.data_collector.env, num_episodes=10)
        
        result = {
            'iteration': iteration,
            'dataset_size': len(self.dataset),
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'losses': losses
        }
        
        self.results.append(result)
        print(f"Iteration {iteration}: Mean Reward = {eval_results['mean_reward']:.2f}")
        
        return result
    
    def run_dagger(self, num_iterations=10):
        for iteration in range(num_iterations):
            self.dagger_iteration(iteration)
        
        return self.results

class Evaluator:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.expert_policy = ExpertPolicy(env_name)
    
    def evaluate_policy(self, policy, num_episodes=10):
        total_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                action = policy.get_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            
            if "CartPole" in self.env.spec.id:
                if episode_reward >= 195:
                    success_count += 1
            elif "LunarLander" in self.env.spec.id:
                if episode_reward >= 200:
                    success_count += 1
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': success_count / num_episodes,
            'rewards': total_rewards
        }
    
    def compare_policies(self, expert_policy, student_policy, num_episodes=10):
        expert_results = self.evaluate_policy(expert_policy, num_episodes)
        student_results = self.evaluate_policy(student_policy, num_episodes)
        
        return {
            'expert': expert_results,
            'student': student_results,
            'performance_gap': expert_results['mean_reward'] - student_results['mean_reward']
        }
    
def plot_results(results):
    iterations = [r['iteration'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    dataset_sizes = [r['dataset_size'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(iterations, mean_rewards, 'b-o')
    ax1.set_xlabel('DAgger Iteration')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Performance over DAgger Iterations')
    ax1.grid(True)
    
    ax2.plot(iterations, dataset_sizes, 'r-o')
    ax2.set_xlabel('DAgger Iteration')
    ax2.set_ylabel('Dataset Size')
    ax2.set_title('Dataset Growth')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    env_name = "CartPole-v1"
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    collector = DataCollector(env_name)
    
    print("Collecting expert data...")
    expert_data = collector.collect_expert_rollouts(500)
    print(f"Collected {len(expert_data)} expert transitions")
    
    bc_trainer = BC(state_dim, action_dim)
    
    print("Training initial BC policy...")
    bc_losses = bc_trainer.train(expert_data, epochs=100, batch_size=32)
    
    evaluator = Evaluator(env_name)
    initial_eval = bc_trainer.evaluate(collector.env, num_episodes=10)
    print(f"Initial BC performance: {initial_eval['mean_reward']:.2f}")
    
    dagger = DAgger(bc_trainer, collector)
    dagger.dataset = expert_data
    
    print("Running DAgger...")
    dagger_results = dagger.run_dagger(num_iterations=10)
    
    final_eval = bc_trainer.evaluate(collector.env, num_episodes=10)
    print(f"Final performance: {final_eval['mean_reward']:.2f}")
    
    plot_results(dagger_results)
    
    return dagger_results

if __name__ == "__main__":
    results = main()
