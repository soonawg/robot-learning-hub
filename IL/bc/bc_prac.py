# """
# Behavior Cloning 실습
# LunarLander-v3 환경에서 expert 데이터를 이용한 지도학습
# """

# import numpy as np
# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import random
# from collections import deque


# # ===== 2. Expert 데이터 수집 =====
# class ExpertDataCollector:
#     def __init__(self, env):
#         self.env = env
#         self.expert_data = []
    
#     def collect_expert_data(self, num_episodes=500):
#         """
#         Expert 정책으로 데이터 수집
#         """
#         for episode in range(num_episodes):
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0
#             episode_data = []

#             while not (done or truncated):
#                 # 간단한 규칙 기반 정책 사용
#                 action = self.simple_policy(state)

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)

#                 # 데이터 저장
#                 episode_data.append((state, action))

#                 # 상태 업데이트
#                 state = next_state
#                 total_reward += reward
        
#             # 성공적인 에피소드만 유지 (조건을 완화)
#             if total_reward > 100:  # 조건을 200에서 100으로 완화
#                 self.expert_data.extend(episode_data)  # extend 사용하여 모든 step 데이터 추가
        
#             if episode % 100 == 0:
#                 print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}")

#     def simple_policy(self, state):
#         """
#         간단한 규칙 기반 정책 구현
#         """
#         # 높이에 따른 메인 엔진 제어
#         if state[1] > 0.5:  # 높이가 높으면
#             return 1  # 메인 엔진
        
#         # 각도에 따른 회전 제어
#         if state[4] > 0.1:  # 오른쪽으로 기울어지면
#             return 2  # 왼쪽 엔진
#         elif state[4] < -0.1:  # 왼쪽으로 기울어지면
#             return 3  # 오른쪽 엔진
        
#         # 속도에 따른 제어
#         if abs(state[2]) > 0.1:  # x 속도가 너무 크면
#             if state[2] > 0:
#                 return 2  # 왼쪽 엔진
#             else:
#                 return 3  # 오른쪽 엔진
        
#         # 기본값
#         return 0  # 아무것도 안함

#     def get_training_data(self):
#         """
#         수집된 데이터를 학습용 형태로 변환
#         """
#         if not self.expert_data:
#             raise ValueError("No expert data collected!")
        
#         # 데이터 분리
#         states = [data[0] for data in self.expert_data]
#         actions = [data[1] for data in self.expert_data]

#         # numpy 배열로 변환
#         states = np.array(states)
#         actions = np.array(actions)

#         # 텐서로 변환
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)

#         return states, actions

# # ===== 3. 정책 네트워크 =====
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=64):
#         super(PolicyNetwork, self).__init__()
        
#         # 네트워크 레이어 정의
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, state):
#         """
#         순전파 구현
#         """
#         # 순전파 구현
#         x = self.relu(self.fc1(state))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)
    

# # ===== 4. BC 학습기 =====
# class BehaviorCloning:
#     def __init__(self, env, policy_network, learning_rate=0.001):
#         self.env = env
#         self.policy = policy_network
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
#         self.criterion = nn.CrossEntropyLoss()  # discrete action space
        
#     def train(self, expert_states, expert_actions, epochs=1000, batch_size=32):
#         """
#         BC 학습 루프 구현
#         """
#         # 데이터 크기 확인 및 배치 크기 조정
#         data_size = len(expert_states)
#         if data_size < batch_size:
#             print(f"Warning: Data size ({data_size}) is smaller than batch size ({batch_size}). Using data size as batch size.")
#             batch_size = data_size
        
#         losses = []

#         for epoch in range(epochs):
#             # 배치 샘플링
#             indices = random.sample(range(len(expert_states)), batch_size)
#             batch_states = expert_states[indices]
#             batch_actions = expert_actions[indices]

#             # 순전파
#             action_probs = self.policy(batch_states)

#             # 손실 계산
#             loss = self.criterion(action_probs, batch_actions)

#             # 역전파 및 최적화
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # 손실 기록
#             losses.append(loss.item())

#             # 진행 상황 출력
#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
#         return losses

#     def evaluate(self, num_episodes=10):
#         """
#         학습된 정책 평가
#         """
#         total_rewards = []

#         for episode in range(num_episodes):
#             # 에피소드 시작
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0

#             while not (done or truncated):
#                 # 정책에서 action 선택
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 action_probs = self.policy(state_tensor)
#                 action = torch.argmax(action_probs).item()

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 total_reward += reward
#                 state = next_state
            
#             # 보상 기록
#             total_rewards.append(total_reward)
        
#         # 평균 보상 계산
#         avg_reward = np.mean(total_rewards)
#         return avg_reward, total_rewards

# # ==== 5. 메인 실행 함수 ====
# def main():
#     # 환경 설정
#     env = gym.make('LunarLander-v3')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
#     # Expert 데이터 수집
#     print("Collecting expert data...")
#     collector = ExpertDataCollector(env)
#     collector.collect_expert_data(num_episodes=500)
    
#     try:
#         expert_states, expert_actions = collector.get_training_data()
#         print(f"Collected {len(expert_states)} expert samples")
#     except ValueError as e:
#         print(f"Error: {e}")
#         return
    
#     # 정책 네트워크 생성
#     policy = PolicyNetwork(state_dim, action_dim)
    
#     # BC 학습
#     print("Training BC model...")
#     bc = BehaviorCloning(env, policy)
#     losses = bc.train(expert_states, expert_actions)
    
#     # 성능 평가
#     print("Evaluating policy...")
#     avg_reward, rewards = bc.evaluate()
#     print(f"Average reward: {avg_reward:.2f}")
    
#     # 결과 시각화
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(losses)
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(rewards)
#     plt.title('Evaluation Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

# """
# Behavior Cloning 실습
# LunarLander-v3 환경에서 expert 데이터를 이용한 지도학습
# """

# import numpy as np
# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import random
# from collections import deque


# # ===== 2. Expert 데이터 수집 =====
# class ExpertDataCollector:
#     def __init__(self, env):
#         self.env = env
#         self.expert_data = []
    
#     def collect_expert_data(self, num_episodes=500):
#         """
#         Expert 정책으로 데이터 수집
#         """
#         all_rewards = []
        
#         for episode in range(num_episodes):
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0
#             episode_data = []

#             while not (done or truncated):
#                 # 간단한 규칙 기반 정책 사용
#                 action = self.simple_policy(state)

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)

#                 # 데이터 저장
#                 episode_data.append((state, action))

#                 # 상태 업데이트
#                 state = next_state
#                 total_reward += reward
        
#             all_rewards.append(total_reward)
            
#             # 상위 50% 에피소드만 저장 (처음 100개 에피소드 후부터 적용)
#             if episode >= 100:
#                 threshold = np.percentile(all_rewards, 50)  # 상위 50%
#                 if total_reward >= threshold:
#                     self.expert_data.extend(episode_data)
#             else:
#                 # 처음 100개는 -200보다 좋은 것들만 저장
#                 if total_reward > -200:
#                     self.expert_data.extend(episode_data)
        
#             if episode % 100 == 0:
#                 if episode >= 100:
#                     threshold = np.percentile(all_rewards, 50)
#                     print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}, Threshold: {threshold:.2f}")
#                 else:
#                     print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}")

#     def simple_policy(self, state):
#         """
#         개선된 규칙 기반 정책 구현
#         state: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
#         """
#         x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
#         # 착륙 감지 (두 다리가 모두 접촉하면 아무것도 하지 않음)
#         if left_contact and right_contact:
#             return 0
        
#         # 높이가 매우 낮을 때는 메인 엔진 사용
#         if y < 0.1:
#             return 1
        
#         # 각도 제어가 최우선 (안정성을 위해)
#         if abs(angle) > 0.5:  # 각도가 크게 기울어진 경우
#             if angle > 0:
#                 return 2  # 왼쪽 엔진 (반시계방향 회전)
#             else:
#                 return 3  # 오른쪽 엔진 (시계방향 회전)
        
#         # 각속도 제어
#         if abs(angular_vel) > 0.5:
#             if angular_vel > 0:
#                 return 2  # 왼쪽 엔진
#             else:
#                 return 3  # 오른쪽 엔진
        
#         # 수평 속도 제어
#         if abs(vx) > 0.5:
#             if vx > 0:
#                 return 2  # 왼쪽 엔진 (왼쪽으로 밀기)
#             else:
#                 return 3  # 오른쪽 엔진 (오른쪽으로 밀기)
        
#         # 수직 속도 제어 (너무 빠르게 떨어지면 메인 엔진)
#         if vy < -0.5:
#             return 1  # 메인 엔진
        
#         # 높이가 높고 안정적이면 메인 엔진으로 천천히 하강
#         if y > 0.3 and abs(angle) < 0.1 and abs(vx) < 0.1:
#             return 1
        
#         # 기본값: 아무것도 하지 않음
#         return 0

#     def get_training_data(self):
#         """
#         수집된 데이터를 학습용 형태로 변환
#         """
#         if not self.expert_data:
#             raise ValueError("No expert data collected!")
        
#         # 데이터 분리
#         states = [data[0] for data in self.expert_data]
#         actions = [data[1] for data in self.expert_data]

#         # numpy 배열로 변환
#         states = np.array(states)
#         actions = np.array(actions)

#         # 텐서로 변환
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)

#         return states, actions

# # ===== 3. 정책 네트워크 =====
# class PolicyNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=64):
#         super(PolicyNetwork, self).__init__()
        
#         # 네트워크 레이어 정의
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, state):
#         """
#         순전파 구현
#         """
#         # 순전파 구현
#         x = self.relu(self.fc1(state))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)
    

# # ===== 4. BC 학습기 =====
# class BehaviorCloning:
#     def __init__(self, env, policy_network, learning_rate=0.001):
#         self.env = env
#         self.policy = policy_network
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
#         self.criterion = nn.CrossEntropyLoss()  # discrete action space
        
#     def train(self, expert_states, expert_actions, epochs=1000, batch_size=32):
#         """
#         BC 학습 루프 구현
#         """
#         # 데이터 크기 확인 및 배치 크기 조정
#         data_size = len(expert_states)
#         if data_size < batch_size:
#             print(f"Warning: Data size ({data_size}) is smaller than batch size ({batch_size}). Using data size as batch size.")
#             batch_size = data_size
        
#         losses = []

#         for epoch in range(epochs):
#             # 배치 샘플링
#             indices = random.sample(range(len(expert_states)), batch_size)
#             batch_states = expert_states[indices]
#             batch_actions = expert_actions[indices]

#             # 순전파
#             action_probs = self.policy(batch_states)

#             # 손실 계산
#             loss = self.criterion(action_probs, batch_actions)

#             # 역전파 및 최적화
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#             # 손실 기록
#             losses.append(loss.item())

#             # 진행 상황 출력
#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
#         return losses

#     def evaluate(self, num_episodes=10):
#         """
#         학습된 정책 평가
#         """
#         total_rewards = []

#         for episode in range(num_episodes):
#             # 에피소드 시작
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0

#             while not (done or truncated):
#                 # 정책에서 action 선택
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 action_probs = self.policy(state_tensor)
#                 action = torch.argmax(action_probs).item()

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 total_reward += reward
#                 state = next_state
            
#             # 보상 기록
#             total_rewards.append(total_reward)
        
#         # 평균 보상 계산
#         avg_reward = np.mean(total_rewards)
#         return avg_reward, total_rewards

# # ==== 5. 메인 실행 함수 ====
# def main():
#     # 환경 설정
#     env = gym.make('LunarLander-v3')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
#     # Expert 데이터 수집
#     print("Collecting expert data...")
#     collector = ExpertDataCollector(env)
#     collector.collect_expert_data(num_episodes=500)
    
#     try:
#         expert_states, expert_actions = collector.get_training_data()
#         print(f"Collected {len(expert_states)} expert samples")
#     except ValueError as e:
#         print(f"Error: {e}")
#         return
    
#     # 정책 네트워크 생성
#     policy = PolicyNetwork(state_dim, action_dim)
    
#     # BC 학습
#     print("Training BC model...")
#     bc = BehaviorCloning(env, policy)
#     losses = bc.train(expert_states, expert_actions)
    
#     # 성능 평가
#     print("Evaluating policy...")
#     avg_reward, rewards = bc.evaluate()
#     print(f"Average reward: {avg_reward:.2f}")
    
#     # 결과 시각화
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(losses)
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(rewards)
#     plt.title('Evaluation Rewards')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()

"""
Behavior Cloning 실습
LunarLander-v3 환경에서 expert 데이터를 이용한 지도학습
"""

# import numpy as np
# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import random
# from collections import deque


# # ===== 2. Expert 데이터 수집 =====
# class ExpertDataCollector:
#     """
#     Expert 정책을 사용해 (state, action) 쌍을 수집하는 클래스
#     - '간단한 규칙 기반 정책'을 expert로 사용합니다.
#     """
#     def __init__(self, env):
#         self.env = env
#         self.expert_data = [] # (state, action) 튜플을 저장할 리스트
    
#     def collect_expert_data(self, num_episodes=500):
#         """
#         Expert 정책으로 데이터 

#         지정된 에피소드 수만큼 데이터를 수집합니다.
#         - 초기에는 안정적으로 데이터를 쌓고, 이후에는 좋은 성능의 에피소드만 선별합니다.
#         """
#         all_rewards = []  # 모든 에피소드의 보상을 기록 
        
#         for episode in range(num_episodes):
#             state, _ = self.env.reset()  # 환경 초기화
#             done = False
#             truncated = False
#             total_reward = 0
#             episode_data = []  # 현재 에피소드의 데이터를 임시 저장

#             while not (done or truncated):
#                 # 간단한 규칙 기반 expert 정책 사용
#                 action = self.simple_policy(state)

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)

#                 # (state, action) 쌍 데이터 저장
#                 episode_data.append((state, action))

#                 # 상태 업데이트 및 보상 누적
#                 state = next_state
#                 total_reward += reward
        
#             all_rewards.append(total_reward)
            
#             # 상위 50% 에피소드만 저장 (처음 100개 에피소드 후부터 적용)
#             if episode >= 100:
#                 threshold = np.percentile(all_rewards, 50)  # 상위 50%
#                 if total_reward >= threshold:
#                     self.expert_data.extend(episode_data)
#             else:
#                 # 초기 100개 에피소드에서는 너무 나쁜 데이터만 아니면 저장
#                 if total_reward > -200:
#                     self.expert_data.extend(episode_data)
        
#             if episode % 100 == 0:
#                 # 데이터 수집 진행 상황 출력
#                 if episode >= 100:
#                     threshold = np.percentile(all_rewards, 50)
#                     print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}, Threshold: {threshold:.2f}")
#                 else:
#                     print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}")

#     def simple_policy(self, state):
#         """
#         개선된 규칙 기반 정책 구현
#         state: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
#         """
#         x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
#         # 착륙 감지 (두 다리가 모두 접촉하면 아무것도 하지 않음)
#         if left_contact and right_contact:
#             return 0
        
#         # 높이가 매우 낮을 때는 메인 엔진 사용
#         if y < 0.1:
#             return 1
        
#         # 각도 제어가 최우선 (안정성을 위해)
#         if abs(angle) > 0.5:  # 각도가 크게 기울어진 경우
#             if angle > 0:
#                 return 2  # 왼쪽 엔진 (반시계방향 회전)
#             else:
#                 return 3  # 오른쪽 엔진 (시계방향 회전)
        
#         # 각속도 제어
#         if abs(angular_vel) > 0.5:
#             if angular_vel > 0:
#                 return 2  # 왼쪽 엔진
#             else:
#                 return 3  # 오른쪽 엔진
        
#         # 수평 속도 제어
#         if abs(vx) > 0.5:
#             if vx > 0:
#                 return 2  # 왼쪽 엔진 (왼쪽으로 밀기)
#             else:
#                 return 3  # 오른쪽 엔진 (오른쪽으로 밀기)
        
#         # 수직 속도 제어 (너무 빠르게 떨어지면 메인 엔진)
#         if vy < -0.5:
#             return 1  # 메인 엔진
        
#         # 높이가 높고 안정적이면 메인 엔진으로 천천히 하강
#         if y > 0.3 and abs(angle) < 0.1 and abs(vx) < 0.1:
#             return 1
        
#         # 기본값: 아무것도 하지 않음
#         return 0

#     def get_training_data(self):
#         """
#         수집된 데이터를 PyTorch 학습용 형태로 변환
#         """
#         if not self.expert_data:
#             raise ValueError("Expert 데이터가 수집되지 않았습니다!")
        
#         # 데이터 분리
#         states = [data[0] for data in self.expert_data]
#         actions = [data[1] for data in self.expert_data]

#         # numpy 배열로 변환
#         states = np.array(states)
#         actions = np.array(actions)

#         # 텐서로 변환
#         states = torch.FloatTensor(states)
#         actions = torch.LongTensor(actions)

#         return states, actions

# # ===== 3. 정책 네트워크 =====
# class PolicyNetwork(nn.Module):
#     """
#     State를 입력받아 각 Action에 대한 확률을 출력하는 신경망
#     """
#     def __init__(self, state_dim, action_dim, hidden_dim=64):
#         super(PolicyNetwork, self).__init__()
        
#         # 네트워크 레이어 정의 - 3개의 Fully Connected Layer로 구성
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1) # 출력: 확률 분포
    
#     def forward(self, state):
#         """
#         순전파 구현
#         """
#         # 순전파 구현
#         x = self.relu(self.fc1(state))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)
    

# # ===== 4. BC 학습기 =====
# class BehaviorCloning:
#     """ BC 학습 및 평가를 담당하는 클래스 """
#     def __init__(self, env, policy_network, learning_rate=0.001):
#         self.env = env
#         self.policy = policy_network
#         # Adam 옵티마이저: 파라미터 업데이트
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
#         # CrossEntropyLoss: 분류 문제(discrete action)의 손실 함수
#         self.criterion = nn.CrossEntropyLoss()
        
#     def train(self, expert_states, expert_actions, epochs=1000, batch_size=32):
#         """
#         BC 학습 루프 구현 - Expert 데이터로 정책 네트워크를 학습시킨다.
#         """
#         # 데이터 크기 확인 및 배치 크기 조정
#         data_size = len(expert_states)
#         if data_size < batch_size:
#             print(f"Warning: Data size ({data_size}) is smaller than batch size ({batch_size}). Using data size as batch size.")
#             batch_size = data_size
        
#         losses = []  # 학습 손실 기록

#         for epoch in range(epochs):
#             # 미니배치(mini-batch) 무작위 샘플링
#             indices = random.sample(range(len(expert_states)), batch_size)
#             batch_states = expert_states[indices]
#             batch_actions = expert_actions[indices]

#             # 순전파: 정책 네트워크로 action 확률 예측
#             action_probs = self.policy(batch_states)

#             # 손실 계산: 예측과 expert의 실제 action 비교
#             loss = self.criterion(action_probs, batch_actions)

#             # 역전파 및 최적화
#             self.optimizer.zero_grad()  # 그래디언트 초기화
#             loss.backward()             # 그래디언트 계산
#             self.optimizer.step()       # 파라미터 업데이트

#             # 손실 기록
#             losses.append(loss.item())

#             # 진행 상황 출력
#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
#         return losses

#     def evaluate(self, num_episodes=10):
#         """
#         학습된 정책 평가
#         """
#         total_rewards = []

#         for episode in range(num_episodes):
#             # 에피소드 시작
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0

#             while not (done or truncated):
#                 # 정책에서 action 선택
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 action_probs = self.policy(state_tensor)
#                 action = torch.argmax(action_probs).item()

#                 # 환경에서 한 스텝 진행
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 total_reward += reward
#                 state = next_state
            
#             # 보상 기록
#             total_rewards.append(total_reward)
        
#         # 평균 보상 계산
#         avg_reward = np.mean(total_rewards)
#         return avg_reward, total_rewards

# def main():
#     # 환경 설정
#     env = gym.make('LunarLander-v3')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
#     # Expert 데이터 수집
#     print("Collecting expert data...")
#     collector = ExpertDataCollector(env)
#     collector.collect_expert_data(num_episodes=1000)  # 더 많은 에피소드
    
#     try:
#         expert_states, expert_actions = collector.get_training_data()
#         print(f"Collected {len(expert_states)} expert samples")
#     except ValueError as e:
#         print(f"Error: {e}")
#         return
    
#     # 정책 네트워크 생성 (더 큰 네트워크)
#     policy = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
    
#     # BC 학습 (더 긴 학습)
#     print("Training BC model...")
#     bc = BehaviorCloning(env, policy, learning_rate=0.0001)  # 더 작은 학습률
#     losses = bc.train(expert_states, expert_actions, epochs=2000, batch_size=64)
    
#     # 성능 평가
#     print("Evaluating policy...")
#     avg_reward, rewards = bc.evaluate(num_episodes=20)
#     print(f"Average reward: {avg_reward:.2f}")
    
#     # 추가: 더 나은 expert 정책으로 재학습
#     print("\n=== Trying with better expert policy ===")
#     collector_v2 = ImprovedExpertDataCollector(env)
#     collector_v2.collect_expert_data(num_episodes=500)
    
#     try:
#         expert_states_v2, expert_actions_v2 = collector_v2.get_training_data()
#         print(f"Collected {len(expert_states_v2)} improved expert samples")
        
#         if len(expert_states_v2) > 1000:  # 충분한 데이터가 있는 경우만
#             policy_v2 = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
#             bc_v2 = BehaviorCloning(env, policy_v2, learning_rate=0.0001)
#             losses_v2 = bc_v2.train(expert_states_v2, expert_actions_v2, epochs=2000)
            
#             avg_reward_v2, rewards_v2 = bc_v2.evaluate(num_episodes=20)
#             print(f"Improved BC Average reward: {avg_reward_v2:.2f}")
            
#             # 결과 비교 시각화
#             plt.figure(figsize=(15, 5))
#             plt.subplot(1, 3, 1)
#             plt.plot(losses, label='Original BC')
#             if 'losses_v2' in locals():
#                 plt.plot(losses_v2, label='Improved BC')
#             plt.title('Training Loss')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.legend()
            
#             plt.subplot(1, 3, 2)
#             plt.plot(rewards, label='Original BC')
#             if 'rewards_v2' in locals():
#                 plt.plot(rewards_v2, label='Improved BC')
#             plt.title('Evaluation Rewards')
#             plt.xlabel('Episode')
#             plt.ylabel('Reward')
#             plt.legend()
            
#             plt.subplot(1, 3, 3)
#             methods = ['Original BC', 'Improved BC'] if 'avg_reward_v2' in locals() else ['Original BC']
#             avg_rewards = [avg_reward, avg_reward_v2] if 'avg_reward_v2' in locals() else [avg_reward]
#             plt.bar(methods, avg_rewards)
#             plt.title('Average Performance')
#             plt.ylabel('Average Reward')
#             plt.axhline(y=200, color='r', linestyle='--', label='Success threshold')
#             plt.legend()
            
#             plt.tight_layout()
#             plt.show()
        
#     except ValueError as e:
#         print(f"Error with improved expert: {e}")
        
#         # 기본 결과만 시각화
#         plt.figure(figsize=(12, 4))
#         plt.subplot(1, 2, 1)
#         plt.plot(losses)
#         plt.title('Training Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
        
#         plt.subplot(1, 2, 2)
#         plt.plot(rewards)
#         plt.title('Evaluation Rewards')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         plt.tight_layout()
#         plt.show()


# # ===== 개선된 Expert 정책 =====
# class ImprovedExpertDataCollector:
#     def __init__(self, env):
#         self.env = env
#         self.expert_data = []
    
#     def collect_expert_data(self, num_episodes=500):
#         """더 나은 전략으로 데이터 수집"""
#         successful_episodes = []
        
#         for episode in range(num_episodes):
#             state, _ = self.env.reset()
#             done = False
#             truncated = False
#             total_reward = 0
#             episode_data = []

#             while not (done or truncated):
#                 action = self.improved_policy(state)
#                 next_state, reward, done, truncated, info = self.env.step(action)
#                 episode_data.append((state, action))
#                 state = next_state
#                 total_reward += reward
            
#             successful_episodes.append((total_reward, episode_data))
            
#             if episode % 100 == 0:
#                 # 현재까지의 상위 30% 성능 확인
#                 rewards = [r for r, _ in successful_episodes]
#                 if len(rewards) > 10:
#                     top_30_threshold = np.percentile(rewards, 70)
#                     top_30_count = sum(1 for r in rewards if r >= top_30_threshold)
#                     print(f"Episode {episode}, Top 30% threshold: {top_30_threshold:.2f}, Count: {top_30_count}")
        
#         # 상위 30%의 에피소드만 사용
#         rewards = [r for r, _ in successful_episodes]
#         threshold = np.percentile(rewards, 70)
        
#         for reward, episode_data in successful_episodes:
#             if reward >= threshold:
#                 self.expert_data.extend(episode_data)
        
#         print(f"Final expert data: {len(self.expert_data)} samples from episodes with reward >= {threshold:.2f}")
    
#     def improved_policy(self, state):
#         """더 보수적이고 안정적인 정책"""
#         x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
#         # 성공적인 착륙 조건
#         if left_contact and right_contact and abs(vx) < 0.1 and abs(vy) < 0.1:
#             return 0
        
#         # 위험한 상황에서는 메인 엔진 우선
#         if y < 0.2 and vy < -0.3:
#             return 1
        
#         # 각도가 너무 기울어진 경우 즉시 보정
#         if abs(angle) > 0.3:
#             if angle > 0.05:
#                 return 2
#             elif angle < -0.05:
#                 return 3
        
#         # 각속도 제어
#         if abs(angular_vel) > 0.3:
#             if angular_vel > 0.1:
#                 return 2
#             elif angular_vel < -0.1:
#                 return 3
        
#         # 수평 속도 제어 (더 보수적으로)
#         if abs(vx) > 0.3:
#             if vx > 0.1:
#                 return 2
#             elif vx < -0.1:
#                 return 3
        
#         # 수직 속도가 너무 빠르면 메인 엔진
#         if vy < -0.4:
#             return 1
        
#         # 안정적인 하강
#         if y > 0.5 and abs(angle) < 0.1 and abs(vx) < 0.2:
#             if vy < -0.2:  # 너무 빠르게 떨어지지 않도록
#                 return 1
        
#         return 0
    
#     def get_training_data(self):
#         if not self.expert_data:
#             raise ValueError("No expert data collected!")
        
#         states = [data[0] for data in self.expert_data]
#         actions = [data[1] for data in self.expert_data]
        
#         states = torch.FloatTensor(np.array(states))
#         actions = torch.LongTensor(np.array(actions))
        
#         return states, actions

# if __name__ == "__main__":
#     main()

# State dim: 8, Action dim: 4
# Collecting expert data...
# Episode 0, Collected 0 samples, Last reward: -342.17
# Episode 100, Collected 612 samples, Last reward: -513.04, Threshold: -523.66
# Episode 200, Collected 4601 samples, Last reward: -387.04, Threshold: -526.82
# Episode 300, Collected 8596 samples, Last reward: -465.37, Threshold: -526.82
# Episode 400, Collected 12339 samples, Last reward: -601.86, Threshold: -529.32
# Episode 500, Collected 16183 samples, Last reward: -515.00, Threshold: -530.96
# Episode 600, Collected 19926 samples, Last reward: -414.75, Threshold: -532.41
# Episode 700, Collected 23878 samples, Last reward: -484.77, Threshold: -532.26
# Episode 800, Collected 28819 samples, Last reward: -619.94, Threshold: -529.22
# Episode 900, Collected 32574 samples, Last reward: -610.41, Threshold: -530.49
# Collected 36966 expert samples
# Training BC model...
# Epoch 0, Loss: 1.3946
# Epoch 100, Loss: 1.3109
# Epoch 200, Loss: 1.1822
# Epoch 300, Loss: 1.1068
# Epoch 400, Loss: 1.1420
# Epoch 500, Loss: 1.0241
# Epoch 600, Loss: 1.0136
# Epoch 700, Loss: 1.0656
# Epoch 800, Loss: 1.0486
# Epoch 900, Loss: 1.0436
# Epoch 1000, Loss: 1.0615
# Epoch 1100, Loss: 1.0147
# Epoch 1200, Loss: 0.9503
# Epoch 1300, Loss: 0.9569
# Epoch 1400, Loss: 0.9744
# Epoch 1500, Loss: 0.9478
# Epoch 1600, Loss: 0.8676
# Epoch 1700, Loss: 0.9422
# Epoch 1800, Loss: 0.9647
# Epoch 1900, Loss: 0.9010
# Evaluating policy...
# Average reward: -519.91

# === Trying with better expert policy ===
# Episode 100, Top 30% threshold: -537.57, Count: 31
# Episode 200, Top 30% threshold: -526.47, Count: 61
# Episode 300, Top 30% threshold: -511.09, Count: 91
# Episode 400, Top 30% threshold: -513.36, Count: 121
# Final expert data: 11000 samples from episodes with reward >= -511.22
# Collected 11000 improved expert samples
# Epoch 0, Loss: 1.4040
# Epoch 100, Loss: 1.2929
# Epoch 200, Loss: 1.0744
# Epoch 300, Loss: 1.1471
# Epoch 400, Loss: 1.1549
# Epoch 500, Loss: 1.0149
# Epoch 600, Loss: 0.9451
# Epoch 700, Loss: 0.9860
# Epoch 800, Loss: 1.0752
# Epoch 900, Loss: 0.9637
# Epoch 1000, Loss: 1.1144
# Epoch 1100, Loss: 0.9350
# Epoch 1200, Loss: 1.0888
# Epoch 1300, Loss: 0.9880
# Epoch 1400, Loss: 0.8697
# Epoch 1500, Loss: 0.9100
# Epoch 1600, Loss: 1.0907
# Epoch 1700, Loss: 1.0692
# Epoch 1800, Loss: 1.0173
# Epoch 1900, Loss: 0.9845
# Improved BC Average reward: -867.37

"""
Behavior Cloning (BC) 실습
- LunarLander-v3 환경에서 expert 데이터를 이용해 정책을 학습합니다.
- BC는 expert의 행동을 모방하는 지도학습 기법입니다.
"""

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from collections import deque

# ===== Expert 데이터 수집기 클래스 =====
class ExpertDataCollector:
    """
    Expert 정책을 사용해 (state, action) 쌍을 수집하는 클래스
    - '간단한 규칙 기반 정책'을 expert로 사용합니다.
    """
    def __init__(self, env):
        self.env = env
        self.expert_data = []  # (state, action) 튜플을 저장할 리스트
    
    def collect_expert_data(self, num_episodes=500):
        """
        지정된 에피소드 수만큼 데이터를 수집합니다.
        - 초기에는 안정적으로 데이터를 쌓고, 이후에는 좋은 성능의 에피소드만 선별합니다.
        """
        all_rewards = []  # 모든 에피소드의 보상을 기록
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # 환경 초기화
            done = False
            truncated = False
            total_reward = 0
            episode_data = []  # 현재 에피소드의 데이터를 임시 저장

            while not (done or truncated):
                # 규칙 기반 expert 정책으로 행동 결정
                action = self.simple_policy(state)

                # 환경에서 한 스텝 진행
                next_state, reward, done, truncated, info = self.env.step(action)

                # (state, action) 쌍 저장
                episode_data.append((state, action))

                # 상태 업데이트 및 보상 누적
                state = next_state
                total_reward += reward
        
            all_rewards.append(total_reward)
            
            # 상위 50% 에피소드만 저장 (데이터 다양성 확보 후 적용)
            if episode >= 100:
                threshold = np.percentile(all_rewards, 50)  # 보상 기준 (중앙값)
                if total_reward >= threshold:
                    self.expert_data.extend(episode_data)
            else:
                # 초기 100개 에피소드에서는 너무 나쁜 데이터만 아니면 저장
                if total_reward > -200:
                    self.expert_data.extend(episode_data)
        
            if episode % 100 == 0:
                # 데이터 수집 진행 상황 출력
                if episode >= 100:
                    threshold = np.percentile(all_rewards, 50)
                    print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}, Threshold: {threshold:.2f}")
                else:
                    print(f"Episode {episode}, Collected {len(self.expert_data)} samples, Last reward: {total_reward:.2f}")

    def simple_policy(self, state):
        """
        간단한 규칙 기반의 expert 정책
        - state: [x, y, vx, vy, angle, angular_velocity, leg1, leg2]
        """
        x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
        # 1. 착륙 성공 시: 아무것도 안 함
        if left_contact and right_contact:
            return 0
        
        # 2. 각도 제어 (안정성 최우선)
        if abs(angle) > 0.5:
            return 2 if angle > 0 else 3
        
        # 3. 각속도 제어
        if abs(angular_vel) > 0.5:
            return 2 if angular_vel > 0 else 3
        
        # 4. 수평 속도 제어
        if abs(vx) > 0.5:
            return 2 if vx > 0 else 3
        
        # 5. 수직 속도 제어 (너무 빠르게 떨어지면 엔진 켜기)
        if vy < -0.5:
            return 1
        
        # 6. 안정적 하강
        if y > 0.3 and abs(angle) < 0.1 and abs(vx) < 0.1:
            return 1
        
        # 기본: 아무것도 안 함
        return 0

    def get_training_data(self):
        """
        수집된 데이터를 PyTorch 학습에 사용할 수 있는 텐서로 변환합니다.
        """
        if not self.expert_data:
            raise ValueError("Expert 데이터가 수집되지 않았습니다!")
        
        states = [data[0] for data in self.expert_data]
        actions = [data[1] for data in self.expert_data]

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)) # CrossEntropyLoss는 LongTensor를 요구

        return states, actions

# ===== 정책 네트워크 (Policy Network) =====
class PolicyNetwork(nn.Module):
    """
    State를 입력받아 각 Action에 대한 확률을 출력하는 신경망
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        # 3개의 Fully Connected Layer로 구성
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1) # 출력: 확률 분포
    
    def forward(self, state):
        """ 순전파 로직 """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
# ===== Behavioral Cloning 학습기 =====
class BehavioralCloning:
    """ BC 학습 및 평가를 담당하는 클래스 """
    def __init__(self, env, policy_network, learning_rate=0.001):
        self.env = env
        self.policy = policy_network
        # Adam 옵티마이저: 파라미터 업데이트
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # CrossEntropyLoss: 분류 문제(discrete action)의 손실 함수
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, expert_states, expert_actions, epochs=1000, batch_size=32):
        """
        Expert 데이터로 정책 네트워크를 학습시킵니다.
        """
        data_size = len(expert_states)
        # 데이터가 배치 크기보다 작을 경우 예외 처리
        if data_size < batch_size:
            print(f"Warning: 데이터 크기({data_size})가 배치 크기({batch_size})보다 작습니다. 배치 크기를 데이터 크기로 조정합니다.")
            batch_size = data_size
        
        losses = []  # 학습 손실 기록

        for epoch in range(epochs):
            # 미니배치(mini-batch) 무작위 샘플링
            indices = random.sample(range(len(expert_states)), batch_size)
            batch_states = expert_states[indices]
            batch_actions = expert_actions[indices]

            # 1. 순전파: 정책 네트워크로 action 확률 예측
            action_probs = self.policy(batch_states)

            # 2. 손실 계산: 예측과 expert의 실제 action 비교
            loss = self.criterion(action_probs, batch_actions)

            # 3. 역전파 및 최적화
            self.optimizer.zero_grad()  # 그래디언트 초기화
            loss.backward()             # 그래디언트 계산
            self.optimizer.step()       # 파라미터 업데이트

            losses.append(loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return losses

    def evaluate(self, num_episodes=10):
        """
        학습된 정책의 성능을 평가합니다.
        """
        total_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                # 학습된 정책으로 행동 결정
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy(state_tensor)
                action = torch.argmax(action_probs).item() # 가장 확률이 높은 action 선택

                # 환경에서 한 스텝 진행
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                state = next_state
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        return avg_reward, total_rewards

# ===== 개선된 Expert 정책 데이터 수집기 =====
class ImprovedExpertDataCollector(ExpertDataCollector):
    """
    기존 ExpertDataCollector를 상속받아, 더 정교한 정책과 데이터 수집 전략을 사용합니다.
    """
    def __init__(self, env):
        super().__init__(env)

    def collect_expert_data(self, num_episodes=500):
        """
        더 나은 전략으로 데이터를 수집합니다.
        - 모든 에피소드를 우선 수집한 후, 보상 기준으로 상위 30% 에피소드만 선별합니다.
        - 이는 더 품질 좋은 expert 데이터를 얻기 위함입니다.
        """
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
        
        # 전체 에피소드 중 보상 상위 30% 데이터만 사용
        rewards = [r for r, _ in successful_episodes]
        threshold = np.percentile(rewards, 70)
        
        for reward, episode_data in successful_episodes:
            if reward >= threshold:
                self.expert_data.extend(episode_data)
        
        print(f"Final expert data: {len(self.expert_data)} samples from episodes with reward >= {threshold:.2f}")

    def improved_policy(self, state):
        """
        더 보수적이고 안정적인 expert 정책
        - 각 조건의 임계값을 더 세밀하게 조정하여 안정성을 높입니다.
        """
        x, y, vx, vy, angle, angular_vel, left_contact, right_contact = state
        
        if left_contact and right_contact: return 0
        if y < 0.2 and vy < -0.3: return 1
        if abs(angle) > 0.3: return 2 if angle > 0.05 else 3
        if abs(angular_vel) > 0.3: return 2 if angular_vel > 0.1 else 3
        if abs(vx) > 0.3: return 2 if vx > 0.1 else 3
        if vy < -0.4: return 1
        if y > 0.5 and abs(angle) < 0.1 and abs(vx) < 0.2 and vy < -0.2: return 1
        
        return 0

# ===== 메인 실행 함수 =====
def main():
    # 1. 환경 설정
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # 2. 첫 번째 Expert 데이터 수집 및 학습
    print("Collecting expert data (v1)...")
    collector = ExpertDataCollector(env)
    collector.collect_expert_data(num_episodes=1000)
    
    try:
        expert_states, expert_actions = collector.get_training_data()
        print(f"Collected {len(expert_states)} expert samples")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # 3. BC 모델 생성 및 학습 (v1)
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
    print("Training BC model (v1)...")
    bc = BehavioralCloning(env, policy, learning_rate=0.0001)
    losses = bc.train(expert_states, expert_actions, epochs=2000, batch_size=64)
    
    # 4. 성능 평가 (v1)
    print("Evaluating policy (v1)...")
    avg_reward, rewards = bc.evaluate(num_episodes=20)
    print(f"Average reward (v1): {avg_reward:.2f}")
    
    # 5. 개선된 Expert 데이터로 재학습 (v2)
    print("\n=== Trying with improved expert policy (v2) ===")
    collector_v2 = ImprovedExpertDataCollector(env)
    collector_v2.collect_expert_data(num_episodes=500)
    
    try:
        expert_states_v2, expert_actions_v2 = collector_v2.get_training_data()
        print(f"Collected {len(expert_states_v2)} improved expert samples")
        
        # 충분한 데이터가 수집된 경우에만 학습 진행
        if len(expert_states_v2) > 1000:
            policy_v2 = PolicyNetwork(state_dim, action_dim, hidden_dim=128)
            bc_v2 = BehavioralCloning(env, policy_v2, learning_rate=0.0001)
            losses_v2 = bc_v2.train(expert_states_v2, expert_actions_v2, epochs=2000)
            
            avg_reward_v2, rewards_v2 = bc_v2.evaluate(num_episodes=20)
            print(f"Improved BC Average reward (v2): {avg_reward_v2:.2f}")
            
            # 6. 결과 비교 시각화
            plt.figure(figsize=(15, 5))
            # 손실 곡선
            plt.subplot(1, 3, 1)
            plt.plot(losses, label='Original BC')
            if 'losses_v2' in locals(): plt.plot(losses_v2, label='Improved BC')
            plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            
            # 평가 보상
            plt.subplot(1, 3, 2)
            plt.plot(rewards, label='Original BC')
            if 'rewards_v2' in locals(): plt.plot(rewards_v2, label='Improved BC')
            plt.title('Evaluation Rewards'); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.legend()
            
            # 평균 성능 비교
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
        
        # v1 결과만 시각화
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses); plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(rewards); plt.title('Evaluation Rewards'); plt.xlabel('Episode'); plt.ylabel('Reward')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()