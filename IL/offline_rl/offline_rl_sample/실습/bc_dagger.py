# BC (Behavioral Cloning) + DAgger (Dataset Aggregation) 통합 실습
# 
# 목적: 전문가 데이터로 초기 학습 후, DAgger를 통해 정책 개선
# 환경: CartPole-v1, LunarLander-v2
# 
# 전체 구조:
# ├── 1. 라이브러리 임포트 및 설정
# ├── 2. 정책 네트워크 클래스들
# │   ├── PolicyNetwork (학습 대상)
# │   └── ExpertPolicy (전문가 정책)
# ├── 3. 데이터 관리 클래스
# │   ├── DataCollector (데이터 수집)
# │   └── Dataset (데이터 저장/로드)
# ├── 4. 학습 알고리즘 클래스들
# │   ├── BC (Behavioral Cloning)
# │   └── DAgger (Dataset Aggregation)
# ├── 5. 평가 및 유틸리티 클래스
# │   ├── Evaluator (성능 평가)
# │   └── Logger (로그 기록)
# └── 6. 메인 실행 함수
#     └── main() (전체 파이프라인)

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
    """
    학습할 정책 네트워크 (학생 정책)
    
    특징:
    - 3층 신경망 (입력층 -> 히든층1 -> 히든층2 -> 출력층)
    - ReLU 활성화 함수 사용
    - 출력은 행동에 대한 로짓(logit) 값
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 3층 신경망 구성
        self.fc1 = nn.Linear(state_dim, hidden_dim)      # 입력층 -> 히든층1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 히든층1 -> 히든층2
        self.fc3 = nn.Linear(hidden_dim, action_dim)     # 히든층2 -> 출력층
    
    def forward(self, state):
        """순전파: 상태를 받아서 행동 로짓 반환"""
        x = torch.relu(self.fc1(state))  # 첫 번째 히든층 + ReLU
        x = torch.relu(self.fc2(x))      # 두 번째 히든층 + ReLU
        return self.fc3(x)               # 출력층 (로짓)
    
    def get_action(self, state, deterministic=False):
        """
        상태를 받아서 행동 선택
        
        Args:
            state: 환경 상태 (numpy array)
            deterministic: True면 확률이 가장 높은 행동, False면 확률적 샘플링
        
        Returns:
            선택된 행동 (정수)
        """
        # numpy array를 tensor로 변환하고 배치 차원 추가
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            logits = self.forward(state_tensor)           # 로짓 계산
            probs = torch.softmax(logits, dim=1)         # 확률 분포로 변환
        
        if deterministic:
            # 결정적 행동: 확률이 가장 높은 행동 선택
            return torch.argmax(probs, dim=1).item()
        else:
            # 확률적 행동: 확률 분포에서 샘플링
            action = torch.multinomial(probs, 1).item()
        
        return action


class ExpertPolicy:
    """
    전문가 정책 (규칙 기반)
    
    특징:
    - 환경별로 다른 규칙 적용
    - CartPole: 막대 각도 기반 단순 규칙
    - LunarLander: 위치, 속도, 각도 기반 복합 규칙
    """
    def __init__(self, env_name):
        self.env_name = env_name
    
    def get_action(self, state):
        """환경에 따라 적절한 전문가 정책 호출"""
        if "CartPole" in self.env_name:
            return self._cartpole_expert(state)
        elif "LunarLander" in self.env_name:
            return self._lunar_lander_expert(state)
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
    
    def _cartpole_expert(self, state):
        """
        CartPole 전문가 정책
        
        규칙:
        - 막대가 오른쪽으로 기울어지면 왼쪽으로 이동 (action=0)
        - 막대가 왼쪽으로 기울어지면 오른쪽으로 이동 (action=1)
        - 각속도도 고려하여 더 안정적인 제어
        
        Args:
            state: [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
        """
        angle = state[2]          # 막대 각도
        angle_velocity = state[3] # 막대 각속도

        # 막대가 오른쪽으로 기울어지거나, 거의 수직이지만 오른쪽으로 움직이는 경우
        if angle > 0.1 or (angle > -0.1 and angle_velocity > 0):
            return 1  # 오른쪽으로 이동
        else:
            return 0  # 왼쪽으로 이동
    
    def _lunar_lander_expert(self, state):
        """
        LunarLander 전문가 정책
        
        규칙:
        - 착륙장 근처에서는 속도 조절에 집중
        - 일반 비행 중에는 각도 조정과 하강 속도 제어
        
        Args:
            state: [x위치, y위치, x속도, y속도, 각도, 각속도, 왼쪽다리접촉, 오른쪽다리접촉]
        """
        x_pos, y_pos = state[0], state[1]  # 위치
        x_vel, y_vel = state[2], state[3]  # 속도
        angle = state[4]                   # 각도

        if y_pos < 0.5:  # 착륙장 근처 (y < 0.5)
            if abs(x_vel) > 0.1:  # 수평 속도가 크면
                return 1 if x_vel < 0 else 2  # 좌/우 엔진으로 속도 조절
            elif y_vel < -0.1:  # 하강 속도가 크면
                return 0  # 메인 엔진으로 감속
            else:
                return 3  # 아무것도 안함
        
        # 일반적인 비행 중
        if abs(angle) > 0.1:  # 각도 조정 필요
            return 1 if angle > 0 else 2  # 좌/우 엔진으로 각도 조정
        elif y_vel < -0.5:  # 하강 속도가 크면
            return 0  # 메인 엔진으로 감속
        else:
            return 3  # 아무것도 안함

class DataCollector:
    """
    데이터 수집 관리 클래스
    
    기능:
    - 전문가 정책으로 초기 데이터 수집
    - DAgger용 데이터 수집 (학생 정책 실행 + 전문가 라벨링)
    """
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.expert_policy = ExpertPolicy(env_name)
        self.data = []
    
    def collect_expert_rollouts(self, num_episodes):
        """
        전문가 정책으로 데이터 수집
        
        Args:
            num_episodes: 수집할 에피소드 수
        
        Returns:
            수집된 데이터 리스트 [(state, action, reward, next_state, done), ...]
        """
        data = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()  # 환경 초기화
            episode_data = []

            while True:
                # 전문가 정책으로 행동 선택
                expert_action = self.expert_policy.get_action(state)
                # 환경에서 행동 실행
                next_state, reward, terminated, truncated, _ = self.env.step(expert_action)
                done = terminated or truncated

                # 데이터 저장
                episode_data.append({
                    'state': state.copy(),           # 현재 상태
                    'action': expert_action,         # 수행된 행동
                    'reward': reward,                # 받은 보상
                    'next_state': next_state.copy(), # 다음 상태
                    'done': done                     # 에피소드 종료 여부
                })

                state = next_state
                if done:
                    break
            
            data.extend(episode_data)
        
        return data
    
    def collect_dagger_rollouts(self, student_policy, num_episodes):
        """
        DAgger용 데이터 수집
        
        특징:
        - 학생 정책으로 환경에서 실행
        - 전문가가 올바른 행동을 라벨로 제공
        - DAgger의 핵심: 현재 정책의 분포에서 데이터 수집
        
        Args:
            student_policy: 현재 학습된 학생 정책
            num_episodes: 수집할 에피소드 수
        
        Returns:
            DAgger용 데이터 리스트
        """
        data = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_data = []
            
            while True:
                # 학생 정책으로 행동 선택 (실제로는 이 행동을 실행)
                student_action = student_policy.get_action(state)
                # 전문가가 올바른 행동 제공 (라벨로 사용)
                expert_action = self.expert_policy.get_action(state)
                
                # 학생 행동으로 환경 실행
                next_state, reward, terminated, truncated, _ = self.env.step(student_action)
                done = terminated or truncated
                
                # DAgger 데이터 저장 (전문가 행동을 라벨로 사용)
                episode_data.append({
                    'state': state.copy(),
                    'action': expert_action,      # 전문가 행동을 라벨로 사용
                    'student_action': student_action,  # 학생 행동 기록 (분석용)
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
    """
    Behavioral Cloning (행동 복제)
    
    목적: 전문가의 행동을 모방하는 정책 학습
    방법: 지도학습으로 상태-행동 쌍 학습
    """
    def __init__(self, state_dim, action_dim, learning_rate=1e-3):
        self.policy = PolicyNetwork(state_dim, action_dim)  # 학습할 정책
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)  # Adam 옵티마이저
        self.criterion = nn.CrossEntropyLoss()  # 분류 손실 함수
    
    def train_step(self, states, actions):
        """
        단일 배치 학습
        
        Args:
            states: 상태 배치 (batch_size, state_dim)
            actions: 행동 배치 (batch_size,)
        
        Returns:
            손실 값
        """
        self.optimizer.zero_grad()  # 그래디언트 초기화
        
        logits = self.policy(states)  # 정책으로 행동 로짓 계산
        loss = self.criterion(logits, actions)  # 손실 계산

        loss.backward()  # 역전파
        self.optimizer.step()  # 파라미터 업데이트

        return loss.item()
    
    def train(self, dataset, epochs=100, batch_size=32):
        """
        전체 데이터셋으로 학습
        
        Args:
            dataset: 학습 데이터 [(state, action, ...), ...]
            epochs: 학습 에포크 수
            batch_size: 배치 크기
        
        Returns:
            에포크별 손실 리스트
        """
        # 데이터를 텐서로 변환
        states = torch.FloatTensor([d['state'] for d in dataset])
        actions = torch.LongTensor([d['action'] for d in dataset])

        n_samples = len(dataset)
        losses = []

        for epoch in range(epochs):
            # 에포크마다 데이터 순서를 랜덤하게 섞기
            indices = torch.randperm(n_samples)
            total_loss = 0

            # 배치 단위로 학습
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]

                loss = self.train_step(batch_states, batch_actions)
                total_loss += loss
            
            # 평균 손실 계산
            avg_loss = total_loss / (n_samples // batch_size)
            losses.append(avg_loss)

            # 진행 상황 출력
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses

    def evaluate(self, env, num_episodes=10):
        """
        정책 성능 평가
        
        Args:
            env: 평가할 환경
            num_episodes: 평가할 에피소드 수
        
        Returns:
            평가 결과 딕셔너리
        """
        total_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            while True:
                # 결정적 행동 선택 (평가시에는 확률적 행동 대신 최적 행동)
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
    """
    Dataset Aggregation (데이터셋 집계)
    
    목적: BC의 분포 이동 문제 해결
    방법: 현재 정책의 분포에서 데이터를 수집하고 전문가가 라벨링
    """
    def __init__(self, bc_trainer, data_collector):
        self.bc = bc_trainer  # BC 트레이너
        self.data_collector = data_collector  # 데이터 수집기
        self.dataset = []  # 전체 데이터셋
        self.results = []  # DAgger 반복 결과들
    
    def dagger_iteration(self, iteration, num_episodes=100):
        """
        DAgger 반복 수행
        
        과정:
        1. 현재 정책으로 데이터 수집
        2. 전문가가 올바른 행동 라벨링
        3. 확장된 데이터셋으로 재학습
        4. 성능 평가
        
        Args:
            iteration: 현재 반복 번호
            num_episodes: 수집할 에피소드 수
        
        Returns:
            반복 결과 딕셔너리
        """
        print(f"DAgger Iteration {iteration}")
        
        # 1. 현재 정책으로 데이터 수집
        new_data = self.data_collector.collect_dagger_rollouts(
            self.bc.policy, num_episodes
        )
        
        # 2. 데이터셋에 추가 (기존 데이터 + 새로운 데이터)
        self.dataset.extend(new_data)
        
        # 3. 확장된 데이터셋으로 재학습
        losses = self.bc.train(self.dataset, epochs=50, batch_size=32)
        
        # 4. 성능 평가
        eval_results = self.bc.evaluate(self.data_collector.env, num_episodes=10)
        
        # 결과 저장
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
        """
        전체 DAgger 실행
        
        Args:
            num_iterations: DAgger 반복 횟수
        
        Returns:
            모든 반복의 결과 리스트
        """
        for iteration in range(num_iterations):
            self.dagger_iteration(iteration)
        
        return self.results

class Evaluator:
    """
    정책 성능 평가 클래스
    
    기능:
    - 정책 성능 평가 (평균 보상, 성공률 등)
    - 전문가와 학생 정책 비교
    """
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.expert_policy = ExpertPolicy(env_name)
    
    def evaluate_policy(self, policy, num_episodes=10):
        """
        정책 성능 평가
        
        Args:
            policy: 평가할 정책
            num_episodes: 평가할 에피소드 수
        
        Returns:
            평가 결과 딕셔너리
        """
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
            
            # 성공 기준 판단 (환경별로 다름)
            if "CartPole" in self.env.spec.id:
                if episode_reward >= 195:  # CartPole 성공 기준 (500 스텝 중 195 이상)
                    success_count += 1
            elif "LunarLander" in self.env.spec.id:
                if episode_reward >= 200:  # LunarLander 성공 기준
                    success_count += 1
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': success_count / num_episodes,
            'rewards': total_rewards
        }
    
    def compare_policies(self, expert_policy, student_policy, num_episodes=10):
        """
        전문가와 학생 정책 비교
        
        Args:
            expert_policy: 전문가 정책
            student_policy: 학생 정책
            num_episodes: 비교할 에피소드 수
        
        Returns:
            비교 결과 딕셔너리
        """
        expert_results = self.evaluate_policy(expert_policy, num_episodes)
        student_results = self.evaluate_policy(student_policy, num_episodes)
        
        return {
            'expert': expert_results,
            'student': student_results,
            'performance_gap': expert_results['mean_reward'] - student_results['mean_reward']
        }
    
def plot_results(results):
    """
    DAgger 결과 시각화
    
    Args:
        results: DAgger 반복 결과 리스트
    """
    iterations = [r['iteration'] for r in results]
    mean_rewards = [r['mean_reward'] for r in results]
    dataset_sizes = [r['dataset_size'] for r in results]
    
    # 2개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 보상 변화 그래프
    ax1.plot(iterations, mean_rewards, 'b-o')
    ax1.set_xlabel('DAgger Iteration')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Performance over DAgger Iterations')
    ax1.grid(True)
    
    # 2. 데이터셋 크기 변화 그래프
    ax2.plot(iterations, dataset_sizes, 'r-o')
    ax2.set_xlabel('DAgger Iteration')
    ax2.set_ylabel('Dataset Size')
    ax2.set_title('Dataset Growth')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    메인 실행 함수 - 전체 BC + DAgger 파이프라인
    
    실행 순서:
    1. 환경 설정 및 정보 확인
    2. 전문가 데이터 수집 (500 에피소드)
    3. BC로 초기 정책 학습
    4. DAgger 반복 수행 (10회)
    5. 최종 성능 평가
    6. 결과 시각화
    """
    # 환경 설정
    env_name = "CartPole-v1"  # 또는 "LunarLander-v2"
    
    # 환경 정보 가져오기
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]  # 상태 차원
    action_dim = env.action_space.n             # 행동 차원
    env.close()
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # 1. 데이터 수집기 초기화
    collector = DataCollector(env_name)
    
    # 2. 초기 전문가 데이터 수집 (500 에피소드)
    print("Collecting expert data...")
    expert_data = collector.collect_expert_rollouts(500)
    print(f"Collected {len(expert_data)} expert transitions")
    
    # 3. BC 트레이너 초기화
    bc_trainer = BC(state_dim, action_dim)
    
    # 4. 초기 BC 학습
    print("Training initial BC policy...")
    bc_losses = bc_trainer.train(expert_data, epochs=100, batch_size=32)
    
    # 5. 초기 성능 평가
    evaluator = Evaluator(env_name)
    initial_eval = bc_trainer.evaluate(collector.env, num_episodes=10)
    print(f"Initial BC performance: {initial_eval['mean_reward']:.2f}")
    
    # 6. DAgger 초기화 및 실행
    dagger = DAgger(bc_trainer, collector)
    dagger.dataset = expert_data  # 초기 데이터셋 설정
    
    print("Running DAgger...")
    dagger_results = dagger.run_dagger(num_iterations=10)
    
    # 7. 최종 성능 평가
    final_eval = bc_trainer.evaluate(collector.env, num_episodes=10)
    print(f"Final performance: {final_eval['mean_reward']:.2f}")
    
    # 8. 결과 시각화
    plot_results(dagger_results)
    
    return dagger_results

if __name__ == "__main__":
    results = main()
