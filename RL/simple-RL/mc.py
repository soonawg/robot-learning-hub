import numpy as np
import gymnasium as gym
import random

env = gym.make("Taxi-v3", render_mode="human")
n_states = env.observation_space.n  # 상태 공간의 크기 (500가지 상태)
n_actions = env.action_space.n      # 행동 공간의 크기 (6가지 행동)

# 각 상태-행동 쌍의 가치를 저장하는 Q-테이블 초기화
Q = np.zeros((n_states, n_actions))

# 학습 알고리즘의 하이퍼파라미터 설정
alpha = 0.1   # 학습률: 새로운 정보를 얼마나 반영할지 결정
gamma = 0.9   # 할인율: 미래 보상의 중요도를 결정
epsilon = 0.1 # 탐험 확률: epsilon 확률로 무작위 행동 선택
episodes = 1000  # 학습할 총 에피소드 수

# First-visit Monte Carlo를 위한 상태-행동 쌍 방문 횟수 기록
N = np.zeros((n_states, n_actions))

# Monte Carlo 학습 루프
for episode in range(episodes):
    state, _ = env.reset()  # 새로운 에피소드 시작
    episode_data = []       # 현재 에피소드의 (상태, 행동, 보상) 이력 저장
    done = False

    # 한 에피소드 동안의 경험 수집
    while not done:
        # epsilon-greedy 정책으로 행동 선택
        if random.random() < epsilon:
            action = env.action_space.sample()  # 탐험: 무작위 행동
        else:
            action = np.argmax(Q[state])       # 활용: 최적 행동
        
        # 선택한 행동을 실행하고 결과 관찰
        next_state, reward, done, truncated, info = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
    
    # 에피소드 종료 후 반환(G) 계산 및 Q-값 업데이트
    G = 0  # 누적 할인 보상
    visited = set()  # First-visit을 체크하기 위한 집합
    # 에피소드의 마지막부터 거꾸로 진행
    for t in range(len(episode_data) -1, -1, -1):
        state, action, reward = episode_data[t]
        state_action = (state, action)
        G = reward + gamma * G  # 할인된 보상 계산
        
        # First-visit Monte Carlo: 각 상태-행동 쌍의 첫 방문에 대해서만 업데이트
        if state_action not in visited:
            visited.add(state_action)
            N[state, action] += 1  # 방문 횟수 증가
            # Q-값 업데이트: Q(s,a) ← Q(s,a) + α[G - Q(s,a)]
            Q[state, action] += alpha * (G - Q[state, action])
    
    # 학습 진행 상황 출력
    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# 학습된 정책 테스트: 학습된 Q-값을 사용하여 최적 행동 선택
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])  # 학습된 Q-값 중 최대값을 가진 행동 선택
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    env.render()  # 환경 시각화

# 최종 결과 출력 및 환경 종료
print(f"Total reward: {total_reward}")
env.close()
