import gymnasium as gym
import numpy as np

# 환경 생성
env = gym.make("Taxi-v3", render_mode="human")

## Q-테이블 초기화 (상태 수 X 행동 수)
n_states = env.observation_space.n # 500
n_actions = env.action_space.n # 6
Q = np.zeros((n_states, n_actions)) # 500x6 크기의 0으로 구성된 테이블 생성

## 파라미터 설정
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.1 # epsilon-greedy
episodes = 1000 # 에피소드 수


## 학습 루프
for episode in range(episodes):
    state, _ = env.reset() # 환경 초기화 
    done = False
    while not done:
        # epsilon-greedy로 행동 선택
        if np.random.rand() < epsilon:
            action = env.action_space.sample() # 랜덤 행동 (탐험)
        else:
            action = np.argmax(Q[state]) # 최적 행동 (활용)

        # 행동 실행
        next_state, reward, done, truncated, info = env.step(action)

        # Q-값 업데이트
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        # 상태 업데이트
        state = next_state
    
    if episode % 100 == 0:
        print(f"Episode {episode} completed")

    # 학습된 정책 테스트
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        env.render()

    print(f"Total reward: {reward}")
    env.close()
