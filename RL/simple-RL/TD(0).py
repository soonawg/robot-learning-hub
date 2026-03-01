import gymnasium as gym
import numpy as np
import random

# 환경 생성
env = gym.make("Taxi-v3", render_mode="human")
n_states = env.observation_space.n # 500
# model free prediction이기에 action 값이 불필요

# 가치 함수 초기화
V = np.zeros(n_states)

# 하이퍼파라미터
alpha = 0.1 # 학습률
gamma = 0.9 # discount factor
episodes = 1000 # 에피소드 수

# 고정된 policy (랜덤 policy)
def random_policy(state):
    return env.action_space.sample()

# TD(0)로 policy evaluation
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # 랜덤 policy으로 행동 선택
        action = random_policy(state)
        next_state, reward, done, truncated, info = env.step(action)

        # TD(0) 업데이트
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])

        state = next_state
    
    if episode % 100 == 0:
        print(f"Episode {episode} completed")
    
# 결과 출력
print("TD(0) Policy Evaluation 완료!")
print("상태 0의 가치:", V[0])
print("상태 100의 가치:", V[100])
