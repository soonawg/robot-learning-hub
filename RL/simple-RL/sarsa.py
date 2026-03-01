import gymnasium as gym
import numpy as np
import random

# 환경 생성
env = gym.make("Taxi-v3", render_mode="human")
n_states = env.observation_space.n # 500
n_actions = env.action_space.n # 6

# Q-테이블 초기화 (상태 수 X 행동 수)
Q = np.zeros((n_states, n_actions))

# 하이퍼파라미터
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.1 # epsilon-greedy
episodes = 1000 # 에피소드 수

# SARSA 학습
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    # 초기 행동 선택
    action = random.random() < epsilon and env.action_space.sample() or np.argmax(Q[state])

    while not done:
        # 행동 실행
        next_state, reward, done, truncated, info = env.step(action)

        # 다음 행동 선택
        next_action = random.random() < epsilon and env.action_space.sample() or np.argmax(Q[next_state])

        # SARSA 업데이트
        target = reward + gamma * Q[next_state, next_action] * (not done)
        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        action = next_action

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

# 최적 policy 테스트
state, _ = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    env.render()

print(f"Total reward: {total_reward}")
env.close()
