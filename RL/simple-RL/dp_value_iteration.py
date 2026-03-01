import gymnasium as gym
import numpy as np

# 환경 설정
# Taxi-v3는 4x4 그리드에서 승객을 목적지까지 운송하는 환경
# 상태 공간: 500 (25개 위치 x 5개 승객 위치 x 4개 목적지)
# 행동 공간: 6 (상, 하, 좌, 우, 승객 태우기, 승객 내리기)
env = gym.make("Taxi-v3", render_mode="human")
n_states = env.observation_space.n  # 500개의 상태
n_actions = env.action_space.n      # 6개의 행동

# 가치 함수와 정책 초기화
# V: 각 상태의 가치를 저장하는 배열
# policy: 각 상태에서 취할 최적 행동을 저장하는 배열
V = np.zeros(n_states)
policy = np.zeros(n_states, dtype=int)

# 하이퍼파라미터 설정
gamma = 0.9    # 미래 보상에 대한 할인율 (0~1)
theta = 1e-6   # 가치 함수 수렴 판단 기준

# 환경의 동적 모델 가져오기
# P[state][action] = [(확률, 다음_상태, 보상, 종료_여부), ...]
P = env.unwrapped.P

# Value Iteration 알고리즘 구현
while True:
    delta = 0  # 가치 함수의 최대 변화량
    for state in range(n_states):
        v_old = V[state]
        # 현재 상태에서 가능한 모든 행동의 기대 가치 계산
        action_values = []
        for action in range(n_actions):
            total_value = 0
            # 각 행동의 모든 가능한 결과에 대한 기대값 계산
            for prob, next_state, reward, done in P[state][action]:
                # Bellman 방정식 적용
                total_value += prob * (reward + gamma * V[next_state] * (not done))
            action_values.append(total_value)
        # 최대 가치로 업데이트
        V[state] = max(action_values)
        delta = max(delta, abs(v_old - V[state]))
    
    # 수렴 확인
    if delta < theta:
        break

# 최적 정책 도출
# 각 상태에서 가장 높은 기대 가치를 주는 행동 선택
for state in range(n_states):
    action_values = []
    for action in range(n_actions):
        total_value = 0
        for prob, next_state, reward, done in P[state][action]:
            total_value += prob * (reward + gamma * V[next_state] * (not done))
        action_values.append(total_value)
    policy[state] = np.argmax(action_values)

print("Value iteration completed")
print("상태 0의 가치:", V[0])
print("상태 0의 policy (행동):", policy[0])

# 학습된 정책 테스트
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    # 현재 상태에서 정책이 지시하는 행동 수행
    action = policy[state]
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    total_reward += reward
    env.render()
print(f"Total reward: {total_reward}")
env.close()
