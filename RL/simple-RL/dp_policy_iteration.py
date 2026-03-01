import gymnasium as gym
import numpy as np

# 환경 설정
env = gym.make("Taxi-v3", render_mode="human")  # Taxi-v3 환경을 생성하고 시각화 모드 설정
n_states = env.observation_space.n  # 상태 공간의 크기 (가능한 상태의 수)
n_actions = env.action_space.n  # 행동 공간의 크기 (가능한 행동의 수)

# 가치 함수와 정책 초기화
V = np.zeros(n_states)  # 모든 상태의 가치 함수를 0으로 초기화
policy = np.random.randint(0, n_actions, n_states)  # 각 상태에 대해 무작위 행동을 선택하는 초기 정책

# 하이퍼파라미터
gamma = 0.9  # 할인율: 미래 보상의 중요도를 결정 (0에 가까울수록 근시안적, 1에 가까울수록 장기적)
theta = 1e-6  # 정책 평가 종료 조건: 가치 함수의 변화가 이 값보다 작으면 수렴했다고 판단

# 환경 모델
P = env.unwrapped.P  # 환경의 전이 확률 모델 (P[state][action] = [(prob, next_state, reward, done), ...])

# 정책 반복 알고리즘 시작
while True:
    # 정책 평가 단계: 현재 정책에 따른 상태 가치 함수 계산
    while True:
        delta = 0  # 가치 함수의 최대 변화량 초기화
        for state in range(n_states):  # 모든 상태에 대해 반복
            v_old = V[state]  # 현재 상태의 이전 가치 저장
            action = policy[state]  # 현재 정책에 따른 행동 선택
            total_value = 0  # 현재 상태-행동에 대한 가치 초기화
            
            # 현재 상태-행동에서 가능한 모든 다음 상태에 대한 기대값 계산
            for prob, next_state, reward, done in P[state][action]:
                # prob: 전이 확률, next_state: 다음 상태, reward: 보상, done: 종료 여부
                total_value += prob * (reward + gamma * V[next_state] * (not done))
                # (not done)은 종료 상태일 경우 미래 보상이 없음을 의미
            
            V[state] = total_value  # 계산된 가치로 업데이트
            delta = max(delta, abs(v_old - V[state]))  # 가치 변화의 최대값 갱신
        
        # 모든 상태의 가치 변화가 임계값(theta) 미만이면 정책 평가 종료
        if delta < theta:
            break
    
    # 정책 개선 단계: 계산된 가치 함수를 바탕으로 더 나은 정책 찾기
    policy_stable = True  # 정책 안정성 확인 변수 (정책이 변하지 않으면 True)
    for state in range(n_states):  # 모든 상태에 대해 반복
        old_action = policy[state]  # 현재 정책의 행동 저장
        action_values = []  # 각 행동의 가치를 저장할 리스트
        
        # 가능한 모든 행동에 대한 가치 계산
        for action in range(n_actions):
            total_value = 0
            for prob, next_state, reward, done in P[state][action]:
                total_value += prob * (reward + gamma * V[next_state] * (not done))
            action_values.append(total_value)
        
        policy[state] = np.argmax(action_values)  # 가장 높은 가치를 가진 행동으로 정책 업데이트
        
        # 정책이 변경되었는지 확인
        if old_action != policy[state]:
            policy_stable = False  # 하나라도 변경되었다면 정책이 안정적이지 않음
    
    # 정책이 안정적이면 (더 이상 개선되지 않으면) 알고리즘 종료
    if policy_stable:
        break

print("Policy Iteration 완료!")  # 알고리즘 완료 메시지
print("상태 0의 가치:", V[0])  # 첫 번째 상태의 가치 출력
print("상태 0의 정책 (행동):", policy[0])  # 첫 번째 상태에서의 최적 행동 출력

# 학습된 정책 테스트
state, _ = env.reset()  # 환경 초기화 및 초기 상태 얻기
done = False  # 에피소드 종료 여부
total_reward = 0  # 총 보상 초기화
while not done:  # 에피소드가 끝날 때까지 반복
    action = policy[state]  # 학습된 정책에 따라 행동 선택
    next_state, reward, done, truncated, info = env.step(action)  # 환경에서 한 스텝 진행
    total_reward += reward  # 보상 누적
    state = next_state  # 상태 업데이트
    env.render()  # 환경 시각화
print(f"Total Reward: {total_reward}")  # 총 획득 보상 출력
env.close()  # 환경 종료
