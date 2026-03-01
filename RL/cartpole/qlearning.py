import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random


# 1. Q-테이블 초기화
# CartPole은 연속적인 상태 공간을 가지므로 이산화가 필요함
# 상태 공간을 적절한 구간으로 나누어 이산화 함수 구현
def discretize_state(state, bins):
    # CartPole-v1 상태 범위 설정(대략적인 값)
    # [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
    state_bounds = [
        (-4.8, 4.8), # 카트 위치
        (-5.0, 5.0), # 카트 속도
        (-0.418, 0.418), # 폴 각도 범위
        (-5.0, 5.0) # 폴 각속도
    ]

    
    # 각 상태 요소를 해당 구간을 이산화
    discretized = []
    for i in range(len(state)):
        feature_min, feature_max = state_bounds[i]
        # np.linspace로 각 범위를 bins+1개 구간으로 나눔
        bin_edges = np.linspace(feature_min, feature_max, bins + 1)
        # np.linspace 함수 : 지정된 범위 내에서 균등 간격으로 배열을 생성하는데 사용됨
        # np.linspace(start, stop, num)
        # ex : np.linspace(0, 10, num=5) => [0. 2.5 5. 7.5 10.]
        
        # np.digitize로 상태값이 어느 구간에 속하는지 확인
        bin_idx = np.digitize(state[i], bin_edges[1:-1])
        # np.digitize 함수 : 주어진 숫자 배열의 원소들을 특정한 구간(bin)에 할당하여 해당 구간의 인덱스를 반환하는 함수
        # np.digitize(x, bins)
        # ex : np.digitize([1.5, 2.5, 3.5], [0, 2, 4]) => [1 2 3]
        discretized.append(bin_idx)
    
    # 튜플로 반환하여 해시 가능하게 만듬 (Q-테이블 인덱싱용)
    return tuple(discretized)


# 2. Epsilon-greedy 정책
def choose_action(state, q_table, epsilon, env):
    # 랜덤 확률이 epsilon보다 작으면 랜덤 액션 선택
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        # Q-테이블에서 현재 상태가 없으면 새로 추가
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        # Q 값이 가장 높은 행동 선택
        return np.argmax(q_table[state])

# 3. Q-테이블 업데이트
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    # 현재 상태가 Q-테이블에 없으면 새로 추가
    if state not in q_table:
        q_table[state] = np.zeros(2) # CartPole은 2개 행동
    
    # 다음 상태가 Q-테이블에 없으면 새로 추가
    if next_state not in q_table:
        q_table[next_state] = np.zeros(2)
    
    # Q-러닝 업데이트 공식
    # Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
    q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

    return q_table

# 4. 학습 루프
def train(env, episodes, bins, alpha, gamma, epsilon, min_epsilon, epsilon_decay):
    # Q-테이블 초기화 (딕셔너리 형태)
    q_table = {}

    # 결과 추적용 리스트
    rewards = []
    steps = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)

        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False

        # 에피소드 실행
        while not (done or truncated):
            # 행동 선택
            action = choose_action(state, q_table, epsilon, env)

            # 환경에서 한 스텝 진행
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)

            # Q-테이블 업데이트
            q_table = update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

            # 상태 업데이트
            state = next_state

            # 보상과 스텝 기록
            episode_reward += reward
            episode_steps += 1

        # 결과 저장
        rewards.append(episode_reward)
        steps.append(episode_steps)

        # Epsilon 감소
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 진행 상황 출력
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}, Steps: {episode_steps}, Epsilon: {epsilon:.4f}")
    
    return q_table, rewards, steps

# 5. 테스트 함수
def test(env, q_table, episodes, bins):
    test_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)

        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # 학습된 정책에 따라 행동 선택 (탐험 없음)
            if state not in q_table:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state]) # 알 수 없는 상태면 무작위 행동
            
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, bins)

            state = next_state
            episode_reward += reward

            # 선택적: 테스트 중 환경 렌더링
            # env.render()
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}/{episodes}, Reward: {episode_reward}")
    
    print(f"Average Test Reward: {np.mean(test_rewards)}")
    return test_rewards


if __name__ == "__main__":
    # 환경 생성
    env = gym.make("CartPole-v1", render_mode="human")

    # 하이퍼파라미터
    episodes = 1000
    bins = 10 # 각 상태 차원당 구간 수
    alpha = 0.1 # 학습률
    gamma = 0.99 # 할인율
    epsilon = 1.0 # 초기 탐험 확률
    min_epsilon = 0.01 # 최소 탐험 확률
    epsilon_decay = 0.998 # 탐험 확률 감소율

    # 학습
    q_table, rewards, steps = train(env, episodes, bins, alpha, gamma, epsilon, min_epsilon, epsilon_decay)

    # 결과 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.show()
    
    # 테스트
    test_episodes = 10
    test_rewards = test(env, q_table, test_episodes, bins)
    
    env.close()