import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# CartPole 환경 생성 (기본 리워드 시스템 사용)
env = gym.make("CartPole-v1", render_mode="human")

# 상태 공간 이산화
bins = [np.linspace(-2.4, 2.4, 12),      
        np.linspace(-3.0, 3.0, 12),      
        np.linspace(-0.2, 0.2, 12),      
        np.linspace(-2.0, 2.0, 12)]

def discretize(state):
    discrete_state = []
    for i in range(4):
        clipped_val = np.clip(state[i], bins[i][0], bins[i][-1])
        discrete_idx = np.clip(np.digitize(clipped_val, bins[i]) - 1, 0, 11)
        discrete_state.append(discrete_idx)
    return tuple(discrete_state)

# Q-테이블 초기화
q_table = np.zeros([12, 12, 12, 12, 2])

# 하이퍼파라미터
alpha = 0.15      # 학습률
gamma = 0.99      # 할인계수
epsilon = 1.0     # 초기 탐험률
epsilon_min = 0.01
epsilon_decay = 0.995

# 성능 추적
episode_rewards = []
episode_lengths = []
success_episodes = []  # 195+ 스텝 달성한 에피소드

print("올바른 리워드 시스템으로 Q-Learning 훈련 시작...")
print("매 스텝마다 +1 리워드, 목표: 오래 버티기")

for episode in range(1500):
    state, _ = env.reset()
    state = discretize(state)
    
    total_reward = 0
    steps = 0
    
    while True:
        # 입실론-greedy 행동 선택
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # 행동 수행
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_state)
        
        # 기본 리워드 시스템 사용 (매 스텝 +1)
        # 추가적인 리워드 엔지니어링은 하지 않음
        
        # Q-값 업데이트
        if terminated or truncated:
            # 에피소드 종료 시
            q_table[state][action] += alpha * (reward - q_table[state][action])
        else:
            # 일반 Q-learning 업데이트
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    # 성능 기록
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    
    # 성공 에피소드 기록 (195+ 스텝)
    if steps >= 195:
        success_episodes.append(episode)
    
    # 입실론 감소
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # 진행상황 출력
    if (episode + 1) % 150 == 0:
        recent_100 = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
        avg_length = np.mean(recent_100)
        max_length = max(recent_100)
        success_rate = sum(1 for x in recent_100 if x >= 195) / len(recent_100)
        
        print(f"에피소드 {episode + 1:4d}: 평균={avg_length:6.1f}, 최대={max_length:3d}, 성공률={success_rate:5.1%}, ε={epsilon:.3f}")

env.close()

# 최종 성능 분석
final_100 = episode_lengths[-100:] if len(episode_lengths) >= 100 else episode_lengths
avg_final = np.mean(final_100)
max_final = max(final_100)
success_final = sum(1 for x in final_100 if x >= 195) / len(final_100)

print(f"\n{'='*50}")
print(f"최종 성능 분석 (최근 {len(final_100)}개 에피소드)")
print(f"{'='*50}")
print(f"평균 에피소드 길이: {avg_final:.1f} 스텝")
print(f"최대 에피소드 길이: {max_final} 스텝")
print(f"성공률 (195+ 스텝): {success_final:.1%}")
print(f"총 성공 에피소드: {len(success_episodes)}개")

# 성능 등급 판정
if success_final >= 0.9 and avg_final >= 195:
    grade = "🏆 우수 (환경 해결!)"
elif avg_final >= 200:
    grade = "🥈 매우 좋음"
elif avg_final >= 150:
    grade = "🥉 좋음" 
elif avg_final >= 100:
    grade = "📈 보통"
else:
    grade = "📉 개선 필요"

print(f"성능 등급: {grade}")

# 해결 기준 체크 (연속 100회에서 평균 195+ 달성)
consecutive_success = 0
max_consecutive = 0
current_consecutive = 0

for length in episode_lengths[-100:]:
    if length >= 195:
        current_consecutive += 1
        max_consecutive = max(max_consecutive, current_consecutive)
    else:
        current_consecutive = 0

if avg_final >= 195:
    print(f"🎉 CartPole-v1 해결 기준 달성!")
else:
    print(f"💪 해결까지 {195 - avg_final:.1f} 스텝 더 필요")

# 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(episode_lengths)
plt.axhline(y=195, color='r', linestyle='--', label='Success Threshold (195)')
plt.axhline(y=500, color='g', linestyle='--', label='Maximum (500)')
plt.title('Episode Lengths Over Time')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
# 이동평균 (50 에피소드)
window = 50
if len(episode_lengths) > window:
    moving_avg = [np.mean(episode_lengths[i:i+window]) for i in range(len(episode_lengths)-window)]
    plt.plot(range(window, len(episode_lengths)), moving_avg, 'b-', label=f'{window}-Episode Moving Average')
    plt.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
    plt.title(f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# 성공률 추이
success_rate_history = []
window = 50
for i in range(window, len(episode_lengths)):
    recent = episode_lengths[i-window:i]
    success_rate = sum(1 for x in recent if x >= 195) / window
    success_rate_history.append(success_rate)

if success_rate_history:
    plt.plot(range(window, len(episode_lengths)), success_rate_history, 'g-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='High Success (90%)')
    plt.title(f'Success Rate ({window}-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 최종 테스트 (학습된 정책 평가)
print(f"\n{'='*30}")
print("최종 테스트 (10회)")
print(f"{'='*30}")

test_env = gym.make("CartPole-v1", render_mode="human")
test_results = []

for test_episode in range(10):
    state, _ = test_env.reset()
    state = discretize(state)
    steps = 0
    
    while True:
        # 학습된 정책 사용 (탐험 없음)
        action = np.argmax(q_table[state])
        state, _, terminated, truncated, _ = test_env.step(action)
        state = discretize(state)
        steps += 1
        
        if terminated or truncated:
            break
    
    test_results.append(steps)
    status = "✅" if steps >= 195 else "❌"
    print(f"테스트 {test_episode + 1:2d}: {steps:3d} 스텝 {status}")

test_env.close()

test_avg = np.mean(test_results)
test_success = sum(1 for x in test_results if x >= 195) / 10

print(f"\n테스트 결과:")
print(f"평균: {test_avg:.1f} 스텝")
print(f"성공률: {test_success:.0%}")
print(f"최고: {max(test_results)} 스텝")
print(f"최저: {min(test_results)} 스텝")
