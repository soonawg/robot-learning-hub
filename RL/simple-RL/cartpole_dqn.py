## CartPole-v1 환경개요
# 상태 공간 : 4차원 연속 벡터 (카트 위치, 속도, 폴 각도, 폴 각속도)
# 행동 공간 : 2개 (왼쪽 힘, 오른쪽 힘)
# 보상 : 폴이 세워진 상태를 유지하면 +1, 넘어지면 종료
# 에피소드 종료 : 폴이 12도 이상 기울거나 카트가 2.4 단위 벗어나면 종료

import gymnasium as gym  # gymnasium 사용 - OpenAI Gym의 개선된 버전
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘
import numpy as np  # 수치 계산 라이브러리
import random  # 랜덤 샘플링을 위한 라이브러리
from collections import deque  # 경험 리플레이 메모리 구현을 위한 양방향 큐

# 하이퍼파라미터 설정
LEARNING_RATE = 0.001  # 학습률: 경사 하강법의 스텝 크기
GAMMA = 0.99  # 할인율: 미래 보상의 중요도 (0~1 사이, 1에 가까울수록 미래 보상 중요)
EPSILON = 1.0  # 초기 탐험률: 처음에는 100% 랜덤 행동
EPSILON_DECAY = 0.995  # 탐험률 감소 비율: 에피소드마다 이 값을 곱함
MIN_EPSILON = 0.01  # 최소 탐험률: 이 값 이하로는 감소하지 않음
EPISODES = 1000  # 최대 학습 에피소드 수
MEMORY_SIZE = 10000  # 경험 리플레이 메모리 크기
BATCH_SIZE = 64  # 한 번에 학습할 배치 크기 (학습 안정성 위해 64로 증가)
TARGET_UPDATE = 10  # 타겟 네트워크 업데이트 주기 (에피소드 단위)

# Q-네트워크 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # 3층 신경망 구조: 입력층 -> 128 -> 64 -> 출력층
        self.fc1 = nn.Linear(input_size, 128)  # 첫 번째 완전연결층: 상태 차원 -> 128 뉴런
        self.fc2 = nn.Linear(128, 64)  # 두 번째 완전연결층: 128 -> 64 뉴런
        self.fc3 = nn.Linear(64, output_size)  # 출력층: 64 -> 행동 수 (2)

    def forward(self, x):
        # 순전파 함수: 입력을 각 층에 통과시키며 활성화 함수(ReLU) 적용
        x = torch.relu(self.fc1(x))  # ReLU 활성화 함수: max(0, x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 출력층에는 활성화 함수 없음 (Q값은 임의의 실수)
        return x

# 경험 리플레이 메모리
class ReplayMemory:
    def __init__(self, capacity):
        # 최대 용량이 정해진 양방향 큐 생성 (오래된 경험은 자동으로 삭제됨)
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 새로운 경험(transition)을 메모리에 추가
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 메모리에서 무작위로 batch_size만큼 경험 샘플링
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # 현재 저장된 경험의 개수 반환
        return len(self.memory)

# 환경 설정
env = gym.make('CartPole-v1', render_mode=None)  # render_mode=None으로 불필요한 렌더링 방지하여 학습 속도 향상
input_size = env.observation_space.shape[0]  # 상태 공간 차원: 4 (카트 위치, 속도, 폴 각도, 각속도)
output_size = env.action_space.n  # 행동 공간 차원: 2 (왼쪽, 오른쪽)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능하면 GPU, 아니면 CPU 사용

# 네트워크 초기화
policy_net = DQN(input_size, output_size).to(device)  # 현재 정책을 결정하는 네트워크
target_net = DQN(input_size, output_size).to(device)  # Q-학습 타겟값 계산에 사용되는 네트워크 (안정성 향상)
target_net.load_state_dict(policy_net.state_dict())  # 타겟 네트워크를 정책 네트워크와 동일하게 초기화
target_net.eval()  # 타겟 네트워크는 학습하지 않고 평가 모드로 설정
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)  # Adam 최적화 알고리즘 사용
criterion = nn.MSELoss()  # 평균 제곱 오차(MSE) 손실 함수 사용
memory = ReplayMemory(MEMORY_SIZE)  # 경험 리플레이 메모리 초기화

# 훈련 루프: 지정된 에피소드 수만큼 반복
for episode in range(EPISODES):
    state, info = env.reset()  # 환경 초기화 (gymnasium API: state와 info 반환)
    state = np.array(state)  # numpy.ndarray로 변환
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # PyTorch 텐서로 변환 및 배치 차원 추가
    total_reward = 0  # 에피소드 총 보상 초기화
    done = False  # 종료 상태 초기화

    # 한 에피소드 실행
    while not done:
        # 엡실론 탐욕 정책: 확률 epsilon으로 무작위 행동, (1-epsilon)으로 최적 행동 선택
        if random.random() < EPSILON:
            action = env.action_space.sample()  # 무작위 행동 선택 (탐험)
        else:
            with torch.no_grad():  # 그래디언트 계산 비활성화 (추론 모드)
                action = policy_net(state).argmax().item()  # 최대 Q값을 가진 행동 선택 (활용)

        # 선택한 행동으로 환경 진행 (gymnasium API)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)  # 텐서 변환
        total_reward += reward  # 보상 누적

        # 종료 조건 통합 (terminated: 실패로 인한 종료, truncated: 최대 스텝 도달로 인한 종료)
        done = terminated or truncated

        # 경험 저장: (현재 상태, 행동, 보상, 다음 상태, 종료 여부)
        memory.push(state, action, reward, next_state, done)

        # 메모리에 충분한 경험이 쌓이면 학습 진행
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)  # 무작위로 배치 샘플링
            states, actions, rewards, next_states, dones = zip(*batch)  # 배치 데이터 언패킹

            # 배치 데이터 텐서 변환 및 처리
            states = torch.cat(states)  # 상태들을 하나의 배치 텐서로 결합
            actions = torch.tensor(actions, device=device)  # 행동들을 텐서로 변환
            rewards = torch.tensor(rewards, device=device)  # 보상들을 텐서로 변환
            next_states = torch.cat(next_states)  # 다음 상태들을 하나의 배치 텐서로 결합
            dones = torch.tensor(dones, device=device, dtype=torch.float32)  # 종료 여부를 텐서로 변환

            # 현재 Q값 계산: 선택한 행동에 대한 Q값만 추출
            q_values = policy_net(states).gather(1, actions.unsqueeze(1))
            
            # 타겟 Q값 계산: Q-학습 업데이트 식 사용
            # Q(s,a) = r + γ * max_a'(Q(s',a')) * (1-done)
            with torch.no_grad():  # 타겟 계산에는 그래디언트 필요 없음
                target_q_values = rewards + GAMMA * target_net(next_states).max(1)[0] * (1 - dones)
            
            # 손실 계산: 현재 Q값과 타겟 Q값의 차이 (MSE)
            loss = criterion(q_values.squeeze(), target_q_values)

            # 역전파 및 최적화
            optimizer.zero_grad()  # 그래디언트 초기화
            loss.backward()  # 역전파로 그래디언트 계산
            optimizer.step()  # 파라미터 업데이트

        state = next_state  # 상태 업데이트

    # 타겟 네트워크 주기적 업데이트 (학습 안정성 향상)
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 엡실론 감소: 점점 무작위 행동 비율 줄임 (탐험에서 활용으로 전환)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {EPSILON}")

    # 조기 종료 조건: 문제 해결 시 학습 중단
    if total_reward >= 500:  # CartPole-v1의 최대 보상 (gymnasium 기준)
        print(f"Solved in {episode} episodes!")
        break

env.close()  # 환경 종료 및 자원 해제
