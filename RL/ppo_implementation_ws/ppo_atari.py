import argparse  # 명령줄 인자를 파싱하기 위한 모듈
import os  # 파일 경로 조작 및 시스템 관련 작업을 위한 모듈
import random  # 무작위 수 생성 (시드 설정 등에 사용)
import time  # 학습 시간 측정을 위한 모듈
from distutils.util import strtobool  # 문자열을 불리언 값으로 변환하는 유틸리티

import gym  # OpenAI Gym 환경을 제공하는 라이브러리
import numpy as np  # 배열 연산 및 수학 계산을 위한 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈 (레이어, 활성화 함수 등)
import torch.optim as optim  # 최적화 알고리즘 (Adam 등)
from torch.distributions.categorical import Categorical  # 이산 확률 분포 (행동 선택에 사용)
from torch.utils.tensorboard import SummaryWriter  # 학습 과정을 TensorBoard로 시각화

# Atari 환경 전처리를 위한 래퍼 (stable-baselines3에서 가져옴)
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,  # 보상을 [-1, 1]로 클리핑하여 학습 안정성 향상
    EpisodicLifeEnv,  # 생명 단위로 에피소드를 종료 처리
    FireResetEnv,  # 'FIRE' 행동으로 환경 초기화 (게임 시작)
    MaxAndSkipEnv,  # 프레임 스킵으로 계산 효율성 증가
    NoopResetEnv,  # 초기 상태를 무작위로 설정하여 다양성 확보
)

# 명령줄 인자를 파싱하여 실험 설정을 정의하는 함수
def parse_args():
    parser = argparse.ArgumentParser()  # ArgumentParser 객체 생성
    # 실험 이름 설정 (기본값은 현재 파일 이름에서 '.py'를 제거한 값)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="실험의 이름 (결과 저장 및 식별에 사용)")
    # 사용할 Gym 환경 ID (기본값은 Atari 게임 'BreakoutNoFrameskip-v4')
    parser.add_argument("--gym-id", type=str, default="BreakoutNoFrameskip-v4",
        help="Gym 환경의 ID (Atari 게임 환경 지정)")
    # Adam 최적화기의 초기 학습률 (작은 값으로 설정해 안정적인 학습 보장)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="최적화기의 학습률 (네트워크 파라미터 업데이트 속도 조절)")
    # 난수 생성 시드 (재현성을 보장하기 위해 설정)
    parser.add_argument("--seed", type=int, default=1,
        help="실험의 시드 (결과 재현성을 위한 값)")
    # 총 학습 타임스텝 수 (Breakout은 복잡하므로 1000만 스텝 설정)
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="총 학습 타임스텝 수 (학습 기간 설정)")
    # PyTorch 연산의 결정론적 설정 여부 (True면 동일한 결과 보장)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="torch의 결정론적 설정 여부 (재현성 강화)")
    # CUDA(GPU) 사용 여부 (True면 GPU 가속 사용)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="CUDA 사용 여부 (GPU 가속 옵션)")
    # Weights & Biases로 실험 추적 여부 (False면 로컬에서만 실행)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Weights and Biases로 실험 추적 여부 (원격 모니터링)")
    # wandb 프로젝트 이름 (실험 결과를 그룹화)
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="wandb 프로젝트 이름 (실험 그룹 식별)")
    # wandb 팀 이름 (협업 시 사용, 기본값 None)
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="wandb 팀 이름 (협업 팀 지정)")
    # 에이전트 성능 비디오 녹화 여부 (True면 videos 폴더에 저장)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="에이전트 성능 비디오 녹화 여부 (디버깅 및 시각화)")

    # PPO 알고리즘 관련 하이퍼파라미터
    # 병렬로 실행할 환경 수 (데이터 수집 효율성을 높임)
    parser.add_argument("--num-envs", type=int, default=8,
        help="병렬 환경 수 (데이터 병렬 수집)")
    # 각 롤아웃에서 환경별로 수집할 스텝 수 (128 스텝씩 수집)
    parser.add_argument("--num-steps", type=int, default=128,
        help="각 롤아웃에서 환경별 스텝 수 (롤아웃 크기)")
    # 학습률 점진적 감소 여부 (장기 학습 안정성 향상)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="학습률 점진적 감소 여부 (과적합 방지)")
    # GAE(Generalized Advantage Estimation) 사용 여부 (True면 이점 계산 개선)
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="GAE 사용 여부 (이점 추정 개선)")
    # 할인율 (미래 보상의 중요도, 0.99는 장기 보상 중시)
    parser.add_argument("--gamma", type=float, default=0.99,
        help="할인율 (미래 보상의 중요도 조절)")
    # GAE의 람다 파라미터 (편향-분산 트레이드오프 조절)
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE의 람다 파라미터 (이점 계산 안정성)")
    # 배치 데이터를 나눌 미니배치 수 (메모리 효율성 및 학습 안정성)
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="미니배치 수 (배치 분할)")
    # 정책 업데이트 반복 횟수 (4번 반복 학습)
    parser.add_argument("--update-epochs", type=int, default=4,
        help="정책 업데이트 에포크 수 (반복 학습 횟수)")
    # 이점 정규화 여부 (True면 이점 값의 분산 감소)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="이점 정규화 여부 (학습 안정성 향상)")
    # PPO의 클리핑 계수 (정책 업데이트 범위 제한, 0.1로 설정)
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="클리핑 계수 (정책 업데이트 안정성 조절)")
    # 가치 함수 손실 클리핑 여부 (True면 가치 손실 안정화)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="가치 함수 손실 클리핑 여부 (가치 추정 안정성)")
    # 엔트로피 손실 계수 (탐험을 장려, 0.01로 설정)
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="엔트로피 계수 (탐험 촉진)")
    # 가치 함수 손실 가중치 (가치 학습 비중, 0.5로 설정)
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="가치 함수 계수 (가치 손실 비중)")
    # 그래디언트 클리핑 최대 노름 (그래디언트 폭발 방지)
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="그래디언트 클리핑 최대 노름 (학습 안정성)")
    # KL 발산 목표치 (정책 변화 과도 시 조기 중단, 기본값 None)
    parser.add_argument("--target-kl", type=float, default=None,
        help="KL 발산 목표치 (조기 중단 기준)")

    args = parser.parse_args()  # 인자를 파싱하여 객체로 반환
    args.batch_size = int(args.num_envs * args.num_steps)  # 배치 크기 계산: 환경 수 * 스텝 수
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 미니배치 크기 계산: 배치 크기 / 미니배치 수
    return args  # 파싱된 인자 객체 반환

# Atari 환경을 생성하고 전처리하는 함수
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)  # 지정된 ID로 Gym 환경 생성 (예: BreakoutNoFrameskip-v4)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # 에피소드 통계 기록 (보상, 길이 등)
        if capture_video and idx == 0:  # 비디오 녹화 옵션 (첫 번째 환경만)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")  # videos 폴더에 비디오 저장
        # Atari 전처리 래퍼 적용
        env = NoopResetEnv(env, noop_max=30)  # 초기 프레임에서 최대 30번 'NOOP' 행동으로 상태 다양화
        env = MaxAndSkipEnv(env, skip=4)  # 4프레임 스킵으로 계산량 감소, 행동 반복 적용
        env = EpisodicLifeEnv(env)  # 생명 하나를 잃으면 에피소드 종료로 간주
        if "FIRE" in env.unwrapped.get_action_meanings():  # 환경에 'FIRE' 행동이 있으면
            env = FireResetEnv(env)  # 'FIRE'로 환경 초기화 (게임 시작 트리거)
        env = ClipRewardEnv(env)  # 보상을 [-1, 1]로 클리핑하여 학습 안정성 향상
        env = gym.wrappers.ResizeObservation(env, (84, 84))  # 관측을 84x84 크기로 조정 (CNN 입력 준비)
        env = gym.wrappers.GrayScaleObservation(env)  # RGB를 그레이스케일로 변환 (계산량 감소)
        env = gym.wrappers.FrameStack(env, 4)  # 4프레임을 스택하여 시간적 정보 제공 (CNN 학습 강화)
        env.seed(seed)  # 환경 난수 시드 설정 (재현성 보장)
        env.action_space.seed(seed)  # 행동 공간 시드 설정
        env.observation_space.seed(seed)  # 관측 공간 시드 설정
        return env  # 전처리된 환경 반환
    return thunk  # 클로저 반환 (병렬 환경 생성 시 호출)

# 신경망 레이어를 초기화하는 헬퍼 함수
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # 가중치를 직교 초기화 (기울기 전파 안정화, ReLU에 적합)
    torch.nn.init.constant_(layer.bias, bias_const)  # 편향을 상수로 초기화 (기본값 0)
    return layer  # 초기화된 레이어 반환

# 에이전트 클래스: PPO의 Actor와 Critic을 정의
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # 공유 네트워크: CNN과 MLP로 이미지 입력 처리
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),  # 입력: 4채널(프레임 스택), 출력: 32채널, 8x8 커널, 스트라이드 4
            nn.ReLU(),  # 비선형 활성화 함수로 특징 추출 강화
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # 32채널 -> 64채널, 4x4 커널, 스트라이드 2
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # 64채널 유지, 3x3 커널, 스트라이드 1
            nn.ReLU(),
            nn.Flatten(),  # 3D 텐서를 1D 벡터로 변환 (64*7*7)
            layer_init(nn.Linear(64 * 7 * 7, 512)),  # CNN 출력을 512차원 벡터로 변환
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)  # 행동 확률 출력 (작은 std로 초기화)
        self.critic = layer_init(nn.Linear(512, 1), std=1)  # 상태 가치 출력 (std=1로 설정)

    # 상태의 가치를 계산하는 메서드
    def get_value(self, x):
        return self.critic(self.network(x / 255.0))  # 관측을 0~1로 정규화 후 공유 네트워크와 Critic으로 가치 계산

    # 행동과 관련 정보를 반환하는 메서드
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)  # 관측 정규화 후 공유 네트워크로 특징 추출
        logits = self.actor(hidden)  # Actor로 행동 logits 계산
        probs = Categorical(logits=logits)  # 이산 확률 분포 생성 (Atari는 이산 행동 공간)
        if action is None:
            action = probs.sample()  # 확률 분포에서 행동 샘플링
        # 행동, 로그 확률, 엔트로피, 가치 반환 (PPO 학습에 필요한 모든 값)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# 메인 실행 블록
if __name__ == "__main__":
    args = parse_args()  # 명령줄 인자 파싱
    # 고유한 실험 이름 생성 (환경ID_실험이름_시드_시간)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:  # wandb로 실험 추적 설정
        import wandb
        wandb.init(
            project=args.wandb_project_name,  # 프로젝트 이름
            entity=args.wandb_entity,  # 팀 이름 (옵션)
            sync_tensorboard=True,  # TensorBoard와 동기화
            config=vars(args),  # 하이퍼파라미터 저장
            name=run_name,  # 실험 이름
            monitor_gym=True,  # Gym 환경 모니터링
            save_code=True,  # 코드 저장
        )
    writer = SummaryWriter(f"runs/{run_name}")  # TensorBoard 기록 설정
    # 하이퍼파라미터를 텍스트로 기록 (마크다운 형식)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 시드 설정 (재현성 보장)
    random.seed(args.seed)  # Python 기본 난수 생성기
    np.random.seed(args.seed)  # NumPy 난수 생성기
    torch.manual_seed(args.seed)  # PyTorch 난수 생성기
    torch.backends.cudnn.deterministic = args.torch_deterministic  # CUDA 연산 결정론적 설정

    # 장치 설정 (GPU 사용 가능 시 CUDA, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 병렬 환경 생성 (num_envs 개의 환경 동시 실행)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    # 환경이 이산 행동 공간인지 확인 (PPO 구현이 이산 공간에 특화됨)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "이 코드는 이산 행동 공간만 지원합니다"

    # 에이전트와 최적화기 초기화
    agent = Agent(envs).to(device)  # 에이전트를 지정된 장치로 이동
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam 최적화기 (eps는 수치 안정성)

    # 데이터 저장을 위한 텐서 초기화 (num_steps x num_envs 크기)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # 관측 저장
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # 행동 저장
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 로그 확률 저장
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 보상 저장
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 종료 여부 저장
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 가치 저장

    # 학습 시작 설정
    global_step = 0  # 전체 스텝 수를 추적
    start_time = time.time()  # 학습 시작 시간 기록
    next_obs = torch.Tensor(envs.reset()).to(device)  # 초기 관측값 (환경 리셋 후 텐서로 변환)
    next_done = torch.zeros(args.num_envs).to(device)  # 초기 종료 상태 (모두 0)
    num_updates = args.total_timesteps // args.batch_size  # 총 업데이트 횟수 계산

    # 학습 루프
    for update in range(1, num_updates + 1):
        # 학습률 점진적 감소 (anneal_lr=True일 경우)
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates  # 학습 진행률에 따라 감소 비율 계산
            lrnow = frac * args.learning_rate  # 현재 학습률 계산
            optimizer.param_groups[0]["lr"] = lrnow  # 최적화기에 적용

        # 롤아웃 데이터 수집
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs  # 병렬 환경 수만큼 스텝 증가
            obs[step] = next_obs  # 현재 관측 저장
            dones[step] = next_done  # 현재 종료 상태 저장
            # 추론 모드에서 행동과 가치 계산 (기울기 계산 비활성화)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()  # 가치 저장 (1D로 변환)
            actions[step] = action  # 행동 저장
            logprobs[step] = logprob  # 로그 확률 저장
            # 환경에서 한 스텝 진행 (행동을 CPU로 변환 후 실행)
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # 보상 저장
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)  # 다음 상태 업데이트

            # 에피소드 종료 시 통계 기록
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)  # 보상 기록
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)  # 길이 기록
                    break

        # 이점(advantage) 및 반환(return) 계산 (부트스트래핑)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)  # 다음 상태의 가치 계산
            if args.gae:  # GAE 사용 시 (더 정확한 이점 추정)
                advantages = torch.zeros_like(rewards).to(device)  # 이점 초기화
                lastgaelam = 0  # 마지막 GAE 값
                # 역순으로 이점 계산 (시간적 의존성 반영)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:  # 마지막 스텝
                        nextnonterminal = 1.0 - next_done  # 종료 여부 반영 (0이면 종료)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    # TD 오류 계산: r + γ * V(next) - V(current)
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    # GAE 공식: delta + γ * λ * 다음 이점
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values  # 반환 = 이점 + 가치
            else:  # GAE 미사용 시 기본 반환 계산
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return  # 기본 반환 공식
                advantages = returns - values  # 이점 = 반환 - 가치

        # 배치 데이터 평탄화 (2D -> 1D로 변환하여 학습 준비)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 정책 및 가치 네트워크 최적화
        b_inds = np.arange(args.batch_size)  # 배치 인덱스 생성
        clipfracs = []  # 클리핑 비율 저장 (디버깅용)
        for epoch in range(args.update_epochs):  # 지정된 에포크 수만큼 반복
            np.random.shuffle(b_inds)  # 미니배치 순서 무작위화 (과적합 방지)
            for start in range(0, args.batch_size, args.minibatch_size):  # 미니배치 단위로 처리
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # 현재 미니배치 인덱스
                # 새 정책으로 행동과 가치 재계산
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]  # 새 로그 확률과 이전 로그 확률의 차이
                ratio = logratio.exp()  # 확률 비율 (정책 변화 측정)

                # KL 발산 계산 (정책 변화의 정도를 모니터링)
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()  # 이전 KL 근사값
                    approx_kl = ((ratio - 1) - logratio).mean()  # 현재 KL 근사값
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  # 클리핑 비율

                mb_advantages = b_advantages[mb_inds]  # 미니배치 이점
                if args.norm_adv:  # 이점 정규화 (분산 감소로 학습 안정성 향상)
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 정책 손실 계산 (PPO의 핵심)
                pg_loss1 = -mb_advantages * ratio  # 클리핑 없는 손실
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # 클리핑 적용
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 더 큰 손실 선택 (보수적 업데이트)

                # 가치 손실 계산
                newvalue = newvalue.view(-1)  # 가치 텐서 평탄화
                if args.clip_vloss:  # 가치 손실 클리핑 사용 시
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2  # 클리핑 없는 MSE
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)  # 가치 클리핑
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2  # 클리핑된 MSE
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)  # 더 큰 손실 선택
                    v_loss = 0.5 * v_loss_max.mean()  # 평균 제곱 오차의 절반
                else:  # 클리핑 미사용 시 기본 MSE
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()  # 엔트로피 손실 (탐험 장려)
                # 총 손실: 정책 손실 - 엔트로피 손실 + 가치 손실
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 최적화 단계
                optimizer.zero_grad()  # 그래디언트 초기화
                loss.backward()  # 역전파로 그래디언트 계산
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # 그래디언트 클리핑
                optimizer.step()  # 파라미터 업데이트

            # KL 발산이 목표치를 넘으면 조기 중단 (정책 변화 제한)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 설명된 분산 계산 (가치 함수 성능 평가)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 학습 진행 상황 기록 (TensorBoard)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))  # 초당 스텝 수 출력
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()  # 환경 종료
    writer.close()  # TensorBoard 기록 종료