import argparse  # 명령줄에서 입력된 인자를 파싱하기 위한 모듈: 학습 설정을 사용자 정의로 조정 가능
import os  # 파일 경로를 다루거나 시스템 관련 작업(예: 디렉토리 생성)을 수행하기 위한 모듈
import random  # 무작위 수 생성(예: 시드 설정, 데이터 셔플링)을 위한 모듈
import time  # 학습 시간 측정 및 초당 스텝 수(SPS) 계산에 사용되는 모듈
from distutils.util import strtobool  # 문자열("True"/"False")을 불리언 값으로 변환하는 유틸리티 함수
import gym  # OpenAI Gym 환경을 생성하고 관리하기 위한 모듈: 강화학습 환경 제공
import gym_microrts  # Microrts 환경을 위한 모듈: 다중 이산 행동 공간(MultiDiscrete)을 지원
import numpy as np  # 배열 연산과 수학적 계산(예: 행렬 연산, 통계 계산)을 위한 라이브러리
import torch  # PyTorch 딥러닝 프레임워크: 신경망 학습 및 텐서 연산의 핵심
import torch.nn as nn  # 신경망 레이어 및 구조(예: Conv2d, Linear)를 정의하기 위한 PyTorch 모듈
import torch.optim as optim  # Adam과 같은 최적화 알고리즘을 제공하는 모듈: 신경망 학습 최적화
from torch.distributions.categorical import Categorical  # 다중 이산 행동 공간에서 행동을 샘플링하기 위한 이산 분포 모듈
from torch.utils.tensorboard import SummaryWriter  # 학습 과정을 TensorBoard로 시각화(손실, 보상 등 기록)하기 위한 모듈

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()  # 명령줄 인자를 처리하기 위한 파서 객체 생성
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="실험의 이름 (기본값은 현재 파일 이름에서 '.py'를 제거한 값)")  # 실험 식별용 이름
    parser.add_argument("--gym-id", type=str, default="MicrortsMining-v1",
        help="사용할 Gym 환경의 ID (기본값: MicrortsMining-v1)")  # Microrts 환경 설정
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="옵티마이저의 학습률 (기본값: 0.00025)")  # 신경망 파라미터 업데이트 속도
    parser.add_argument("--seed", type=int, default=1,
        help="실험 재현성을 위한 시드 값 (기본값: 1)")  # 난수 생성의 초기값 설정
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="총 학습 타임스텝 수 (기본값: 200만)")  # 학습 진행 길이
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="PyTorch의 결정론적 연산 사용 여부 (기본값: True)")  # 동일한 시드에서 동일 결과 보장
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="CUDA(GPU) 사용 여부 (기본값: True)")  # GPU 가속으로 학습 속도 향상
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Weights & Biases로 실험 추적 여부 (기본값: False)")  # 학습 과정 모니터링 옵션
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="Weights & Biases 프로젝트 이름 (기본값: ppo-implementation-details)")  # 추적 프로젝트 이름
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="Weights & Biases 팀 이름 (기본값: None)")  # 팀 또는 개인 계정 설정
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="에이전트 수행 비디오 녹화 여부 (기본값: False, 'videos' 폴더에 저장)")  # 학습 시각화 옵션

    # PPO 알고리즘 하이퍼파라미터
    parser.add_argument("--num-envs", type=int, default=8,
        help="병렬로 실행할 환경 수 (기본값: 8)")  # 병렬 데이터 수집으로 학습 효율성 증가
    parser.add_argument("--num-steps", type=int, default=128,
        help="각 롤아웃에서 환경별로 실행할 스텝 수 (기본값: 128)")  # 한 번에 수집할 데이터 크기
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="학습률을 점진적으로 감소시킬지 여부 (기본값: True)")  # 학습 후반 안정성 향상
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="GAE(Generalized Advantage Estimation) 사용 여부 (기본값: True)")  # 이점 계산 개선
    parser.add_argument("--gamma", type=float, default=0.99,
        help="할인율 (기본값: 0.99)")  # 미래 보상의 현재 가치 반영 비율
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE 계산에 사용할 람다 값 (기본값: 0.95)")  # 이점의 분산과 편향 조절
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="미니배치 수 (기본값: 4)")  # 배치 데이터를 나눠 학습 안정성 확보
    parser.add_argument("--update-epochs", type=int, default=4,
        help="정책 업데이트를 위한 에포크 수 (기본값: 4)")  # 한 번의 데이터로 반복 학습 횟수
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="이점 정규화 여부 (기본값: True)")  # 이점 값의 분포 안정화
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="대리 손실 클리핑 계수 (기본값: 0.1)")  # PPO의 핵심: 정책 업데이트 크기 제한
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="가치 함수 손실 클리핑 여부 (기본값: True)")  # 가치 학습의 큰 변화 방지
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="엔트로피 손실 계수 (기본값: 0.01, 탐험 활성화)")  # 행동 분포의 다양성 유지
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="가치 함수 손실 계수 (기본값: 0.5)")  # 가치 학습의 비중 조절
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="그래디언트 클리핑의 최대 노름 (기본값: 0.5)")  # 학습 안정성을 위한 그래디언트 제한
    parser.add_argument("--target-kl", type=float, default=None,
        help="KL 발산 목표치 (기본값: None)")  # 정책 변화의 상한 설정 (선택적)

    args = parser.parse_args()  # 모든 인자를 파싱하여 객체로 반환
    args.batch_size = int(args.num_envs * args.num_steps)  # 배치 크기 = 병렬 환경 수 * 롤아웃 스텝 수
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 미니배치 크기 = 배치 크기 / 미니배치 수
    # fmt: on
    return args

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)  # 지정된 ID로 Gym 환경 생성 (예: MicrortsMining-v1)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # 에피소드 통계(총 보상, 길이 등)를 기록하는 래퍼 추가
        if capture_video:  # 비디오 녹화 옵션이 활성화된 경우
            if idx == 0:  # 첫 번째 환경에서만 비디오 녹화 (리소스 절약)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")  # 비디오를 'videos/실험명' 폴더에 저장
        env.seed(seed)  # 환경의 난수 생성 시드 설정: 재현성 보장
        env.action_space.seed(seed)  # 행동 공간의 난수 생성 시드 설정
        env.observation_space.seed(seed)  # 관측 공간의 난수 생성 시드 설정
        return env
    return thunk  # 환경 생성 함수를 지연 실행(호출 시 생성)으로 반환

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # 신경망 레이어 초기화 함수
    torch.nn.init.orthogonal_(layer.weight, std)  # 가중치를 직교 초기화(기본 std=√2): 기울기 전파 안정화
    torch.nn.init.constant_(layer.bias, bias_const)  # 편향을 지정된 상수(기본값: 0)로 초기화
    return layer  # 초기화된 레이어 반환

class Transpose(nn.Module):  # 텐서 차원 재배열을 위한 사용자 정의 모듈
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation  # 재배열할 차원 순서 (예: (0, 3, 1, 2))

    def forward(self, x):
        return x.permute(self.permutation)  # 지정된 순서로 차원 재배열 (예: HWC -> CHW)

class Agent(nn.Module):
    def __init__(self, envs):  # 에이전트 초기화: 환경 정보를 입력으로 받아 구조 정의
        super(Agent, self).__init__()
        self.network = nn.Sequential(  # 공유 네트워크: CNN과 MLP로 구성된 특징 추출기
            Transpose((0, 3, 1, 2)),  # (batch, height, width, channels) -> (batch, channels, height, width)
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),  # 27채널 입력 -> 16채널 출력, 공간 크기 감소
            nn.ReLU(),  # 비선형 활성화 함수: 특징 추출 강화
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),  # 16채널 -> 32채널, 추가 특징 추출
            nn.ReLU(),
            nn.Flatten(),  # 2D 특징 맵을 1D 벡터로 변환
            layer_init(nn.Linear(32 * 3 * 3, 128)),  # 32*3*3=288 -> 128 차원으로 압축
            nn.ReLU(),
        )
        self.nvec = envs.single_action_space.nvec  # MultiDiscrete 행동 공간 벡터 (예: [3, 4, 5])
        self.actor = layer_init(nn.Linear(128, self.nvec.sum()), std=0.01)  # 128 -> 행동 차원 합계, 작은 std로 초기화
        self.critic = layer_init(nn.Linear(128, 1), std=1)  # 128 -> 1 (가치 출력), 표준 std로 초기화

    def get_value(self, x):  # 상태 가치를 계산하는 함수
        return self.critic(self.network(x))  # 공유 네트워크로 특징 추출 후 Critic으로 가치 예측

    def get_action_and_value(self, x, action=None):  # 행동과 가치를 동시에 계산
        hidden = self.network(x)  # 공유 네트워크로 관측값에서 특징 추출
        logits = self.actor(hidden)  # Actor를 통해 로짓(확률 분포의 입력) 생성
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)  # 로짓을 행동 차원별로 분리 (예: 3, 4, 5)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]  # 각 차원별 Categorical 분포 생성
        if action is None:  # 행동이 주어지지 않은 경우: 샘플링으로 생성
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])  # 각 차원의 로그 확률
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])  # 각 차원의 엔트로피
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)  # 행동, 총 로그 확률, 총 엔트로피, 가치 반환

if __name__ == "__main__":
    args = parse_args()  # 명령줄 인자 파싱: 학습 설정 로드
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"  # 고유한 실험 이름 생성: 환경+실험명+시드+타임스탬프
    if args.track:  # Weights & Biases 추적 활성화 시
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)  # wandb 초기화
    writer = SummaryWriter(f"runs/{run_name}")  # TensorBoard 로그 기록 설정: runs/실험명 디렉토리에 저장
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))  # 하이퍼파라미터를 표로 기록

    # 시드 설정: 재현성 보장
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic  # PyTorch의 결정론적 연산 설정

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")  # GPU 사용 가능 시 CUDA 설정, 아니면 CPU

    # 병렬 환경 생성
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])  # num_envs 개의 동기화된 환경 생성
    assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), "only MultiDiscrete action space is supported"  # MultiDiscrete 행동 공간 확인

    agent = Agent(envs).to(device)  # 에이전트 생성 및 디바이스(GPU/CPU) 할당
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam 옵티마이저 설정: 학습률과 안정성 파라미터 지정

    # 롤아웃 데이터 저장용 텐서: num_steps x num_envs 크기로 초기화
    obs = torch.zeros((args.num_steps, args.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # 관측값 저장
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # 행동 저장
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 로그 확률 저장
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 보상 저장
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 종료 신호 저장
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 상태 가치 저장

    # 학습 시작
    global_step = 0  # 총 스텝 수 추적
    start_time = time.time()  # 학습 시작 시간 기록: SPS 계산용
    next_obs = torch.Tensor(envs.reset()).to(device)  # 초기 관측값: 환경 리셋 후 텐서로 변환
    next_done = torch.zeros(args.num_envs).to(device)  # 초기 종료 신호: 모두 0으로 시작
    num_updates = args.total_timesteps // args.batch_size  # 총 업데이트 횟수: 전체 타임스텝 / 배치 크기

    for update in range(1, num_updates + 1):  # 업데이트 루프: 정책과 가치 개선
        if args.anneal_lr:  # 학습률 점진적 감소 옵션
            frac = 1.0 - (update - 1.0) / num_updates  # 남은 업데이트 비율 계산
            lrnow = frac * args.learning_rate  # 현재 학습률: 초기값에서 선형 감소
            optimizer.param_groups[0]["lr"] = lrnow  # 옵티마이저에 반영

        for step in range(0, args.num_steps):  # 롤아웃 스텝: 데이터 수집
            global_step += 1 * args.num_envs  # 글로벌 스텝 증가: 병렬 환경 수만큼
            obs[step] = next_obs  # 현재 관측값 저장
            dones[step] = next_done  # 현재 종료 신호 저장
            with torch.no_grad():  # 그래디언트 계산 비활성화: 데이터 수집 단계
                action, logprob, _, value = agent.get_action_and_value(next_obs)  # 행동과 가치 계산
                values[step] = value.flatten()  # 가치 저장: 1D로 변환
            actions[step] = action  # 행동 저장
            logprobs[step] = logprob  # 로그 확률 저장
            next_obs, reward, done, info = envs.step(action.cpu().numpy())  # 환경에서 행동 실행: numpy로 변환 후 진행
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # 보상 저장: 텐서로 변환
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)  # 다음 상태 업데이트

            for item in info:  # 에피소드 종료 시 통계 기록
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)  # 보상 기록
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)  # 길이 기록
                    break

        # GAE 또는 반환값 계산
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)  # 다음 상태 가치 계산
            if args.gae:  # GAE 사용 시: 이점 추정 개선
                advantages = torch.zeros_like(rewards).to(device)  # 이점 초기화
                lastgaelam = 0  # 마지막 GAE 값 초기화
                for t in reversed(range(args.num_steps)):  # 역방향으로 계산
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done  # 종료 여부: 0(종료) or 1(미종료)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]  # TD 오류 계산
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam  # GAE 계산
                returns = advantages + values  # 반환값 = 이점 + 가치
            else:  # GAE 미사용 시: 기본 반환값 계산
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return  # 할인된 누적 보상
                advantages = returns - values  # 이점 = 반환값 - 가치

        # 배치 데이터 평탄화: 학습에 사용할 1D 형태로 변환
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 정책 및 가치 네트워크 최적화
        b_inds = np.arange(args.batch_size)  # 배치 인덱스 생성: 0 ~ batch_size-1
        clipfracs = []  # 클리핑 비율 추적: PPO 클리핑 동작 모니터링
        for epoch in range(args.update_epochs):  # 에포크별 업데이트: 데이터 재사용
            np.random.shuffle(b_inds)  # 인덱스 섞기: 학습 다양성 증가
            for start in range(0, args.batch_size, args.minibatch_size):  # 미니배치 단위 처리
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # 미니배치 인덱스 추출
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)  # 새 예측값 계산
                logratio = newlogprob - b_logprobs[mb_inds]  # 새/기존 로그 확률 비율
                ratio = logratio.exp()  # 확률 비율: exp로 변환
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()  # 근사 KL 발산 계산: 정책 변화 측정
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  # 클리핑 비율 기록
                mb_advantages = b_advantages[mb_inds]  # 미니배치 이점
                if args.norm_adv:  # 이점 정규화: 평균 0, 표준편차 1로 조정
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio  # 비클리핑 정책 손실
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # 클리핑 정책 손실
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 최대 손실 선택: PPO의 핵심 안정성 메커니즘
                newvalue = newvalue.view(-1)
                if args.clip_vloss:  # 가치 손실 클리핑: 큰 변화 방지
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()  # 기본 가치 손실: MSE
                entropy_loss = entropy.mean()  # 엔트로피 손실: 탐험 장려
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # 총 손실: 정책 + 가치 + 엔트로피
                optimizer.zero_grad()  # 그래디언트 초기화
                loss.backward()  # 역전파: 그래디언트 계산
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # 그래디언트 클리핑: 안정성 보장
                optimizer.step()  # 파라미터 업데이트

        # 학습 통계 기록: TensorBoard에 저장
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)  # 초당 스텝 수

    envs.close()  # 환경 종료: 리소스 해제
    writer.close()  # TensorBoard 종료: 로그 기록 완료