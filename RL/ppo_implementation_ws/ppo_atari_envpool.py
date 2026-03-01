import argparse  # 명령줄 인자를 파싱하기 위한 모듈
import os  # 파일 경로 및 시스템 작업을 처리하기 위한 모듈
import random  # 무작위 수 생성 (예: 시드 설정)용 모듈
import time  # 학습 시간 측정 및 성능 평가를 위한 모듈
from collections import deque  # 최근 에피소드 반환값의 평균을 추적하기 위한 양방향 큐
from distutils.util import strtobool  # 문자열을 불리언 값으로 변환하는 유틸리티

import envpool  # 고성능 병렬 환경을 제공하는 라이브러리
import gym  # OpenAI Gym 환경을 위한 모듈
import numpy as np  # 배열 연산 및 수학 계산을 위한 라이브러리
import torch  # PyTorch 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈 (레이어 정의 등)
import torch.optim as optim  # 최적화 알고리즘 (Adam 등)
from torch.distributions.categorical import Categorical  # 이산 확률 분포를 위한 모듈 (행동 선택 시 사용)
from torch.utils.tensorboard import SummaryWriter  # 학습 과정 시각화를 위한 TensorBoard 모듈

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()  # 명령줄 인자 파서를 생성
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")  # 실험 이름 (기본값: 파일 이름에서 '.py' 제거)
    parser.add_argument("--gym-id", type=str, default="Pong-v5",
        help="the id of the gym environment")  # Gym 환경 ID (기본값: Pong-v5)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")  # 학습률 (기본값: 0.00025)
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")  # 실험 재현성을 위한 시드 (기본값: 1)
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")  # 총 학습 타임스텝 수 (기본값: 1000만)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")  # PyTorch의 결정론적 동작 설정 (기본값: True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")  # CUDA 사용 여부 (기본값: True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")  # Weights & Biases로 실험 추적 여부 (기본값: False)
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")  # wandb 프로젝트 이름 (기본값: ppo-implementation-details)
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")  # wandb 팀 이름 (기본값: None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")  # 비디오 녹화 여부 (기본값: False)

    # PPO 알고리즘 관련 하이퍼파라미터
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")  # 병렬 환경 수 (기본값: 16)
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")  # 각 롤아웃에서 환경별 스텝 수 (기본값: 128)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")  # 학습률 점진적 감소 여부 (기본값: True)
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")  # GAE(Generalized Advantage Estimation) 사용 여부 (기본값: True)
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")  # 할인율 (기본값: 0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")  # GAE 람다 (기본값: 0.95)
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")  # 미니배치 수 (기본값: 8)
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")  # 정책 업데이트 에포크 수 (기본값: 3)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")  # 이점 정규화 여부 (기본값: True)
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")  # 클리핑 계수 (기본값: 0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")  # 가치 함수 손실 클리핑 여부 (기본값: True)
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")  # 엔트로피 계수 (기본값: 0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")  # 가치 함수 계수 (기본값: 0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")  # 그래디언트 클리핑 최대 노름 (기본값: 0.5)
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")  # KL 발산 목표치 (기본값: None)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # 배치 크기: 병렬 환경 수 * 스텝 수
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 미니배치 크기: 배치 크기 / 미니배치 수
    # fmt: on
    return args

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):  # 환경 래퍼 초기화, deque_size는 사용되지 않음
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)  # 병렬 환경 수 (기본값: 1)
        self.episode_returns = None  # 에피소드별 누적 보상
        self.episode_lengths = None  # 에피소드별 길이

    def reset(self, **kwargs):  # 환경 초기화
        observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)  # 보상 초기화
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)  # 길이 초기화
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)  # 완료된 에피소드 보상
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)  # 완료된 에피소드 길이
        return observations

    def step(self, action):  # 환경 스텝 실행
        observations, rewards, dones, infos = super(RecordEpisodeStatistics, self).step(action)
        self.episode_returns += rewards  # 보상을 누적
        self.episode_lengths += 1  # 스텝 수 증가
        self.returned_episode_returns[:] = self.episode_returns  # 현재 보상 복사
        self.returned_episode_lengths[:] = self.episode_lengths  # 현재 길이 복사
        self.episode_returns *= 1 - dones  # 에피소드 종료 시 보상 초기화 (done=1이면 0으로)
        self.episode_lengths *= 1 - dones  # 에피소드 종료 시 길이 초기화
        infos["r"] = self.returned_episode_returns  # 정보에 완료된 보상 추가
        infos["l"] = self.returned_episode_lengths  # 정보에 완료된 길이 추가
        return (
            observations,
            rewards,
            dones,
            infos,
        )

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # 신경망 레이어 초기화
    torch.nn.init.orthogonal_(layer.weight, std)  # 가중치를 직교 초기화 (기본 std=√2)
    torch.nn.init.constant_(layer.bias, bias_const)  # 편향을 상수로 초기화 (기본값: 0)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):  # 에이전트 초기화
        super(Agent, self).__init__()
        self.network = nn.Sequential(  # 공유 네트워크: CNN으로 이미지 처리
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),  # 4채널 입력 (프레임 스택), 32 필터, 스트라이드 4
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),  # 32->64 필터, 스트라이드 2
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),  # 64 필터, 스트라이드 1
            nn.ReLU(),
            nn.Flatten(),  # 2D -> 1D 평탄화
            layer_init(nn.Linear(64 * 7 * 7, 512)),  # 64*7*7 -> 512 완전 연결층
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)  # 행동 logits (작은 std로 초기화)
        self.critic = layer_init(nn.Linear(512, 1), std=1)  # 상태 가치 (std=1로 초기화)

    def get_value(self, x):  # 상태 가치 계산
        return self.critic(self.network(x / 255.0))  # 관측값 정규화 (0~255 -> 0~1) 후 가치 반환

    def get_action_and_value(self, x, action=None):  # 행동과 가치 계산
        hidden = self.network(x / 255.0)  # 관측값 정규화 후 특징 추출
        logits = self.actor(hidden)  # 행동 logits 계산
        probs = Categorical(logits=logits)  # 이산 확률 분포 생성
        if action is None:  # 행동이 주어지지 않으면 샘플링
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)  # 행동, 로그 확률, 엔트로피, 가치 반환

if __name__ == "__main__":
    args = parse_args()  # 명령줄 인자 파싱
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"  # 실험 이름 생성 (고유 식별용)
    if args.track:  # wandb로 실험 추적 설정
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")  # TensorBoard 로그 기록 설정
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )  # 하이퍼파라미터를 텍스트로 기록

    # 시드 설정 (재현성 보장)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")  # CUDA 사용 가능 시 GPU 설정

    # 환경 설정
    envs = envpool.make(args.gym_id, env_type="gym", num_envs=args.num_envs)  # envpool로 병렬 환경 생성
    envs.num_envs = args.num_envs  # 환경 수 설정
    envs.single_action_space = envs.action_space  # 단일 환경의 행동 공간
    envs.single_observation_space = envs.observation_space  # 단일 환경의 관측 공간
    envs = RecordEpisodeStatistics(envs)  # 통계 기록 래퍼 적용
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"  # 이산 행동 공간 확인

    agent = Agent(envs).to(device)  # 에이전트 생성 및 디바이스 할당
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam 최적화기 설정

    # 데이터 저장용 텐서 (롤아웃 데이터 저장)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # 관측값
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # 행동
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 로그 확률
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 보상
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 종료 신호
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 상태 가치
    avg_returns = deque(maxlen=20)  # 최근 20개 에피소드 반환값 평균 추적

    # 학습 시작
    global_step = 0  # 총 스텝 수
    start_time = time.time()  # 학습 시작 시간
    next_obs = torch.Tensor(envs.reset()).to(device)  # 초기 관측값
    next_done = torch.zeros(args.num_envs).to(device)  # 초기 종료 신호
    num_updates = args.total_timesteps // args.batch_size  # 총 업데이트 횟수

    for update in range(1, num_updates + 1):  # 업데이트 루프
        if args.anneal_lr:  # 학습률 점진적 감소
            frac = 1.0 - (update - 1.0) / num_updates  # 남은 업데이트 비율
            lrnow = frac * args.learning_rate  # 현재 학습률
            optimizer.param_groups[0]["lr"] = lrnow  # 학습률 업데이트

        for step in range(0, args.num_steps):  # 롤아웃 스텝
            global_step += 1 * args.num_envs  # 글로벌 스텝 증가
            obs[step] = next_obs  # 현재 관측값 저장
            dones[step] = next_done  # 현재 종료 신호 저장

            # 행동 선택 및 가치 계산
            with torch.no_grad():  # 그래디언트 계산 비활성화 (데이터 수집용)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()  # 가치 저장
            actions[step] = action  # 행동 저장
            logprobs[step] = logprob  # 로그 확률 저장

            # 환경 실행
            next_obs, reward, done, info = envs.step(action.cpu().numpy())  # 행동 실행
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # 보상 저장
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)  # 다음 상태 업데이트

            for idx, d in enumerate(done):  # 에피소드 종료 처리
                if d:  # 종료 시
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")  # 반환값 출력
                    avg_returns.append(info["r"][idx])  # 평균 반환값 추적
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)  # 평균 반환값 기록
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)  # 개별 반환값 기록
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)  # 에피소드 길이 기록
                    if np.average(avg_returns) > 17:  # 평균 반환값이 17 초과 시 학습 중단
                        writer.add_scalar("charts/time", time.time() - start_time, global_step)  # 학습 시간 기록
                        quit()

        # 이점 및 반환 계산
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)  # 다음 상태 가치
            if args.gae:  # GAE 사용 시
                advantages = torch.zeros_like(rewards).to(device)  # 이점 초기화
                lastgaelam = 0  # 마지막 GAE 람다 값
                for t in reversed(range(args.num_steps)):  # 역방향으로 계산
                    if t == args.num_steps - 1:  # 마지막 스텝
                        nextnonterminal = 1.0 - next_done  # 종료 여부
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]  # TD 오류
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam  # GAE 계산
                returns = advantages + values  # 반환값 = 이점 + 가치
            else:  # GAE 미사용 시
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return  # 기본 반환값 계산
                advantages = returns - values  # 이점 = 반환값 - 가치

        # 배치 평탄화 (미니배치 처리를 위해)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 정책 및 가치 네트워크 최적화
        b_inds = np.arange(args.batch_size)  # 배치 인덱스
        clipfracs = []  # 클리핑 비율 추적
        for epoch in range(args.update_epochs):  # 에포크별 업데이트
            np.random.shuffle(b_inds)  # 인덱스 섞기
            for start in range(0, args.batch_size, args.minibatch_size):  # 미니배치 단위로 처리
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]  # 미니배치 인덱스

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])  # 새 예측값
                logratio = newlogprob - b_logprobs[mb_inds]  # 로그 확률 비율
                ratio = logratio.exp()  # 확률 비율

                with torch.no_grad():  # KL 발산 계산 (로그 비율 기반)
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]  # 클리핑 비율 기록

                mb_advantages = b_advantages[mb_inds]  # 미니배치 이점
                if args.norm_adv:  # 이점 정규화
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 정책 손실 계산 (PPO 클리핑 적용)
                pg_loss1 = -mb_advantages * ratio  # 비클리핑 손실
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)  # 클리핑 손실
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # 최대 손실 선택

                # 가치 손실 계산
                newvalue = newvalue.view(-1)
                if args.clip_vloss:  # 가치 손실 클리핑
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()  # 엔트로피 손실
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # 총 손실 (정책 + 가치 + 엔트로피)

                optimizer.zero_grad()  # 그래디언트 초기화
                loss.backward()  # 역전파
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # 그래디언트 클리핑
                optimizer.step()  # 파라미터 업데이트

            if args.target_kl is not None and approx_kl > args.target_kl:  # KL 발산 초과 시 중단
                break

        # 학습 통계 계산 및 기록
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()  # 예측값과 실제 반환값
        var_y = np.var(y_true)  # 실제 반환값 분산
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # 설명된 분산

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)  # 학습률 기록
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)  # 가치 손실 기록
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)  # 정책 손실 기록
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)  # 엔트로피 기록
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)  # 이전 KL 발산 기록
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)  # 근사 KL 발산 기록
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)  # 클리핑 비율 기록
        writer.add_scalar("losses/explained_variance", explained_var, global_step)  # 설명된 분산 기록
        print("SPS:", int(global_step / (time.time() - start_time)))  # 초당 스텝 수 출력
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)  # SPS 기록

    envs.close()  # 환경 종료
    writer.close()  # TensorBoard 종료