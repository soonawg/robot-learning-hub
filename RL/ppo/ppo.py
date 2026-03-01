import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 명령줄 인자를 파싱하는 함수: 학습 설정을 정의
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="실험 이름")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="사용할 Gym 환경 ID")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="최적화기의 학습률")
    parser.add_argument("--seed", type=int, default=1,
        help="실험의 시드 (재현성 보장)")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="총 학습 타임스텝 수")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="torch의 결정론적 설정 여부")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="CUDA 사용 여부")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Weights & Biases로 실험 추적 여부")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="wandb 프로젝트 이름")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="wandb 팀 이름")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="에이전트 성능 비디오 녹화 여부")

    # PPO 알고리즘 관련 인자
    parser.add_argument("--num-envs", type=int, default=4,
        help="병렬 환경 수")
    parser.add_argument("--num-steps", type=int, default=128,
        help="각 롤아웃에서 환경별 스텝 수")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="학습률 점진적 감소 여부")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="GAE(Generalized Advantage Estimation) 사용 여부")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="할인율 (미래 보상의 중요도)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="GAE의 람다 파라미터")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="미니배치 수")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="정책 업데이트 에포크 수")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="이점(advantage) 정규화 여부")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="클리핑 계수 (정책 업데이트 안정성)")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="가치 함수 손실 클리핑 여부")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="엔트로피 계수 (탐험 촉진)")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="가치 함수 계수")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="그래디언트 클리핑 최대 노름")
    parser.add_argument("--target-kl", type=float, default=None,
        help="KL 발산 목표치 (조기 중단 기준)")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # 배치 크기 계산
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 미니배치 크기 계산
    return args

# 환경 생성 함수: Gym 환경을 설정하고 초기화
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)  # 환경 생성
        env = gym.wrappers.RecordEpisodeStatistics(env)  # 에피소드 통계 기록
        if capture_video and idx == 0:  # 비디오 녹화 (첫 번째 환경만)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)  # 환경 시드 설정
        env.action_space.seed(seed)  # 행동 공간 시드
        env.observation_space.seed(seed)  # 관측 공간 시드
        return env
    return thunk

# 신경망 레이어 초기화 함수: 가중치와 편향을 설정
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # 직교 초기화
    torch.nn.init.constant_(layer.bias, bias_const)  # 편향 상수 초기화
    return layer

# 에이전트 클래스: Actor와 Critic 신경망 정의
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        # Critic: 상태 → 가치 예측
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor: 상태 → 행동 확률 분포
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):  # 상태의 가치 반환
        return self.critic(x)

    def get_action_and_value(self, x, action=None):  # 행동과 가치 계산
        logits = self.actor(x)  # 행동 logits 계산
        probs = Categorical(logits=logits)  # 확률 분포 생성
        if action is None:
            action = probs.sample()  # 행동 샘플링
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)  # 행동, 로그 확률, 엔트로피, 가치 반환

if __name__ == "__main__":
    args = parse_args()  # 인자 파싱
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"  # 실험 이름 생성
    if args.track:  # Weights & Biases로 추적 설정
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
    writer = SummaryWriter(f"runs/{run_name}")  # TensorBoard 기록 설정
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # 시드 설정: 재현성 보장
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 장치 설정: GPU 사용 가능 시 CUDA, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 병렬 환경 설정
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "이 코드는 이산 행동 공간만 지원합니다"

    # 에이전트와 최적화기 초기화
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 데이터 저장용 텐서 준비
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # 관측
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)  # 행동
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 로그 확률
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 보상
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 종료 여부
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # 가치

    # 학습 시작
    global_step = 0  # 전체 스텝 수
    start_time = time.time()  # 시작 시간
    next_obs = torch.Tensor(envs.reset()).to(device)  # 초기 관측
    next_done = torch.zeros(args.num_envs).to(device)  # 초기 종료 여부
    num_updates = args.total_timesteps // args.batch_size  # 총 업데이트 횟수

    for update in range(1, num_updates + 1):  # 학습 반복
        # 학습률 점진적 감소
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):  # 각 롤아웃 스텝
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # 행동 선택
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # 환경 진행
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # 에피소드 통계 기록
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # 가치 부트스트래핑
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:  # GAE로 이점 계산
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:  # 기본 반환 계산
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # 배치 평탄화: 학습에 사용할 데이터 준비
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 정책 및 가치 네트워크 최적화
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # KL 발산 계산
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:  # 이점 정규화
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 정책 손실 계산
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 가치 손실 계산
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()  # 엔트로피 손실
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # 총 손실

                # 최적화
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:  # KL 발산 초과 시 조기 중단
                break

        # 설명된 분산 계산
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 학습 진행 상황 기록
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