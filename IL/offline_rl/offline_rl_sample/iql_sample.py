import torch
import torch.nn as nn
import torch.nn.functional as F

# 기본 MLP 네트워크
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Expectile Loss
def expectile_loss(diff, tau=0.7):
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)

# Q, V, Policy 네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1)
    def forward(self, s, a):
        return self.q(torch.cat([s, a], dim=-1)).squeeze(-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.v = MLP(state_dim, 1)
    def forward(self, s):
        return self.v(s).squeeze(-1)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mean = MLP(state_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self, s):
        return self.mean(s), torch.exp(self.log_std)
    def sample(self, s):
        m, std = self.forward(s)
        dist = torch.distributions.Normal(m, std)
        a = dist.rsample()
        return a, dist.log_prob(a).sum(-1)

# IQL 학습 루프 (1 step)
def iql_train_step(batch, qnet, vnet, pi, q_opt, v_opt, pi_opt, gamma=0.99, tau=0.7, beta=3.0):
    s, a, r, s_next, d = batch

    # Q 업데이트
    with torch.no_grad():
        v_next = vnet(s_next)
        target = r + gamma * (1 - d) * v_next
    q_pred = qnet(s, a)
    q_loss = F.mse_loss(q_pred, target)

    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()

    # V 업데이트 (expectile)
    with torch.no_grad():
        q_val = qnet(s, a)
    v_pred = vnet(s)
    v_loss = expectile_loss(q_val - v_pred, tau).mean()

    v_opt.zero_grad()
    v_loss.backward()
    v_opt.step()

    # Policy 업데이트
    with torch.no_grad():
        adv = q_val - v_pred
        w = torch.exp(adv / beta).clamp(max=100.0)

    m, std = pi(s)
    dist = torch.distributions.Normal(m, std)
    logp = dist.log_prob(a).sum(-1)
    pi_loss = -(w * logp).mean()

    pi_opt.zero_grad()
    pi_loss.backward()
    pi_opt.step()

    return q_loss.item(), v_loss.item(), pi_loss.item()
