import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------------------------------
# BEAR 핵심 네트워크: Q-function, Policy, Behavior(VAE)
# -----------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

class BehaviorVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=2):
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, s, a):
        # encode
        x = torch.cat([s, a], dim=-1)
        h = self.encoder(x)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(mean)
        # decode
        recon_a = self.decoder(torch.cat([s, z], dim=-1))
        return recon_a, mean, std

# -----------------------------------------------------
# MMD distance helper
# -----------------------------------------------------
def mmd_distance(samples1, samples2, kernel='gaussian', sigma=0.2):
    # samples1, samples2 shape: [batch, action_dim]
    xx = torch.mm(samples1, samples1.t())
    yy = torch.mm(samples2, samples2.t())
    xy = torch.mm(samples1, samples2.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2*xx
    dyy = ry.t() + ry - 2*yy
    dxy = rx.t() + ry - 2*xy
    Kxx = torch.exp(-dxx / (2*sigma**2))
    Kyy = torch.exp(-dyy / (2*sigma**2))
    Kxy = torch.exp(-dxy / (2*sigma**2))
    mmd = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return mmd

# -----------------------------------------------------
# 학습 루프 (핵심 아이디어만)
# -----------------------------------------------------
state_dim, action_dim = 3, 1
qf = QNetwork(state_dim, action_dim)
policy = Policy(state_dim, action_dim)
vae = BehaviorVAE(state_dim, action_dim)
qf_optimizer = optim.Adam(qf.parameters(), lr=1e-3)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)

alpha = 10.0  # MMD penalty weight

for _ in range(10000):
    # 예시 배치 (dummy)
    s = torch.randn(256, state_dim)
    a = torch.randn(256, action_dim)
    r = torch.randn(256, 1)
    s2 = torch.randn(256, state_dim)
    done = torch.zeros(256, 1)

    # --- Q update ---
    q_target = r + 0.99 * qf(s2, policy(s2)) * (1 - done)
    q_loss = nn.MSELoss()(qf(s, a), q_target.detach())
    qf_optimizer.zero_grad()
    q_loss.backward()
    qf_optimizer.step()

    # --- Policy update ---
    # 샘플 행동
    new_actions = policy(s)
    # 데이터셋 행동 분포
    vae_recon, _, _ = vae(s, a)
    # MMD
    mmd = mmd_distance(new_actions, vae_recon)
    # Actor loss
    policy_loss = -qf(s, new_actions).mean() + alpha * mmd
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

