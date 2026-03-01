import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# VAE for action modeling
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # mean + log_std
        )
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        stats = self.encoder(x)
        mean, log_std = stats.chunk(2, dim=-1)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mean, std
    
    def decode(self, state, z=None):
        if z is None:
            z = torch.randn(state.shape[0], 32).to(state.device)
        x = torch.cat([state, z], dim=1)
        return self.decoder(x)

# Perturbation network
class Perturbation(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Perturbation, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return 0.05 * self.net(x)  # small perturbation

# Q-function (standard critic)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q(x)

# BCQ agent skeleton
class BCQAgent:
    def __init__(self, state_dim, action_dim):
        self.vae = VAE(state_dim, action_dim)
        self.perturb = Perturbation(state_dim, action_dim)
        self.q1 = QNetwork(state_dim, action_dim)
        self.q2 = QNetwork(state_dim, action_dim)
        
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=1e-3)
        self.perturb_optimizer = torch.optim.Adam(self.perturb.parameters(), lr=1e-3)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        # sample plausible actions from VAE
        sampled_actions = []
        for _ in range(10):
            a = self.vae.decode(state)
            a = a + self.perturb(state, a)
            sampled_actions.append(a)
        # pick action with max Q
        q_values = torch.cat([self.q1(state, a) for a in sampled_actions], dim=0)
        best = sampled_actions[q_values.argmax()]
        return best.detach().cpu().numpy()

    # update functions are left for full implementation
    # but you would update:
    # 1. VAE reconstruction loss
    # 2. Q-function Bellman loss
    # 3. Perturbation to maximize Q
    # exactly following BCQ paper

