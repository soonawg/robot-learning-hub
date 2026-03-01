import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.logits = nn.Linear(state_dim, action_dim)

    def forward(self, state):
        return F.softmax(self.logits(state), dim=-1)

# CQL Loss 계산
def compute_cql_loss(q_net, q_target, policy, batch, alpha=1.0, gamma=0.99):
    state, action, reward, next_state, done = batch
    batch_size, action_dim = state.size(0), policy.logits.out_features

    # One-hot action
    action_onehot = F.one_hot(action, action_dim).float()
    q_value = q_net(state, action_onehot)

    with torch.no_grad():
        next_probs = policy(next_state)
        all_next_actions = torch.eye(action_dim).unsqueeze(0).repeat(batch_size, 1, 1)
        next_states_expanded = next_state.unsqueeze(1).repeat(1, action_dim, 1)
        next_q_inputs = torch.cat([next_states_expanded, all_next_actions], dim=-1)
        next_q_values = q_target(next_q_inputs.view(-1, next_q_inputs.size(-1))).view(batch_size, action_dim)
        target_q = (next_probs * next_q_values).sum(dim=1, keepdim=True)
        backup = reward + gamma * (1 - done) * target_q

    td_loss = F.mse_loss(q_value, backup)

    # Conservative regularization
    all_actions = torch.eye(action_dim).unsqueeze(0).repeat(batch_size, 1, 1)
    states_exp = state.unsqueeze(1).repeat(1, action_dim, 1)
    q_inputs_all = torch.cat([states_exp, all_actions], dim=-1)
    q_values_all = q_net(q_inputs_all.view(-1, q_inputs_all.size(-1))).view(batch_size, action_dim)
    logsumexp = torch.logsumexp(q_values_all, dim=1, keepdim=True)

    cql_loss = (logsumexp - q_value).mean()
    total_loss = td_loss + alpha * cql_loss
    return total_loss
