"""Reinforcement learning mathematics and target computation for uno-ataraxos.

Provides:
- Exact VRPO Q-Boosting Expected SARSA(λ) with non-alternating Skip/Reverse handling
- Safe masked log-softmax, reverse KL, entropy, and uniform legal log probabilities
- Exponential moving average (EMAgnet) model updates
- WDL (Win-Draw-Loss) expected value and classification targets
- Device selection helper for Apple Silicon MPS
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Return Apple Silicon MPS device if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Compute log-softmax strictly over legal actions with float32 -1e9 masking."""
    neg_large = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
    masked = torch.where(legal_mask, logits, neg_large)
    return F.log_softmax(masked, dim=-1)


def masked_reverse_kl(
    logits: torch.Tensor,
    ref_log_probs: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Safe D_KL(pi || pi_ref) over legal actions only. Zero 0*(-inf) NaNs."""
    log_p = masked_log_softmax(logits, legal_mask)
    p = torch.where(legal_mask, log_p.exp(), torch.zeros_like(log_p))
    kl_terms = torch.where(legal_mask, p * (log_p - ref_log_probs), torch.zeros_like(log_p))
    return kl_terms.sum(dim=-1)


def masked_entropy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Safe entropy H(pi) over legal actions only."""
    log_p = masked_log_softmax(logits, legal_mask)
    p = torch.where(legal_mask, log_p.exp(), torch.zeros_like(log_p))
    ent_terms = torch.where(legal_mask, -p * log_p, torch.zeros_like(log_p))
    return ent_terms.sum(dim=-1)


def uniform_legal_log_probs(legal_mask: torch.Tensor) -> torch.Tensor:
    """Return log(1 / |A_legal|) for legal actions and 0.0 for illegal actions."""
    counts = legal_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    log_u = -counts.log().expand_as(legal_mask)
    return torch.where(legal_mask, log_u, torch.zeros_like(log_u))


@torch.no_grad()
def compute_vrpo_targets(
    rewards: torch.Tensor,          # [T, B] payoff for acting player at t (+1/0/-1 on terminal, else 0)
    dones: torch.Tensor,            # [T, B] float32: 1.0 if step t ended episode else 0.0
    same_player_next: torch.Tensor, # [T, B] float32: 1.0 if player[t+1] == player[t] (Skip/Reverse) else 0.0
    q_taken: torch.Tensor,          # [T, B] Q_critic(s_t, a_t) from acting player's view
    v_expected: torch.Tensor,       # [T+1, B] sum_a pi(a|o_t)*Q_critic(s_t, a), incl. bootstrap step T
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VRPO Q-Boosting Expected SARSA(λ) for two-player zero-sum games

    Correctly handles non-alternating extra turns (Skip/Reverse in 1v1 UNO).
    Sign rule:
    - If acting player gets an extra turn (same_player_next == 1.0): sign = +1.0
    - If turn passes to opponent (same_player_next == 0.0): sign = -1.0
    """
    T, B = rewards.shape
    q_targets = torch.empty_like(rewards)
    trace = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        sign = 2.0 * same_player_next[t] - 1.0  # +1 if extra turn, -1 if opponent acts next
        delta = rewards[t] + not_done * sign * v_expected[t + 1] - q_taken[t]
        trace = delta + not_done * lam * sign * trace
        q_targets[t] = (q_taken[t] + trace).clamp(-1.0, 1.0)

    advantages = q_targets - v_expected[:T]
    return q_targets, advantages


@torch.no_grad()
def update_ema(ema_model: nn.Module, online_model: nn.Module, decay: float) -> None:
    """In-place exponential moving average update of parameters and buffers.

    p_ema = decay * p_ema + (1.0 - decay) * p_online
    """
    for p_ema, p in zip(ema_model.parameters(), online_model.parameters()):
        p_ema.lerp_(p, 1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), online_model.buffers()):
        if b.dtype.is_floating_point:
            b_ema.lerp_(b, 1.0 - decay)
        else:
            b_ema.copy_(b)


def wdl_expected_value(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Compute expected scalar value in [-1, +1] from 3-class WDL logits (Loss=0, Draw=1, Win=2)."""
    probs = F.softmax(wdl_logits, dim=-1)
    return probs[..., 2] - probs[..., 0]


def reward_to_wdl_target(rewards: torch.Tensor) -> torch.Tensor:
    """Map scalar payoffs in {-1.0, 0.0, +1.0} to integer class indices {0, 1, 2}."""
    return (rewards.long() + 1).clamp(0, 2)
