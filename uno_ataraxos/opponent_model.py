"""Online Bayesian Opponent Tracker for OOD-Robust Search (Milestone 8).

Strictly within-game, 100% tournament-fair:
Uses ONLY observed public actions and marginalized predictions from our own previous
turn's belief worlds. Never peeks at opponent hands or private state.

Tracks:
- weights: Posterior over [Style 0: Elite BP, Style 1: Aggressive No-VDraw, Style 2: Unpredictable High-Temp]
- ema_surprisal: Exponential moving average of -ln max(P_pred^(0)(a_opp), 0.01)
- belief_temp & uniform_mix: Dynamic Bayesian surprise tempering parameters for UnoBeliefNet.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch

DEFAULT_PRIOR: Tuple[float, float, float] = (0.50, 0.30, 0.20)


class OnlineOpponentTracker:
    """Tournament-fair within-game online Bayesian opponent style and surprise tracker."""

    def __init__(self, prior: Tuple[float, float, float] = DEFAULT_PRIOR):
        self.prior = list(prior)
        self.weights: List[float] = list(prior)
        self.ema_surprisal: float = 0.5

        # Cached predictions from our own previous search turn
        self._cached_turn: Optional[int] = None
        self._cached_action_probs: Optional[torch.Tensor] = None  # [3, 61] float tensor
        self._last_observed_turn: int = -1

    def reset(self) -> None:
        """Reset tracker state at the start of each game."""
        self.weights = list(self.prior)
        self.ema_surprisal = 0.5
        self._cached_turn = None
        self._cached_action_probs = None
        self._last_observed_turn = -1

    def cache_ply1_prediction(
        self,
        turn_after_my_move: int,
        probs_3x61: torch.Tensor,
    ) -> None:
        """Cache marginalized predicted opponent action probabilities under all 3 styles.

        Args:
            turn_after_my_move: Expected turn_count when opponent acts next.
            probs_3x61: Tensor of shape [3, 61] containing predicted probabilities.
        """
        self._cached_turn = turn_after_my_move
        self._cached_action_probs = probs_3x61.detach().cpu().float()

    def update_from_public_history(
        self,
        current_turn: int,
        history_tokens: torch.Tensor,
    ) -> None:
        """Update Bayesian style posterior and surprisal from observed public move.

        Verifies that an opponent move occurred right after our previous move,
        preventing corrupted updates when our own Skip or Reverse gave us an extra turn.

        Args:
            current_turn: Current game turn count.
            history_tokens: Tensor [B, 48, 5] or [48, 5] public history tokens.
        """
        if self._cached_action_probs is None or self._cached_turn is None:
            return

        # Ensure 2D tensor [48, 5]
        ht = history_tokens.squeeze(0) if history_tokens.ndim == 3 else history_tokens
        if ht.shape[0] == 0:
            return

        # Find the most recently recorded event in history tokens
        # Format: [actor_rel (0=me, 1=opp, 2=pad), action_id (0..61), ...]
        actor_rel = ht[:, 0].cpu().numpy()
        action_ids = ht[:, 1].cpu().numpy()

        # Valid non-pad entries
        valid_indices = [i for i, a in enumerate(actor_rel) if a != 2]
        if not valid_indices:
            self._cached_turn = None
            self._cached_action_probs = None
            return

        last_valid_idx = valid_indices[-1]
        last_actor = int(actor_rel[last_valid_idx])
        last_action = int(action_ids[last_valid_idx])

        # Skip/Reverse Extra Turn Invariant:
        # If the last recorded action was our own move (last_actor == 0), the opponent
        # did NOT act (we got an extra turn from Skip or Reverse). Do NOT consume cache yet!
        if last_actor != 1:
            return

        # Opponent acted: verify action is in valid range 0..60
        if 0 <= last_action <= 60 and current_turn != self._last_observed_turn:
            self._last_observed_turn = current_turn
            cached_probs = self._cached_action_probs  # [3, 61]

            # 1. Likelihood of observed action under each style
            L = [max(float(cached_probs[m, last_action].item()), 0.01) for m in range(3)]

            # 2. Update EMA surprisal with respect to Blueprint (Style 0)
            surprisal = -math.log(L[0])
            self.ema_surprisal = 0.65 * self.ema_surprisal + 0.35 * surprisal

            # 3. Update Bayesian posterior with 0.15 prior mixing floor (anti-baiting safety)
            # w_new[m] proportional to (0.85 * weights[m] + 0.15 * prior[m]) * L_m
            unnorm = [
                (0.85 * self.weights[m] + 0.15 * self.prior[m]) * L[m]
                for m in range(3)
            ]
            total_unnorm = sum(unnorm)
            if total_unnorm > 1e-9:
                w_new = [u / total_unnorm for u in unnorm]
            else:
                w_new = list(self.prior)

            # Clamp each w_new[m] >= 0.12 and renormalize
            clamped = [max(0.12, w) for w in w_new]
            sum_clamped = sum(clamped)
            self.weights = [c / sum_clamped for c in clamped]

        # Consume cache
        self._cached_turn = None
        self._cached_action_probs = None

    def get_belief_params(self) -> Tuple[float, float]:
        """Compute dynamic Bayesian surprise tempering parameters for UnoBeliefNet.

        Returns:
            belief_temp: float in [1.0, 1.6]
            uniform_mix: float in [0.03, 0.25]
        """
        # excess in [0.0, 1.0] when surprisal rises between 1.2 and 3.7
        excess = min(1.0, max(0.0, (self.ema_surprisal - 1.2) / 2.5))
        belief_temp = 1.0 + 0.6 * excess
        uniform_mix = 0.03 + 0.22 * excess
        return belief_temp, uniform_mix
