"""Duplicate-Deck (Mirrored) Evaluation Suite & Heuristic Baselines for 1v1 UNO.

Implements:
1. RandomBot: Uniform random choice among legal actions.
2. GreedyBot: Never draws voluntarily if a card is legal; prefers non-wilds over wilds.
3. ExpertHeuristicBot: Strong RULES.md 1v1 baseline with counter-stacking, color abundance
   tracking, skip/reverse priority for extra turns, and targeted wild/draw_4 usage.
4. evaluate_duplicate: Mirrored-deck evaluator running identical initial hands and draw piles
   with swapped seating order to neutralize card luck.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos.networks import UnoActorNet
from uno_ataraxos.rl_math import get_device, masked_log_softmax


class RandomBot:
    """Selects uniformly at random among all legal actions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act_batch(
        self,
        card_features: np.ndarray,
        global_features: np.ndarray,
        history_tokens: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        B = legal_mask.shape[0]
        actions = np.empty(B, dtype=np.int32)
        for i in range(B):
            legal_indices = np.where(legal_mask[i])[0]
            if len(legal_indices) == 0:
                actions[i] = 60  # fallback draw
            else:
                actions[i] = self.rng.choice(legal_indices)
        return actions

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.act_batch(
            obs["card_features_actor"],
            obs["global_features"],
            obs["history_tokens"],
            obs["legal_mask"],
        )


class GreedyBot:
    """Never draws voluntarily when a card is legal.

    Prefers non-wild card plays (0..51) over wilds (52..59).
    Picks uniformly among legal options within the preferred tier.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act_batch(
        self,
        card_features: np.ndarray,
        global_features: np.ndarray,
        history_tokens: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        B = legal_mask.shape[0]
        actions = np.empty(B, dtype=np.int32)
        for i in range(B):
            mask = legal_mask[i]
            # 1. Non-wild legal plays (0..51)
            non_wilds = np.where(mask[:52])[0]
            if len(non_wilds) > 0:
                actions[i] = self.rng.choice(non_wilds)
                continue

            # 2. Wild legal plays (52..59)
            wilds = np.where(mask[52:60])[0] + 52
            if len(wilds) > 0:
                actions[i] = self.rng.choice(wilds)
                continue

            # 3. Forced draw (60)
            actions[i] = 60
        return actions

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.act_batch(
            obs["card_features_actor"],
            obs["global_features"],
            obs["history_tokens"],
            obs["legal_mask"],
        )


class ExpertHeuristicBot:
    """Strong RULES.md 1v1 heuristic baseline.

    Features:
    1. Facing penalty stack: counters (+2 on +2, +4 on +4) into the color where it holds
       the most remaining cards.
    2. Normal turn: NEVER draws voluntarily when any card is legal.
       Priority:
       - Skip or Reverse (extra turn!): prefer matching active color or switching to most abundant color.
       - Draw 2: if opponent hand <= 3, or we hold backup +2, or have abundant cards of that color.
       - Wild Draw 4: if opponent hand <= 2 (punish immediately!), declaring most abundant color.
       - Colored numbers (0..39): prefer playing from most abundant color.
       - Wild (52..55): declare most abundant color (tiebreak: color with action cards).
       - Wild Draw 4 (56..59): last resort.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _color_abundance(self, my_hand: np.ndarray) -> np.ndarray:
        """Count total cards held in hand for each of the 4 colors {0:r, 1:g, 2:b, 3:y}."""
        counts = np.zeros(4, dtype=np.int32)
        for col in range(4):
            # 10 number cards: col * 10 .. col * 10 + 9
            counts[col] += int(np.sum(my_hand[col * 10 : col * 10 + 10]))
            # 3 action cards: 40 + col * 3 .. 40 + col * 3 + 2
            counts[col] += int(np.sum(my_hand[40 + col * 3 : 40 + col * 3 + 3]))
        return counts

    def _action_card_abundance(self, my_hand: np.ndarray) -> np.ndarray:
        """Count action cards (skip, reverse, draw_2) held in hand per color."""
        action_counts = np.zeros(4, dtype=np.int32)
        for col in range(4):
            action_counts[col] = int(np.sum(my_hand[40 + col * 3 : 40 + col * 3 + 3]))
        return action_counts

    def act_single(
        self,
        card_features: np.ndarray,  # [54, 7]
        global_features: np.ndarray,  # [10]
        legal_mask: np.ndarray,  # [61]
    ) -> int:
        my_hand = card_features[:, 0]  # raw count of cards in my hand
        stack_count = global_features[3] * 16.0
        is_p2_stack = global_features[4] > 0.5
        is_p4_stack = global_features[5] > 0.5
        opp_hand_size = int(round(global_features[1] * 20.0))

        color_counts = self._color_abundance(my_hand)
        action_counts = self._action_card_abundance(my_hand)

        # Best declared color: max color count, tiebreak by action card count, then color index
        best_col = int(np.lexsort((range(4), action_counts, color_counts))[-1])

        # ---------------------------------------------------------------------
        # Case A: Facing a penalty stack (stack > 0)
        # ---------------------------------------------------------------------
        if stack_count > 0:
            if is_p2_stack:
                legal_p2 = [a for a in [42, 45, 48, 51] if legal_mask[a]]
                if legal_p2:
                    # Pick +2 in color where we hold the most cards
                    return max(legal_p2, key=lambda a: color_counts[(a - 40) // 3])
            elif is_p4_stack:
                legal_p4 = [a for a in [56, 57, 58, 59] if legal_mask[a]]
                if legal_p4:
                    # Declare most abundant color
                    target_a = 56 + best_col
                    return target_a if legal_mask[target_a] else legal_p4[0]

            return 60  # Must draw stack

        # ---------------------------------------------------------------------
        # Case B: Normal turn (stack == 0) — Never draw if any card is legal
        # ---------------------------------------------------------------------
        if not legal_mask[:60].any():
            return 60  # Forced draw

        # Active color is given by card_features[:, 4] (which cards have is_active_color == 1)
        active_color_mask = card_features[:40, 4] > 0.5
        active_color = None
        if active_color_mask.any():
            active_color = int(np.where(active_color_mask)[0][0] // 10)

        # 1. Skip (40, 43, 46, 49) or Reverse (41, 44, 47, 50) -> Extra turn in 1v1!
        legal_extra_turns = [a for a in [40, 41, 43, 44, 46, 47, 49, 50] if legal_mask[a]]
        if legal_extra_turns:
            def score_extra_turn(a: int) -> int:
                c = (a - 40) // 3
                return color_counts[c] + (3 if c == active_color else 0)
            return max(legal_extra_turns, key=score_extra_turn)

        # 2. Draw 2: punish if opponent has <= 3 cards or if we have backup
        total_p2_held = sum(my_hand[c] for c in [42, 45, 48, 51])
        legal_p2 = [a for a in [42, 45, 48, 51] if legal_mask[a]]
        if legal_p2:
            best_p2 = max(legal_p2, key=lambda a: color_counts[(a - 40) // 3])
            p2_col = (best_p2 - 40) // 3
            if opp_hand_size <= 3 or total_p2_held >= 2 or color_counts[p2_col] >= 3:
                return best_p2

        # 3. Urgent Wild Draw 4: If opponent is at UNO (opp_hand_size <= 2), punish immediately!
        if opp_hand_size <= 2 and legal_mask[56:60].any():
            target_p4 = 56 + best_col
            if legal_mask[target_p4]:
                return target_p4
            return int(np.where(legal_mask[56:60])[0][0] + 56)

        # 4. Colored number cards (0..39): play from most abundant color
        legal_numbers = [a for a in range(40) if legal_mask[a]]
        if legal_numbers:
            # Score by count of that color, tiebreak by rank (shed higher rank cards)
            def score_num(a: int) -> tuple[int, int]:
                c = a // 10
                rank = a % 10
                return (color_counts[c], rank)
            return max(legal_numbers, key=score_num)

        # 5. Play Draw 2 if we passed on Priority 2 earlier but have no numbers
        if legal_p2:
            return max(legal_p2, key=lambda a: color_counts[(a - 40) // 3])

        # 6. Wild (52..55): declare best color
        if legal_mask[52:56].any():
            target_wild = 52 + best_col
            if legal_mask[target_wild]:
                return target_wild
            return int(np.where(legal_mask[52:56])[0][0] + 52)

        # 7. Wild Draw 4 (56..59) as last resort
        if legal_mask[56:60].any():
            target_p4 = 56 + best_col
            if legal_mask[target_p4]:
                return target_p4
            return int(np.where(legal_mask[56:60])[0][0] + 56)

        # Fallback to any legal card play
        any_card = np.where(legal_mask[:60])[0]
        if len(any_card) > 0:
            return int(any_card[0])
        return 60

    def act_batch(
        self,
        card_features: np.ndarray,
        global_features: np.ndarray,
        history_tokens: np.ndarray,
        legal_mask: np.ndarray,
    ) -> np.ndarray:
        B = legal_mask.shape[0]
        actions = np.empty(B, dtype=np.int32)
        for i in range(B):
            actions[i] = self.act_single(card_features[i], global_features[i], legal_mask[i])
        return actions

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.act_batch(
            obs["card_features_actor"],
            obs["global_features"],
            obs["history_tokens"],
            obs["legal_mask"],
        )


def _get_policy_action_batch(
    policy: Any,
    obs: Dict[str, np.ndarray],
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    """Get batched actions from either a Bot instance or a PyTorch UnoActorNet."""
    if isinstance(policy, (RandomBot, GreedyBot, ExpertHeuristicBot)):
        return policy(obs)

    # PyTorch neural policy (e.g. UnoActorNet)
    if isinstance(policy, torch.nn.Module):
        card_feats = torch.from_numpy(obs["card_features_actor"]).to(device)
        global_feats = torch.from_numpy(obs["global_features"]).to(device)
        hist_tokens = torch.from_numpy(obs["history_tokens"]).to(device)
        legal_mask = torch.from_numpy(obs["legal_mask"]).to(device)

        device_type = "mps" if device.type == "mps" else ("cuda" if device.type == "cuda" else "cpu")
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            policy_logits, _ = policy(card_feats, global_feats, hist_tokens)

        # Float32 masked log-softmax
        log_probs = masked_log_softmax(policy_logits.float(), legal_mask)
        if temperature <= 1e-4:
            # Greedy argmax over legal actions
            actions = log_probs.argmax(dim=-1).cpu().numpy().astype(np.int32)
        else:
            probs = torch.where(legal_mask, (log_probs / temperature).exp(), torch.zeros_like(log_probs))
            # Normalize probs to prevent sum rounding errors
            prob_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            probs = probs / prob_sum
            actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy().astype(np.int32)
        return actions

    if callable(policy):
        return policy(obs)

    raise TypeError(f"Unsupported policy type: {type(policy)}")


def evaluate_duplicate(
    policy_a: Any,
    policy_b: Any,
    num_decks: int = 250,
    base_seed: int = 10000,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Mirrored-deck (duplicate) evaluation between policy_a and policy_b.

    Runs 2 * num_decks total games on identical initial decks with swapped seating:
      - Game 1 (num_decks games): policy_a = Seat 0, policy_b = Seat 1
      - Game 2 (num_decks games): policy_b = Seat 0, policy_a = Seat 1

    Returns dict with win_rate_a, wins, draws, losses, avg_turns, voluntary_draw_rate_a.
    """
    if device is None:
        device = get_device()

    total_games = 2 * num_decks
    wins_a = 0
    draws = 0
    losses_a = 0
    total_turns_completed = 0

    v_draws_a = 0
    total_moves_a = 0

    # -------------------------------------------------------------------------
    # Batch 1: policy_a = Seat 0, policy_b = Seat 1
    # -------------------------------------------------------------------------
    vec1 = ua.VecUnoEnv(num_envs=num_decks, seed=base_seed, auto_reset=False)
    obs1 = vec1.reset()
    active_envs_1 = np.ones(num_decks, dtype=bool)
    game1_turns = np.zeros(num_decks, dtype=np.int32)

    while active_envs_1.any():
        cur_players = obs1["current_player"]  # [num_decks]
        legal_mask = obs1["legal_mask"]
        actions = np.empty(num_decks, dtype=np.int32)

        # Envs where Seat 0 acts (policy_a)
        mask_a = active_envs_1 & (cur_players == 0)
        if mask_a.any():
            sub_obs_a = {k: v[mask_a] for k, v in obs1.items()}
            sub_act_a = _get_policy_action_batch(policy_a, sub_obs_a, device, temperature)
            actions[mask_a] = sub_act_a

            # Track voluntary draw rate for policy_a
            sub_legal = legal_mask[mask_a]
            v_draws_a += int(np.sum((sub_act_a == 60) & sub_legal[:, :60].any(axis=1)))
            total_moves_a += int(np.sum(mask_a))

        # Envs where Seat 1 acts (policy_b)
        mask_b = active_envs_1 & (cur_players == 1)
        if mask_b.any():
            sub_obs_b = {k: v[mask_b] for k, v in obs1.items()}
            sub_act_b = _get_policy_action_batch(policy_b, sub_obs_b, device, temperature)
            actions[mask_b] = sub_act_b

        # Step active envs (inactive envs can send dummy action 60)
        actions[~active_envs_1] = 60
        next_obs1, rewards1, dones1, _ = vec1.step_no_autoreset(actions)
        game1_turns[active_envs_1] += 1

        # Check newly finished envs
        newly_done = active_envs_1 & dones1
        for idx in np.where(newly_done)[0]:
            acting_p = cur_players[idx]
            r = rewards1[idx]
            if r == 1.0:
                winner = acting_p
            elif r == -1.0:
                winner = 1 - acting_p
            else:
                winner = -1  # draw

            if winner == 0:
                wins_a += 1
            elif winner == 1:
                losses_a += 1
            else:
                draws += 1
            total_turns_completed += int(game1_turns[idx])

        active_envs_1[newly_done] = False
        obs1 = next_obs1

    # -------------------------------------------------------------------------
    # Batch 2: policy_b = Seat 0, policy_a = Seat 1 (exact same deck seeds!)
    # -------------------------------------------------------------------------
    vec2 = ua.VecUnoEnv(num_envs=num_decks, seed=base_seed, auto_reset=False)
    obs2 = vec2.reset()
    active_envs_2 = np.ones(num_decks, dtype=bool)
    game2_turns = np.zeros(num_decks, dtype=np.int32)

    while active_envs_2.any():
        cur_players = obs2["current_player"]  # [num_decks]
        legal_mask = obs2["legal_mask"]
        actions = np.empty(num_decks, dtype=np.int32)

        # Envs where Seat 0 acts (policy_b)
        mask_b = active_envs_2 & (cur_players == 0)
        if mask_b.any():
            sub_obs_b = {k: v[mask_b] for k, v in obs2.items()}
            sub_act_b = _get_policy_action_batch(policy_b, sub_obs_b, device, temperature)
            actions[mask_b] = sub_act_b

        # Envs where Seat 1 acts (policy_a)
        mask_a = active_envs_2 & (cur_players == 1)
        if mask_a.any():
            sub_obs_a = {k: v[mask_a] for k, v in obs2.items()}
            sub_act_a = _get_policy_action_batch(policy_a, sub_obs_a, device, temperature)
            actions[mask_a] = sub_act_a

            # Track voluntary draw rate for policy_a
            sub_legal = legal_mask[mask_a]
            v_draws_a += int(np.sum((sub_act_a == 60) & sub_legal[:, :60].any(axis=1)))
            total_moves_a += int(np.sum(mask_a))

        actions[~active_envs_2] = 60
        next_obs2, rewards2, dones2, _ = vec2.step_no_autoreset(actions)
        game2_turns[active_envs_2] += 1

        newly_done = active_envs_2 & dones2
        for idx in np.where(newly_done)[0]:
            acting_p = cur_players[idx]
            r = rewards2[idx]
            if r == 1.0:
                winner = acting_p
            elif r == -1.0:
                winner = 1 - acting_p
            else:
                winner = -1  # draw

            if winner == 1:
                wins_a += 1  # policy_a is seat 1
            elif winner == 0:
                losses_a += 1  # policy_b is seat 0
            else:
                draws += 1
            total_turns_completed += int(game2_turns[idx])

        active_envs_2[newly_done] = False
        obs2 = next_obs2

    win_rate_a = (wins_a + 0.5 * draws) / total_games
    avg_turns = total_turns_completed / total_games
    voluntary_draw_rate_a = v_draws_a / max(1, total_moves_a)

    return {
        "win_rate_a": float(win_rate_a),
        "wins": int(wins_a),
        "draws": int(draws),
        "losses": int(losses_a),
        "avg_turns": float(avg_turns),
        "voluntary_draw_rate_a": float(voluntary_draw_rate_a),
    }
