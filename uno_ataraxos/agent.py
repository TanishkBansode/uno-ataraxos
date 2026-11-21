"""Deployment Wrapper: AtaraxosUnoAgent (Milestone 5).

Unified deployment agent operating in three selectable modes:
1. "blueprint"  - Pure neural blueprint policy (zero search, ~0.8 ms).
2. "mmd_search" - Tier-1 Update-Equivalence Tabular MMD Search.
3. "hybrid"     - Tier-1 MMD Search + Tier-2 Low-Card Endgame PDCFR+ solver.

Features:
- Native fast C++ evaluation via act_from_engine(engine_or_vec).
- STRICT TOURNAMENT FAIRNESS in act_uno_gym(env_or_sim, player_id):
  NEVER peeks at sim.hands[opp] or sim.deck identities!
- Live Belief X-Ray diagnostics tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import uno_ataraxos as ua
from uno_ataraxos import UnoEngine, VecUnoEnv
from uno_ataraxos.belief import UnoBeliefNet
from uno_ataraxos.endgame_cfr import solve_endgame_pdcfr
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import (
    get_device,
    masked_log_softmax,
    uniform_legal_log_probs,
)
from uno_ataraxos.search import SearchResult, ataraxos_mmd_search


class AtaraxosUnoAgent:
    """Unified Ataraxos 1v1 UNO Tournament Agent."""

    def __init__(
        self,
        blueprint_ckpt: Union[str, Path, dict],
        belief_ckpt: Optional[Union[str, Path, dict]] = None,
        critic_ckpt: Optional[Union[str, Path, dict]] = None,
        mode: str = "hybrid",
        device: Optional[torch.device] = None,
        K: int = 128,
        depth: int = 12,
        eta: float = 3.0,
        tau: float = 0.05,
        iterations: int = 150,
        M_max: int = 16,
        max_total_worlds: int = 768,
        leaf_eval: str = "critic",
    ):
        if mode not in ("blueprint", "mmd_search", "hybrid"):
            raise ValueError(f"Unknown mode: {mode}. Choose from ('blueprint', 'mmd_search', 'hybrid')")

        self.mode = mode
        self.device = device or get_device()
        self.K = K
        self.depth = depth
        self.eta = eta
        self.tau = tau
        self.iterations = iterations
        self.M_max = M_max
        self.max_total_worlds = max_total_worlds
        self.leaf_eval = leaf_eval

        # 1. Load Blueprint Actor
        self.actor_net = UnoActorNet().to(self.device)
        if isinstance(blueprint_ckpt, (str, Path)):
            ckpt = torch.load(blueprint_ckpt, map_location=self.device)
        else:
            ckpt = blueprint_ckpt

        actor_state = ckpt.get("actor_state_dict", ckpt.get("actor", ckpt))
        self.actor_net.load_state_dict(actor_state)
        self.actor_net.eval()

        # 2. Load Belief Net (if needed for search)
        self.belief_net: Optional[UnoBeliefNet] = None
        if self.mode in ("mmd_search", "hybrid"):
            if belief_ckpt is not None:
                self.belief_net = UnoBeliefNet().to(self.device)
                if isinstance(belief_ckpt, (str, Path)):
                    bel_ckpt = torch.load(belief_ckpt, map_location=self.device)
                else:
                    bel_ckpt = belief_ckpt
                bel_state = bel_ckpt.get("belief_net", bel_ckpt.get("model_state_dict", bel_ckpt))
                self.belief_net.load_state_dict(bel_state)
                self.belief_net.eval()
            else:
                raise ValueError(f"Mode '{mode}' requires belief_ckpt for opponent hand sampling!")

        # 3. Load Optional Critic Net
        self.critic_net: Optional[UnoQCriticNet] = None
        if critic_ckpt is not None:
            self.critic_net = UnoQCriticNet().to(self.device)
            if isinstance(critic_ckpt, (str, Path)):
                crit_data = torch.load(critic_ckpt, map_location=self.device)
            else:
                crit_data = critic_ckpt
            crit_state = crit_data.get("critic_state_dict", crit_data.get("critic", crit_data))
            self.critic_net.load_state_dict(crit_state)
            self.critic_net.eval()

        # Persistent C++ Rollout Pool for Search
        self.rollout_pool: Optional[VecUnoEnv] = None
        self.internal_eng: UnoEngine = UnoEngine(42)

        # Move & Provenance History Tracker
        self.last_result: Optional[SearchResult] = None
        self._last_discard_len: int = 0

    def reset(self):
        """Reset internal history and agent state for a new game."""
        self.last_result = None
        self._last_discard_len = 0
        self.internal_eng.reset()

    def act_from_engine(
        self,
        engine_or_vec: Union[UnoEngine, VecUnoEnv],
        env_idx: int = 0,
        deterministic: bool = False,
    ) -> Tuple[int, Optional[SearchResult]]:
        """Act directly on a C++ UnoEngine or VecUnoEnv.

        Args:
            engine_or_vec: UnoEngine instance or VecUnoEnv.
            env_idx: Index of environment if VecUnoEnv is passed.
            deterministic: If True, selects argmax action.

        Returns:
            (chosen_action, search_result_or_none)
        """
        if isinstance(engine_or_vec, VecUnoEnv):
            eng = engine_or_vec.get_engine(env_idx)
        else:
            eng = engine_or_vec

        curr_p = int(eng.current_player)
        legal_actions = eng.get_legal_action_ids(curr_p)

        if len(legal_actions) == 0:
            return 60, None
        if len(legal_actions) == 1:
            act = int(legal_actions[0])
            return act, None

        # Mode 1: Pure Blueprint
        if self.mode == "blueprint":
            cf = torch.from_numpy(eng.get_card_features_actor(curr_p)).unsqueeze(0).to(self.device)
            gf = torch.from_numpy(eng.get_global_features(curr_p)).unsqueeze(0).to(self.device)
            ht = torch.from_numpy(eng.get_history_tokens(curr_p)).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(eng.get_action_mask(curr_p)).to(self.device)

            device_type = "mps" if self.device.type == "mps" else ("cuda" if self.device.type == "cuda" else "cpu")
            with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = self.actor_net(cf, gf, ht)

            log_p = masked_log_softmax(logits.float().squeeze(0), mask_t)
            probs = torch.where(mask_t, log_p.exp(), torch.zeros_like(log_p))

            if deterministic:
                chosen = int(probs.argmax().item())
            else:
                chosen = int(torch.multinomial(probs, 1).item())

            self.last_result = None
            return chosen, None

        # Mode 2 or 3: Search
        # Check Tier-2 low-card endgame trigger
        my_hand_size = len(eng.get_hand_ids(curr_p))
        opp_hand_size = len(eng.get_hand_ids(1 - curr_p))
        num_legal = len(legal_actions)

        is_endgame = (
            self.mode == "hybrid"
            and my_hand_size <= 3
            and opp_hand_size <= 3
            and num_legal >= 2
        )

        if is_endgame:
            res = solve_endgame_pdcfr(
                root_engine=eng,
                actor_net=self.actor_net,
                belief_net=self.belief_net,
                q_critic_net=self.critic_net,
                K=self.K,
                M_max=self.M_max,
                iterations=self.iterations,
                deterministic=deterministic,
            )
        else:
            # Reusable rollout pool with adaptive world budget
            K_eff = min(self.K, max(32, self.max_total_worlds // max(1, num_legal)))
            needed_slots = num_legal * K_eff
            if self.rollout_pool is None:
                self.rollout_pool = VecUnoEnv(num_envs=needed_slots, seed=42, auto_reset=False)
            elif self.rollout_pool.num_envs != needed_slots:
                self.rollout_pool.resize(needed_slots)

            res = ataraxos_mmd_search(
                root_engine=eng,
                actor_net=self.actor_net,
                belief_net=self.belief_net,
                q_critic_net=self.critic_net,
                rollout_pool=self.rollout_pool,
                K=self.K,
                depth=self.depth,
                eta=self.eta,
                tau=self.tau,
                max_total_worlds=self.max_total_worlds,
                leaf_eval=self.leaf_eval,
                deterministic=deterministic,
            )

        self.last_result = res
        return res.action, res

    def act_uno_gym(
        self,
        env_or_sim: Any,
        player_id: int,
        deterministic: bool = False,
    ) -> int:
        """Tournament-Fair uno_gym Action Selector.

        STRICT FAIRNESS GUARANTEE:
        Reads ONLY:
          - sim.hands[player_id] (our own private hand)
          - len(sim.hands[1 - player_id]) (opponent card count ONLY)
          - len(sim.deck) (draw pile count ONLY)
          - sim.target, sim.stack, sim.discard_pile, sim.current_player, sim.turn_count
        NEVER peeks at opponent's card identities or draw pile card identities!

        Args:
            env_or_sim: uno_gym.UnoEnv or uno_gym.UnoSim.
            player_id: 0 or 1.
            deterministic: If True, selects argmax action.

        Returns:
            action_id: int (0..60).
        """
        sim = getattr(env_or_sim, "sim", env_or_sim)

        # 1. Strictly Fair Public/Private State Extraction
        my_hand_str = list(sim.hands[player_id])
        opp_hand_len = len(sim.hands[1 - player_id])
        deck_len = len(sim.deck)
        discard_pile_str = list(sim.discard_pile)
        target_str = str(sim.target)
        stack = int(sim.stack)
        current_player = int(sim.current_player)
        turn_count = int(sim.turn_count)
        consecutive_passes = int(getattr(sim, "consecutive_passes", 0))

        # Color mapping
        col_map = {"r": 0, "g": 1, "b": 2, "y": 3}
        active_color = col_map.get(target_str[0], 0) if len(target_str) > 0 else 0

        # Construct dummy placeholders for opponent hand and deck
        dummy_opp_hand = ["r-0"] * opp_hand_len
        dummy_deck = ["r-0"] * max(0, deck_len)

        h0 = my_hand_str if player_id == 0 else dummy_opp_hand
        h1 = dummy_opp_hand if player_id == 0 else my_hand_str

        # Load into internal UnoEngine
        self.internal_eng.load_state(
            h0,
            h1,
            dummy_deck,
            discard_pile_str,
            target_str,
            active_color,
            stack,
            current_player,
            turn_count,
            consecutive_passes,
        )

        # 2. Select Action via act_from_engine
        chosen_action, _ = self.act_from_engine(self.internal_eng, deterministic=deterministic)
        return chosen_action

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Batch callable interface for eval_suite.py compatibility."""
        cf = torch.from_numpy(obs["card_features_actor"]).to(self.device)
        gf = torch.from_numpy(obs["global_features"]).to(self.device)
        ht = torch.from_numpy(obs["history_tokens"]).to(self.device)
        mask = torch.from_numpy(obs["legal_mask"]).to(self.device)

        device_type = "mps" if self.device.type == "mps" else ("cuda" if self.device.type == "cuda" else "cpu")
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, _ = self.actor_net(cf, gf, ht)

        log_p = masked_log_softmax(logits.float(), mask)
        probs = torch.where(mask, log_p.exp(), torch.zeros_like(log_p))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        sampled = torch.multinomial(probs, 1).squeeze(-1)
        return sampled.cpu().numpy().astype(np.int32)
