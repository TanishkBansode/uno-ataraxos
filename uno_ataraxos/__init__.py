"""uno-ataraxos: High-performance C++17 1v1 UNO simulation & vectorized environment."""

from __future__ import annotations

import numpy as np
from uno_ataraxos._cpp import (
    UnoEngine,
    VecUnoEnv,
    NUM_ACTIONS,
    NUM_CARD_TYPES,
    TOTAL_DECK_CARDS,
    OBS_DIM,
    MAX_TURNS,
    get_card_names,
    get_action_names,
    card_name_to_id,
    action_name_to_id,
)

CARD_KEYS: tuple[str, ...] = tuple(get_card_names())
ALL_ACTIONS: tuple[str, ...] = tuple(get_action_names())

ACTION_TO_ID: dict[str, int] = {act: idx for idx, act in enumerate(ALL_ACTIONS)}
ID_TO_ACTION: dict[int, str] = {idx: act for idx, act in enumerate(ALL_ACTIONS)}

CARD_TO_ID: dict[str, int] = {card: idx for idx, card in enumerate(CARD_KEYS)}
ID_TO_CARD: dict[int, str] = {idx: card for idx, card in enumerate(CARD_KEYS)}


def action_to_id(action: str) -> int:
    return ACTION_TO_ID[action]


def id_to_action(action_id: int) -> str:
    return ID_TO_ACTION[action_id]


def card_to_id(card: str) -> int:
    return CARD_TO_ID[card]


def id_to_card(card_id: int) -> str:
    return ID_TO_CARD[card_id]


from uno_ataraxos.symmetry import (
    COLOR_PERMUTATIONS,
    NUM_PERMUTATIONS,
    FWD_CARD_54,
    INV_CARD_54,
    FWD_ACTION_62,
    INV_ACTION_61,
    FWD_COLOR_5,
    INV_COLOR_5,
    INV_PERM_ID,
    get_inverse_perm_id,
    apply_s4_batch,
)
from uno_ataraxos.rl_math import (
    get_device,
    masked_log_softmax,
    masked_reverse_kl,
    masked_entropy,
    uniform_legal_log_probs,
    compute_vrpo_targets,
    update_ema,
    wdl_expected_value,
    reward_to_wdl_target,
)
from uno_ataraxos.networks import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    UnoBackbone,
    UnoActorNet,
    UnoQCriticNet,
)
from uno_ataraxos.eval_suite import (
    RandomBot,
    GreedyBot,
    ExpertHeuristicBot,
    evaluate_duplicate,
)

__all__ = [
    "UnoEngine",
    "VecUnoEnv",
    "NUM_ACTIONS",
    "NUM_CARD_TYPES",
    "TOTAL_DECK_CARDS",
    "OBS_DIM",
    "MAX_TURNS",
    "CARD_KEYS",
    "ALL_ACTIONS",
    "ACTION_TO_ID",
    "ID_TO_ACTION",
    "CARD_TO_ID",
    "ID_TO_CARD",
    "action_to_id",
    "id_to_action",
    "card_to_id",
    "id_to_card",
    # Symmetry
    "COLOR_PERMUTATIONS",
    "NUM_PERMUTATIONS",
    "FWD_CARD_54",
    "INV_CARD_54",
    "FWD_ACTION_62",
    "INV_ACTION_61",
    "FWD_COLOR_5",
    "INV_COLOR_5",
    "INV_PERM_ID",
    "get_inverse_perm_id",
    "apply_s4_batch",
    # RL Math
    "get_device",
    "masked_log_softmax",
    "masked_reverse_kl",
    "masked_entropy",
    "uniform_legal_log_probs",
    "compute_vrpo_targets",
    "update_ema",
    "wdl_expected_value",
    "reward_to_wdl_target",
    # Networks
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
    "UnoBackbone",
    "UnoActorNet",
    "UnoQCriticNet",
    # Eval Suite
    "RandomBot",
    "GreedyBot",
    "ExpertHeuristicBot",
    "evaluate_duplicate",
]
