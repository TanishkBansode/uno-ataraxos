"""Exact S4 (24-fold) color-permutation engine for 1v1 UNO.

Generates exact forward and inverse lookup tables for all 24 permutations of
the 4 colors {0:r, 1:g, 2:b, 3:y} matching cpp/uno_constants.hpp 1-to-1.

Index Transformation Rules:
- FWD tables: transform integer IDs (id_new = fwd[p, id_old])
- INV tables: gather/reindex tensor slots (x_new[..., i] = x_old[..., inv[p, i]])
- Invariant: legal_mask_new[fwd_action[p, a_old]] == legal_mask_old[a_old]
"""

from __future__ import annotations

import itertools
from typing import Optional, Tuple
import torch

# All 24 color permutations of (0, 1, 2, 3) where tuple[c] = sigma(c)
COLOR_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(4)))
NUM_PERMUTATIONS = 24

# Build lookup tables as torch.int64 tensors
_fwd_card_54 = torch.zeros((24, 54), dtype=torch.int64)
_inv_card_54 = torch.zeros((24, 54), dtype=torch.int64)
_fwd_action_62 = torch.zeros((24, 62), dtype=torch.int64)
_inv_action_61 = torch.zeros((24, 61), dtype=torch.int64)
_fwd_color_5 = torch.zeros((24, 5), dtype=torch.int64)
_inv_color_5 = torch.zeros((24, 5), dtype=torch.int64)
_inv_perm_id = torch.zeros((24,), dtype=torch.int64)

for p_idx, sigma in enumerate(COLOR_PERMUTATIONS):
    # 1. Card types (0..53)
    for c in range(54):
        if c < 40:
            col = c // 10
            num = c % 10
            new_c = sigma[col] * 10 + num
        elif c < 52:
            col = (c - 40) // 3
            trait = (c - 40) % 3
            new_c = 40 + sigma[col] * 3 + trait
        else:
            # 52: wild, 53: wild_draw_4 (neutral)
            new_c = c
        _fwd_card_54[p_idx, c] = new_c
        _inv_card_54[p_idx, new_c] = c

    # 2. Actions (0..61)
    for a in range(62):
        if a < 40:
            col = a // 10
            num = a % 10
            new_a = sigma[col] * 10 + num
        elif a < 52:
            col = (a - 40) // 3
            trait = (a - 40) % 3
            new_a = 40 + sigma[col] * 3 + trait
        elif a < 56:
            col = a - 52
            new_a = 52 + sigma[col]
        elif a < 60:
            col = a - 56
            new_a = 56 + sigma[col]
        elif a == 60:
            new_a = 60  # draw
        else:
            new_a = 61  # history padding token
        _fwd_action_62[p_idx, a] = new_a
        if a < 61:
            _inv_action_61[p_idx, new_a] = a

    # 3. Colors (0..4)
    for col in range(5):
        if col < 4:
            new_col = sigma[col]
        else:
            new_col = 4  # history padding color
        _fwd_color_5[p_idx, col] = new_col
        _inv_color_5[p_idx, new_col] = col

    # 4. Inverse permutation index
    sigma_inv = tuple(sigma.index(c) for c in range(4))
    _inv_perm_id[p_idx] = COLOR_PERMUTATIONS.index(sigma_inv)

# Export immutable constants
FWD_CARD_54: torch.Tensor = _fwd_card_54
INV_CARD_54: torch.Tensor = _inv_card_54
FWD_ACTION_62: torch.Tensor = _fwd_action_62
INV_ACTION_61: torch.Tensor = _inv_action_61
FWD_COLOR_5: torch.Tensor = _fwd_color_5
INV_COLOR_5: torch.Tensor = _inv_color_5
INV_PERM_ID: torch.Tensor = _inv_perm_id


def get_inverse_perm_id(perm_id: int) -> int:
    """Return the permutation index of the inverse permutation sigma^-1."""
    return int(INV_PERM_ID[perm_id].item())


def apply_s4_batch(
    card_feats: torch.Tensor,
    history_tokens: torch.Tensor,
    legal_mask: torch.Tensor,
    opp_hand_counts: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    perm_ids: Optional[torch.Tensor | int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Apply S4 color permutation to a batch of game tensors on device.

    Args:
        card_feats: [B, 54, 7] or [B, 54, 8] float32
        history_tokens: [B, 48, 5] int32
        legal_mask: [B, 61] bool
        opp_hand_counts: Optional [B, 54] float32
        actions: Optional [B] int32 or int64
        perm_ids: Optional [B] int64 tensor or int in [0, 23]. If None, random perm per item.

    Returns:
        Tuple containing permuted versions of the inputs.
    """
    B = card_feats.shape[0]
    device = card_feats.device

    if perm_ids is None:
        perm_ids = torch.randint(0, NUM_PERMUTATIONS, (B,), device=device, dtype=torch.int64)
    elif isinstance(perm_ids, int):
        perm_ids = torch.full((B,), perm_ids, device=device, dtype=torch.int64)
    else:
        perm_ids = perm_ids.to(device=device, dtype=torch.int64)

    # 1. Reindex card_feats slots via INV_CARD_54: x_new[..., i] = x_old[..., inv[p, i]]
    inv_card = INV_CARD_54.to(device)[perm_ids]  # [B, 54]
    C_dim = card_feats.shape[-1]
    inv_card_expanded = inv_card.unsqueeze(-1).expand(-1, -1, C_dim)
    card_feats_new = torch.gather(card_feats, dim=1, index=inv_card_expanded)

    # 2. Reindex legal_mask slots via INV_ACTION_61: mask_new[..., a] = mask_old[..., inv[p, a]]
    inv_act = INV_ACTION_61.to(device)[perm_ids]  # [B, 61]
    legal_mask_new = torch.gather(legal_mask, dim=1, index=inv_act)

    # 3. Transform history_tokens integer IDs via FWD tables
    hist_new = history_tokens.clone()
    fwd_act = FWD_ACTION_62.to(device)[perm_ids]  # [B, 62]
    fwd_col = FWD_COLOR_5.to(device)[perm_ids]    # [B, 5]

    # col 1: action_id in [0..61]
    act_old = history_tokens[..., 1].long()
    hist_new[..., 1] = torch.gather(fwd_act, dim=1, index=act_old).to(history_tokens.dtype)

    # col 2: active_color_before in [0..4]
    col_old = history_tokens[..., 2].long()
    hist_new[..., 2] = torch.gather(fwd_col, dim=1, index=col_old).to(history_tokens.dtype)

    outputs = [card_feats_new, hist_new, legal_mask_new]

    # 4. Optional opp_hand_counts: [B, 54] reindexed via INV_CARD_54
    if opp_hand_counts is not None:
        opp_hand_new = torch.gather(opp_hand_counts, dim=1, index=inv_card)
        outputs.append(opp_hand_new)

    # 5. Optional actions: [B] transformed via FWD_ACTION_62
    if actions is not None:
        act_orig = actions.long().unsqueeze(-1)
        actions_new = torch.gather(fwd_act, dim=1, index=act_orig).squeeze(-1).to(actions.dtype)
        outputs.append(actions_new)

    return tuple(outputs)
