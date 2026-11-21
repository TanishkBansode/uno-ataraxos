"""Hard-Masked Autoregressive Generative Belief Network for 1v1 UNO (Milestone 4).

Models p(opp_hand | imperfect_state) autoregressively over a canonical descending
multiset sequence c_0 >= c_1 >= ... >= c_{k-1}.

Core Components:
1. compute_eligible: Exact card counting & reshuffle provenance tracking.
2. belief_step_legal_mask: Mathematical zero-dead-end canonical descending mask.
3. hypergeometric_hand_nll: Exact analytical multivariate hypergeometric prior yardstick.
4. UnoBeliefNet: Fast Board+History Encoder (~680k) + Sufficient-Statistic Multiset Decoder (~1.7M)
   (~2.4M total parameters).
5. sample_opp_hands: Fast batched world sampler (< 20 ms for K=128, k=7 on Apple Silicon M5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uno_ataraxos.networks import RMSNorm, SwiGLU
from uno_ataraxos.symmetry import apply_s4_batch


def compute_eligible(
    card_features_actor: torch.Tensor,
    global_features: torch.Tensor,
) -> torch.Tensor:
    """Compute exact eligible card counts allowed in opponent's hand.

    Args:
        card_features_actor: [..., 54, 7] float32 tensor
        global_features: [..., 10] float32 tensor

    Returns:
        eligible: [..., 54] long tensor of maximum counts opponent could hold.
    """
    unseen_total = card_features_actor[..., 2].round().long()
    recycled_remaining = card_features_actor[..., 3].round().long()
    opp_draws = (global_features[..., 8] * 20.0).round().long()

    # Where opp_draws_since_reshuffle == 0, opponent holds 0 freshly recycled cards
    is_zero_draws = (opp_draws == 0).unsqueeze(-1)
    eligible = torch.where(
        is_zero_draws,
        (unseen_total - recycled_remaining).clamp_min(0),
        unseen_total,
    )
    return eligible


def belief_step_legal_mask(
    remaining_eligible: torch.Tensor,  # [B, 54] long counts of cards still allowed
    prev_card_id: torch.Tensor,        # [B] long in [0..53] (initialized to 53 at j=0)
    cards_still_needed: torch.Tensor,  # [B] long = (k - j), >= 1
) -> torch.Tensor:
    """Zero-Dead-End Canonical Descending Mask.

    Guarantees:
      1) id <= prev_card_id                      (canonical descending order -> unique multiset factorization)
      2) remaining_eligible[id] > 0              (exact 108-card & reshuffle provenance limits)
      3) sum(remaining_eligible[0..id]) >= k - j (MATHEMATICAL ZERO DEAD-END GUARANTEE!)
    """
    B = remaining_eligible.shape[0]
    ids = torch.arange(54, device=remaining_eligible.device).unsqueeze(0).expand(B, 54)
    avail_le = remaining_eligible.cumsum(dim=-1)
    return (
        (ids <= prev_card_id.unsqueeze(-1))
        & (remaining_eligible > 0)
        & (avail_le >= cards_still_needed.unsqueeze(-1))
    )


def _log_comb(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute ln C(n, k) using torch.lgamma.

    Returns 0.0 where k == 0 or k == n. Returns -inf where k < 0 or k > n or n < 0.
    """
    n = n.float()
    k = k.float()
    valid = (k >= 0.0) & (k <= n) & (n >= 0.0)
    n_safe = n.clamp_min(0.0)
    k_safe = k.clamp_min(0.0)
    diff_safe = (n - k).clamp_min(0.0)

    res = torch.lgamma(n_safe + 1.0) - torch.lgamma(k_safe + 1.0) - torch.lgamma(diff_safe + 1.0)
    return torch.where(valid, res, torch.tensor(-float("inf"), device=n.device, dtype=n.dtype))


def hypergeometric_hand_nll(
    opp_hand_counts: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Compute exact multivariate hypergeometric negative log-probability.

    -ln [ prod_c C(E_c, h_c) / C(sum E_c, sum h_c) ]

    Args:
        opp_hand_counts: [..., 54] integer counts of cards in opponent hand
        eligible: [..., 54] integer counts of eligible cards

    Returns:
        nll: [...] float32 negative log likelihood in nats (0.0 for empty hands).
    """
    orig_shape = opp_hand_counts.shape[:-1]
    C = opp_hand_counts.shape[-1]
    h = opp_hand_counts.reshape(-1, C).float()
    e = eligible.reshape(-1, C).float()

    E_tot = e.sum(dim=-1)
    k_tot = h.sum(dim=-1)

    log_denom = _log_comb(E_tot, k_tot)
    log_numer = _log_comb(e, h).sum(dim=-1)

    nll = (log_denom - log_numer).clamp_min(0.0)
    # If hand is empty (k_tot == 0), NLL is 0.0
    nll = torch.where(k_tot == 0, torch.zeros_like(nll), nll)

    if len(orig_shape) == 0:
        return nll.squeeze(0)
    return nll.reshape(orig_shape)


class UnoBeliefNet(nn.Module):
    """Hard-Masked Autoregressive Generative Belief Network (~2.4M params).

    Encoder: Board GEMM + Masked Dilated-Conv History + 4-Query Cross-Attention.
    Decoder: 3 Pre-Norm SwiGLU Residual Blocks conditioned on sufficient statistics.
    """

    def __init__(
        self,
        d_state: int = 256,
        d_hist: int = 128,
        history_len: int = 48,
        history_dropout: float = 0.15,
        d_dec: int = 384,
        d_ff_dec: int = 448,
        n_blocks_dec: int = 3,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_hist = d_hist
        self.history_len = history_len
        self.history_dropout = history_dropout
        self.d_dec = d_dec

        # ====================================================================
        # 1. ENCODER (Card-Table GEMM + Dilated Conv + Cross-Attention)
        # ====================================================================
        in_board_dim = 54 * 7 + 10  # 388
        self.board_proj = nn.Linear(in_board_dim, d_state)
        self.board_norm = RMSNorm(d_state)

        # History embeddings
        self.emb_actor_rel = nn.Embedding(3, 16)
        self.emb_action_id = nn.Embedding(62, 48)
        self.emb_color = nn.Embedding(5, 12)
        self.emb_stack = nn.Embedding(17, 12)
        self.emb_drawn = nn.Embedding(17, 12)
        # Total emb dims = 16 + 48 + 12 + 12 + 12 = 100
        self.hist_proj = nn.Linear(100, d_hist)
        self.hist_pos = nn.Parameter(torch.randn(1, history_len, d_hist) * 0.02)

        # 3 Residual 1D-Conv blocks with dilations {1, 2, 4}
        self.convs = nn.ModuleList([
            nn.Conv1d(d_hist, d_hist, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4]
        ])
        self.conv_norms = nn.ModuleList([
            nn.GroupNorm(1, d_hist) for _ in range(3)
        ])

        # State-Conditioned 4-Query Cross-Attention
        self.n_heads = 4
        self.head_dim = d_hist // self.n_heads
        self.q_proj = nn.Linear(d_state, 4 * d_hist)
        self.null_kv = nn.Parameter(torch.randn(1, 1, d_hist) * 0.02)
        self.k_proj = nn.Linear(d_hist, d_hist, bias=False)
        self.v_proj = nn.Linear(d_hist, d_hist, bias=False)
        self.out_proj = nn.Linear(d_hist, d_hist, bias=False)

        # Context Fusion: 256 (board) + 512 (cross-attn) + 128 (hist-mean) = 896 -> 256
        self.fusion = nn.Linear(d_state + 4 * d_hist + d_hist, d_state)
        self.fusion_norm = RMSNorm(d_state)

        # ====================================================================
        # 2. AUTOREGRESSIVE MULTISET DECODER
        # ====================================================================
        # Inputs:
        # enc_context: 256
        # picked_so_far / 4.0: 54
        # remaining_eligible / 4.0: 54
        # prev_card_emb: 64 (Embedding 55 -> 64, 54 = BOS)
        # cards_still_needed / 20.0: 1
        # Total = 256 + 54 + 54 + 64 + 1 = 429
        self.prev_card_emb = nn.Embedding(55, 64)
        self.dec_in_proj = nn.Linear(429, d_dec)

        self.dec_norms = nn.ModuleList([RMSNorm(d_dec) for _ in range(n_blocks_dec)])
        self.dec_blocks = nn.ModuleList([SwiGLU(d_dec, d_ff_dec) for _ in range(n_blocks_dec)])
        self.dec_final_norm = RMSNorm(d_dec)

        self.dec_out_proj = nn.Linear(d_dec, 54)

    def encode(
        self,
        card_features_actor: torch.Tensor,  # [B, 54, 7]
        global_features: torch.Tensor,      # [B, 10]
        history_tokens: torch.Tensor,       # [B, 48, 5]
    ) -> torch.Tensor:
        """Encode imperfect information into a 256-dim fused context vector."""
        B = card_features_actor.shape[0]
        device = card_features_actor.device

        # 1. Normalize board counts by / 4.0 out-of-place
        norm_cards = torch.cat([
            card_features_actor[:, :, :4] / 4.0,
            card_features_actor[:, :, 4:7],
        ], dim=-1)
        cards_flat = norm_cards.flatten(1)  # [B, 378]
        x_board = torch.cat([cards_flat, global_features], dim=-1)
        state_emb = F.silu(self.board_norm(self.board_proj(x_board)))  # [B, 256]

        # 2. History tokens & structured history dropout
        actor_rel = history_tokens[..., 0].long().clamp(0, 2)
        action_id = history_tokens[..., 1].long().clamp(0, 61)
        color_before = history_tokens[..., 2].long().clamp(0, 4)
        stack_before = history_tokens[..., 3].long().clamp(0, 16)
        cards_drawn = history_tokens[..., 4].long().clamp(0, 16)

        hist_emb = torch.cat([
            self.emb_actor_rel(actor_rel),
            self.emb_action_id(action_id),
            self.emb_color(color_before),
            self.emb_stack(stack_before),
            self.emb_drawn(cards_drawn),
        ], dim=-1)

        valid_mask = (actor_rel != 2).float().unsqueeze(-1)  # [B, 48, 1]

        # Anti-OOD / Anti-Human-Poisoning Regularization:
        # 15% random dropout of valid history tokens during training
        if self.training and self.history_dropout > 0.0:
            drop_mask = (torch.rand((B, self.history_len, 1), device=device) < self.history_dropout)
            valid_mask = valid_mask * (~drop_mask).float()

        h = (self.hist_proj(hist_emb) + self.hist_pos) * valid_mask

        # 3 Lightweight Residual 1D-Conv blocks
        h_conv = h.transpose(1, 2)  # [B, 128, 48]
        vm_conv = valid_mask.transpose(1, 2)  # [B, 1, 48]
        for conv, norm in zip(self.convs, self.conv_norms):
            res = h_conv
            out = F.silu(norm(conv(h_conv)))
            h_conv = (res + out) * vm_conv
        h = h_conv.transpose(1, 2)  # [B, 48, 128]

        # Masked mean over valid history moves
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        hist_mean = h.sum(dim=1) / valid_count
        has_valid = (valid_mask.sum(dim=1) > 0.0)
        hist_mean = torch.where(has_valid, hist_mean, torch.zeros_like(hist_mean))

        # 3. State-Conditioned 4-Query Cross-Attention
        Q = self.q_proj(state_emb).view(B, 4, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        KV_in = torch.cat([self.null_kv.expand(B, -1, -1), h], dim=1)  # [B, 49, 128]
        K = self.k_proj(KV_in).view(B, 49, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(KV_in).view(B, 49, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        null_valid = torch.ones((B, 1), dtype=torch.bool, device=device)
        hist_valid_bool = (valid_mask.squeeze(-1) > 0.5)
        kv_valid = torch.cat([null_valid, hist_valid_bool], dim=1)  # [B, 49]
        float_mask = torch.where(kv_valid.unsqueeze(1).unsqueeze(2), 0.0, -1e9)

        attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=float_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 4, self.d_hist)
        hist_attn_out = self.out_proj(attn_out).reshape(B, 4 * self.d_hist)  # [B, 512]

        # 4. Context Fusion
        fused = torch.cat([state_emb, hist_attn_out, hist_mean], dim=-1)  # [B, 896]
        enc_context = F.silu(self.fusion_norm(self.fusion(fused)))        # [B, 256]

        return enc_context

    def decode_step(
        self,
        enc_context: torch.Tensor,        # [B_steps, 256]
        picked_so_far: torch.Tensor,      # [B_steps, 54] long or float
        remaining_eligible: torch.Tensor, # [B_steps, 54] long or float
        prev_card_token: torch.Tensor,    # [B_steps] long in [0..54] (54 = BOS)
        cards_still_needed: torch.Tensor, # [B_steps] long or float (>= 1)
    ) -> torch.Tensor:
        """Autoregressive step returning unmasked card logits [B_steps, 54]."""
        p_norm = picked_so_far.float() / 4.0
        e_norm = remaining_eligible.float() / 4.0
        prev_emb = self.prev_card_emb(prev_card_token.long().clamp(0, 54))
        needed_norm = cards_still_needed.float().unsqueeze(-1) / 20.0

        x_in = torch.cat([enc_context, p_norm, e_norm, prev_emb, needed_norm], dim=-1)  # [B_steps, 429]
        h = F.silu(self.dec_in_proj(x_in))  # [B_steps, 384]

        for norm, block in zip(self.dec_norms, self.dec_blocks):
            h = h + block(norm(h))

        h = self.dec_final_norm(h)
        logits = self.dec_out_proj(h).float()  # [B_steps, 54] in fp32
        return logits

    def compute_nll_loss(
        self,
        card_features_actor: torch.Tensor,  # [B, 54, 7]
        global_features: torch.Tensor,      # [B, 10]
        history_tokens: torch.Tensor,       # [B, 48, 5]
        opp_hand_counts: torch.Tensor,      # [B, 54]
        apply_s4: bool = True,
    ) -> Dict[str, torch.Tensor | float]:
        """Parallel Teacher-Forced NLL training step.

        Applies S4 augmentation, builds all autoregressive step states,
        runs ONE batched decoder forward pass, masks with belief_step_legal_mask,
        and computes card NLL, hand NLL, and comparison against Hypergeometric Prior.
        """
        B = card_features_actor.shape[0]
        device = card_features_actor.device

        # 1. Critical S4 Augmentation Rule:
        # Apply S4 color permutation BEFORE expanding descending sorted sequences
        if apply_s4 and self.training:
            dummy_mask = torch.ones((B, 61), dtype=torch.bool, device=device)
            card_features_actor, history_tokens, _, opp_hand_counts = apply_s4_batch(
                card_features_actor,
                history_tokens,
                dummy_mask,
                opp_hand_counts=opp_hand_counts,
            )

        # 2. Compute exact eligible card counts
        eligible = compute_eligible(card_features_actor, global_features)

        # 3. Encode imperfect information
        enc_context = self.encode(card_features_actor, global_features, history_tokens)  # [B, 256]

        # 4. Fast Teacher-Forcing State Unpacker (in descending multiset order)
        counts_np = opp_hand_counts.detach().round().long().cpu().numpy()
        elig_np = eligible.detach().cpu().numpy()

        enc_indices = []
        picked_list = []
        elig_list = []
        prev_tokens = []
        prev_ids = []
        needed_list = []
        targets = []

        for i in range(B):
            c_i = counts_np[i]
            k_i = int(c_i.sum())
            if k_i <= 0:
                continue

            e_i = elig_np[i].copy()
            # Canonical descending order (highest card ID down to 0)
            cards = []
            for c in range(53, -1, -1):
                cnt = int(c_i[c])
                if cnt > 0:
                    cards.extend([c] * cnt)

            curr_picked = np.zeros(54, dtype=np.int64)
            prev_tok = 54   # BOS token
            prev_id = 53    # Highest card ID

            for j, c_j in enumerate(cards):
                enc_indices.append(i)
                picked_list.append(curr_picked.copy())
                elig_list.append(e_i.copy())
                prev_tokens.append(prev_tok)
                prev_ids.append(prev_id)
                needed_list.append(k_i - j)
                targets.append(c_j)

                curr_picked[c_j] += 1
                e_i[c_j] -= 1
                prev_tok = c_j
                prev_id = c_j

        if len(targets) == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "loss": zero_loss,
                "card_nll": 0.0,
                "hand_nll": 0.0,
                "hypergeom_nll": 0.0,
                "nll_gain": 0.0,
            }

        # Convert to batch tensors on device
        enc_idx_t = torch.tensor(enc_indices, dtype=torch.long, device=device)
        picked_t = torch.tensor(np.array(picked_list), dtype=torch.long, device=device)
        elig_t = torch.tensor(np.array(elig_list), dtype=torch.long, device=device)
        prev_tok_t = torch.tensor(prev_tokens, dtype=torch.long, device=device)
        prev_id_t = torch.tensor(prev_ids, dtype=torch.long, device=device)
        needed_t = torch.tensor(needed_list, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)

        # 5. One Batched Decoder Forward Pass
        enc_ctx_batch = enc_context[enc_idx_t]  # [N_steps, 256]
        logits = self.decode_step(enc_ctx_batch, picked_t, elig_t, prev_tok_t, needed_t)  # [N_steps, 54]

        # 6. Apply Exact Zero-Dead-End Legal Mask
        mask = belief_step_legal_mask(elig_t, prev_id_t, needed_t)
        masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=device, dtype=logits.dtype))

        # 7. Compute Cross-Entropy NLL
        log_probs = F.log_softmax(masked_logits, dim=-1)
        step_log_p = log_probs.gather(dim=-1, index=targets_t.unsqueeze(-1)).squeeze(-1)
        step_nll = -step_log_p

        card_nll = step_nll.mean()

        # Compute per-hand NLL by summing step NLLs per batch item
        hand_nll_sum = torch.zeros(B, device=device).scatter_add_(0, enc_idx_t, step_nll)
        hand_counts_tot = torch.from_numpy(counts_np.sum(axis=-1)).to(device)
        has_cards = (hand_counts_tot > 0)
        hand_nll = hand_nll_sum[has_cards].mean() if has_cards.any() else torch.tensor(0.0, device=device)

        # 8. Exact Analytical Hypergeometric Prior Baseline
        hypergeom_per_hand = hypergeometric_hand_nll(opp_hand_counts, eligible)
        hypergeom_mean = hypergeom_per_hand[has_cards].mean() if has_cards.any() else torch.tensor(0.0, device=device)

        nll_gain = (hypergeom_mean - hand_nll).item()

        return {
            "loss": card_nll,
            "card_nll": card_nll.item(),
            "hand_nll": hand_nll.item(),
            "hypergeom_nll": hypergeom_mean.item(),
            "nll_gain": nll_gain,
        }

    @torch.no_grad()
    def sample_opp_hands(
        self,
        card_features_actor: torch.Tensor,  # [B=1, 54, 7]
        global_features: torch.Tensor,      # [B=1, 10]
        history_tokens: torch.Tensor,       # [B=1, 48, 5]
        K: int = 128,
        temp: float = 1.0,
        uniform_mix: float = 0.03,
    ) -> torch.Tensor:
        """Fast Batched World Sampler.

        Runs encode(...) ONCE for B=1, expands context to K, and autoregressively
        samples k cards with belief_step_legal_mask.

        Returns:
            sampled_opp_hands: [K, 54] int32 counts summing to opp_hand_size,
                              guaranteed <= eligible with ZERO dead-ends.
        """
        device = card_features_actor.device
        opp_hand_size = int((global_features[0, 1] * 20.0).round().item())

        if opp_hand_size <= 0:
            return torch.zeros((K, 54), dtype=torch.int32, device=device)

        # 1. Eligible pool
        eligible_1 = compute_eligible(card_features_actor, global_features)  # [1, 54]

        # 2. Encode ONCE
        enc_1 = self.encode(card_features_actor, global_features, history_tokens)  # [1, 256]

        # 3. Expand to K
        enc_ctx = enc_1.expand(K, -1)
        remaining_eligible = eligible_1.expand(K, -1).clone()
        picked_so_far = torch.zeros((K, 54), dtype=torch.long, device=device)
        prev_card_token = torch.full((K,), 54, dtype=torch.long, device=device)  # BOS
        prev_card_id = torch.full((K,), 53, dtype=torch.long, device=device)

        k = opp_hand_size

        # 4. Autoregressive loop j = 0 .. k-1
        for j in range(k):
            cards_needed = torch.full((K,), k - j, dtype=torch.long, device=device)
            logits = self.decode_step(
                enc_ctx,
                picked_so_far,
                remaining_eligible,
                prev_card_token,
                cards_needed,
            )

            mask = belief_step_legal_mask(remaining_eligible, prev_card_id, cards_needed)
            masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=device, dtype=logits.dtype))

            t_val = max(float(temp), 1e-4)
            probs = F.softmax(masked_logits / t_val, dim=-1)

            # Uniform hypergeometric mixture safety floor
            if uniform_mix > 0.0:
                unif = (remaining_eligible.float() * mask.float()).clamp_min(0.0)
                unif_sum = unif.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                unif_probs = unif / unif_sum
                probs = (1.0 - uniform_mix) * probs + uniform_mix * unif_probs

            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            # Sample c_j
            c_j = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [K]

            # Update
            picked_so_far.scatter_add_(1, c_j.unsqueeze(-1), torch.ones((K, 1), dtype=torch.long, device=device))
            remaining_eligible.scatter_add_(1, c_j.unsqueeze(-1), -torch.ones((K, 1), dtype=torch.long, device=device))
            prev_card_token = c_j
            prev_card_id = c_j

        return picked_so_far.to(dtype=torch.int32)
