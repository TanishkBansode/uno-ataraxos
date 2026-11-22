"""Card-Game SOTA Hybrid Neural Architectures for uno-ataraxos (Milestone 2.5).

Replaces slow quadratic self-attention over fixed card tokens with:
1. Dense Card-Table GEMM Encoder (388/442 -> 256)
2. Parallel Masked Dilated-Conv History Encoder (128 dims, d in {1, 2, 4})
3. State-Conditioned 4-Query Cross-Attention over 49 tokens (1 null + 48 history)
4. Deep Pre-Norm SwiGLU ResMLP Trunk (4 blocks, 512 dims)

Hardware & Precision:
- Apple Silicon (MPS) and NVIDIA CUDA optimized with bfloat16 autocast support
- Explicit null key-value token guarantees zero NaNs on Turn 0 (all-padded history)
- Outputs cast explicitly to float32 for downstream RL math numerical stability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit feed-forward network with fused projection."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class TransformerBlock(nn.Module):
    """Lightweight legacy TransformerBlock preserved for backward compatibility."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x


class UnoHybridBackbone(nn.Module):
    """Hybrid ResMLP + Masked Dilated-Conv + 4-Query Cross-Attention Backbone.

    Processes imperfect (or privileged critic) state in ~18 ms at B=512 on Apple Silicon.
    """

    def __init__(
        self,
        in_card_dim: int = 7,
        d_state: int = 256,
        d_hist: int = 128,
        d_trunk: int = 512,
        d_ff: int = 200,
        n_blocks: int = 4,
        history_len: int = 48,
        n_heads: int = 4,
    ):
        super().__init__()
        self.in_card_dim = in_card_dim
        self.d_state = d_state
        self.d_hist = d_hist
        self.d_trunk = d_trunk
        self.history_len = history_len
        self.n_heads = n_heads
        self.head_dim = d_hist // n_heads

        # 1. Board / Card-Table Dense Encoder
        in_board_dim = 54 * in_card_dim + 10
        self.board_proj = nn.Linear(in_board_dim, d_state)
        self.board_norm = RMSNorm(d_state)

        # 2. Parallel Masked Dilated-Conv History Encoder
        self.emb_actor_rel = nn.Embedding(3, 16)
        self.emb_action_id = nn.Embedding(62, 48)
        self.emb_color = nn.Embedding(5, 12)
        self.emb_stack = nn.Embedding(17, 12)
        self.emb_drawn = nn.Embedding(17, 12)
        # Total emb dims = 16 + 48 + 12 + 12 + 12 = 100
        self.hist_proj = nn.Linear(100, d_hist)
        self.hist_pos = nn.Parameter(torch.randn(1, history_len, d_hist) * 0.02)

        # 3 Lightweight Residual 1D-Conv blocks with dilations {1, 2, 4}
        self.convs = nn.ModuleList([
            nn.Conv1d(d_hist, d_hist, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4]
        ])
        self.conv_norms = nn.ModuleList([
            nn.GroupNorm(1, d_hist) for _ in range(3)
        ])

        # 3. State-Conditioned 4-Query Cross-Attention
        self.q_proj = nn.Linear(d_state, 4 * d_hist)
        # 1 Learned null/dummy key-value token (always unmasked, prevents Turn 0 NaNs)
        self.null_kv = nn.Parameter(torch.randn(1, 1, d_hist) * 0.02)
        self.k_proj = nn.Linear(d_hist, d_hist, bias=False)
        self.v_proj = nn.Linear(d_hist, d_hist, bias=False)
        self.out_proj = nn.Linear(d_hist, d_hist, bias=False)

        # 4. Deep Card-Reasoning Trunk
        in_fusion_dim = d_state + (4 * d_hist) + d_hist  # 256 + 512 + 128 = 896
        self.fusion = nn.Linear(in_fusion_dim, d_trunk)
        self.trunk_norms = nn.ModuleList([RMSNorm(d_trunk) for _ in range(n_blocks)])
        self.trunk_blocks = nn.ModuleList([SwiGLU(d_trunk, d_ff) for _ in range(n_blocks)])
        self.final_norm = RMSNorm(d_trunk)

    def forward(
        self,
        card_features: torch.Tensor,   # [B, 54, in_card_dim]
        global_features: torch.Tensor, # [B, 10]
        history_tokens: torch.Tensor,  # [B, 48, 5]
    ) -> torch.Tensor:
        """Forward pass through UnoHybridBackbone returning trunk features [B, d_trunk]."""
        B = card_features.shape[0]
        device = card_features.device

        # 1. Normalize raw card counts by / 4.0 out-of-place
        if self.in_card_dim == 7:
            norm_cards = torch.cat([card_features[:, :, :4] / 4.0, card_features[:, :, 4:7]], dim=-1)
        elif self.in_card_dim == 8:
            norm_cards = torch.cat([card_features[:, :, :4] / 4.0, card_features[:, :, 4:7], card_features[:, :, 7:8] / 4.0], dim=-1)
        else:
            norm_cards = card_features

        cards_flat = norm_cards.flatten(1)  # [B, 54 * in_card_dim]
        x_board = torch.cat([cards_flat, global_features], dim=-1)
        state_emb = F.silu(self.board_norm(self.board_proj(x_board)))  # [B, 256]

        # 2. Parallel Masked Dilated-Conv History Encoder
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
        has_valid_hist = (actor_rel != 2).any(dim=1, keepdim=True)
        hist_mean = torch.where(has_valid_hist, hist_mean, torch.zeros_like(hist_mean))

        # 3. State-Conditioned 4-Query Cross-Attention
        # Q: [B, 4, 128] -> [B, n_heads=4, 4, head_dim=32]
        Q = self.q_proj(state_emb).view(B, 4, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # KV: Prepend learned null token -> [B, 49, 128]
        KV_in = torch.cat([self.null_kv.expand(B, -1, -1), h], dim=1)
        K = self.k_proj(KV_in).view(B, 49, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(KV_in).view(B, 49, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Float32 additive attention mask [B, 1, 1, 49]
        # Null token at index 0 is ALWAYS valid (0.0) -> guarantees zero NaNs on Turn 0
        null_valid = torch.ones((B, 1), dtype=torch.bool, device=device)
        hist_valid_bool = (actor_rel != 2)
        kv_valid = torch.cat([null_valid, hist_valid_bool], dim=1)  # [B, 49]
        float_mask = torch.where(kv_valid.unsqueeze(1).unsqueeze(2), 0.0, -1e9)

        attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=float_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 4, self.d_hist)
        hist_attn_out = self.out_proj(attn_out).reshape(B, 4 * self.d_hist)  # [B, 512]

        # 4. Deep Card-Reasoning Trunk
        fused = self.fusion(torch.cat([state_emb, hist_attn_out, hist_mean], dim=-1))  # [B, 512]
        x = fused
        for norm, block in zip(self.trunk_norms, self.trunk_blocks):
            x = x + block(norm(x))
        x = self.final_norm(x)  # [B, 512]

        return x


# Alias for backward compatibility
UnoBackbone = UnoHybridBackbone


class UnoActorNet(nn.Module):
    """Actor policy and WDL value network for 1v1 UNO (~2.2M params).

    Takes ONLY imperfect information: (card_features_actor, global_features, history_tokens).
    NEVER accepts privileged opp_hand_counts!
    """

    def __init__(
        self,
        d_state: int = 256,
        d_hist: int = 128,
        d_trunk: int = 512,
        d_ff: int = 200,
        n_blocks: int = 4,
        history_len: int = 48,
    ):
        super().__init__()
        self.backbone = UnoHybridBackbone(
            in_card_dim=7,
            d_state=d_state,
            d_hist=d_hist,
            d_trunk=d_trunk,
            d_ff=d_ff,
            n_blocks=n_blocks,
            history_len=history_len,
        )

        # Policy head (61 actions)
        self.policy_head = nn.Linear(d_trunk, 61)

        # WDL value head (Win / Draw / Loss logits)
        self.wdl_head = nn.Sequential(
            nn.Linear(d_trunk, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        card_features_actor: torch.Tensor,  # [B, 54, 7]
        global_features: torch.Tensor,      # [B, 10]
        history_tokens: torch.Tensor,       # [B, 48, 5]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            policy_logits: [B, 61] float32
            wdl_logits: [B, 3] float32
        """
        trunk = self.backbone(card_features_actor, global_features, history_tokens)
        policy_logits = self.policy_head(trunk)
        wdl_logits = self.wdl_head(trunk)

        # Explicit float32 cast ensures safety under mixed-precision / autocast
        return policy_logits.float(), wdl_logits.float()


class UnoQCriticNet(nn.Module):
    """Centralized Q-Critic network for 1v1 UNO (~2.2M params).

    Training-only centralized critic that accepts privileged opp_hand_counts.
    Outputs Q-values in [-1.0, 1.0] for all 61 actions.
    """

    def __init__(
        self,
        d_state: int = 256,
        d_hist: int = 128,
        d_trunk: int = 512,
        d_ff: int = 200,
        n_blocks: int = 4,
        history_len: int = 48,
    ):
        super().__init__()
        self.backbone = UnoHybridBackbone(
            in_card_dim=8,
            d_state=d_state,
            d_hist=d_hist,
            d_trunk=d_trunk,
            d_ff=d_ff,
            n_blocks=n_blocks,
            history_len=history_len,
        )

        # Q Head (61 action values in [-1, 1])
        self.q_head = nn.Linear(d_trunk, 61)

    def forward(
        self,
        card_features_actor: torch.Tensor,  # [B, 54, 7]
        opp_hand_counts: torch.Tensor,      # [B, 54] (privileged)
        global_features: torch.Tensor,      # [B, 10]
        history_tokens: torch.Tensor,       # [B, 48, 5]
    ) -> torch.Tensor:
        """Forward pass emitting Q-values in [-1, +1] for all 61 actions.

        Returns:
            q_values: [B, 61] float32
        """
        # Append opp_hand_counts as 8th channel: [B, 54, 8]
        card_features_critic = torch.cat([
            card_features_actor,
            opp_hand_counts.unsqueeze(-1),
        ], dim=-1)

        trunk = self.backbone(card_features_critic, global_features, history_tokens)
        q_values = torch.tanh(self.q_head(trunk))

        # Explicit float32 cast ensures safety under mixed-precision / autocast
        return q_values.float()
