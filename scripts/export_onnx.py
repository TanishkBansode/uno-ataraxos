#!/usr/bin/env python3
"""Export Ataraxos-UNO Models to WebGPU-Native ONNX with Parity Verification.

Exports:
1. UnoActorNet -> model/ataraxos_actor.onnx (FP32 ~8.9 MB, optional INT8 ~2.3 MB)
   - Inputs: card_features [B, 54, 7], global_features [B, 10], history_tokens [B, 48, 5] (int64), legal_mask [B, 61]
   - Outputs: action_probs [B, 61] (masked softmax inside graph), wdl_probs [B, 3] ([P(Loss), P(Draw), P(Win)])
   - WebGPU Safety: Replaces SDPA with explicit matmul -> scale -> add(mask) -> softmax -> matmul.
2. UnoBeliefNet -> model/belief_enc.onnx + model/belief_dec.onnx (for in-browser Belief X-Ray)

Usage:
    python scripts/export_onnx.py --blueprint runs/blueprint_100m/latest.pt --belief runs/belief_20m/best.pt --weight-int8
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from uno_ataraxos import UnoEngine, VecUnoEnv
from uno_ataraxos.belief import UnoBeliefNet
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import masked_log_softmax


class ActorExportWrapper(nn.Module):
    """WebGPU-safe export wrapper for UnoActorNet.

    Replaces F.scaled_dot_product_attention with explicit MatMul + Softmax
    so that 100% of operators execute natively on WebGPU with zero CPU fallbacks.
    Computes masked softmax over legal_mask inside the ONNX graph.
    """

    def __init__(self, actor: UnoActorNet):
        super().__init__()
        self.actor = actor

    def forward(
        self,
        card_features: torch.Tensor,   # [B, 54, 7] float32
        global_features: torch.Tensor, # [B, 10] float32
        history_tokens: torch.Tensor,  # [B, 48, 5] int64
        legal_mask: torch.Tensor,      # [B, 61] float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bb = self.actor.backbone
        B = card_features.shape[0]

        # 1. Normalize raw card counts by / 4.0 out-of-place
        norm_cards = torch.cat([card_features[:, :, :4] / 4.0, card_features[:, :, 4:7]], dim=-1)
        cards_flat = norm_cards.flatten(1)
        x_board = torch.cat([cards_flat, global_features], dim=-1)
        state_emb = F.silu(bb.board_norm(bb.board_proj(x_board)))

        # 2. Parallel Masked Dilated-Conv History Encoder
        actor_rel = history_tokens[..., 0].clamp(0, 2)
        action_id = history_tokens[..., 1].clamp(0, 61)
        color_before = history_tokens[..., 2].clamp(0, 4)
        stack_before = history_tokens[..., 3].clamp(0, 16)
        cards_drawn = history_tokens[..., 4].clamp(0, 16)

        hist_emb = torch.cat([
            bb.emb_actor_rel(actor_rel),
            bb.emb_action_id(action_id),
            bb.emb_color(color_before),
            bb.emb_stack(stack_before),
            bb.emb_drawn(cards_drawn),
        ], dim=-1)

        valid_mask = (actor_rel != 2).float().unsqueeze(-1)
        h = (bb.hist_proj(hist_emb) + bb.hist_pos) * valid_mask

        # 3 Lightweight Residual 1D-Conv blocks
        h_conv = h.transpose(1, 2)
        vm_conv = valid_mask.transpose(1, 2)
        for conv, norm in zip(bb.convs, bb.conv_norms):
            res = h_conv
            out = F.silu(norm(conv(h_conv)))
            h_conv = (res + out) * vm_conv
        h = h_conv.transpose(1, 2)

        # Masked mean over valid history moves
        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        hist_mean = h.sum(dim=1) / valid_count
        has_valid_hist = (actor_rel != 2).any(dim=1, keepdim=True)
        hist_mean = torch.where(has_valid_hist, hist_mean, torch.zeros_like(hist_mean))

        # 3. State-Conditioned 4-Query Cross-Attention (WebGPU native!)
        Q = bb.q_proj(state_emb).view(B, 4, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)
        KV_in = torch.cat([bb.null_kv.expand(B, -1, -1), h], dim=1)
        K = bb.k_proj(KV_in).view(B, 49, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)
        V = bb.v_proj(KV_in).view(B, 49, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)

        null_valid = torch.ones((B, 1), dtype=torch.bool, device=card_features.device)
        hist_valid_bool = (actor_rel != 2)
        kv_valid = torch.cat([null_valid, hist_valid_bool], dim=1)
        float_mask = torch.where(kv_valid.unsqueeze(1).unsqueeze(2), 0.0, -1e9)

        scale = 1.0 / math.sqrt(bb.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale + float_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 4, bb.d_hist)
        hist_attn_out = bb.out_proj(attn_out).reshape(B, 4 * bb.d_hist)

        # 4. Deep Card-Reasoning Trunk
        fused = bb.fusion(torch.cat([state_emb, hist_attn_out, hist_mean], dim=-1))
        x = fused
        for norm, block in zip(bb.trunk_norms, bb.trunk_blocks):
            x = x + block(norm(x))
        trunk = bb.final_norm(x)

        policy_logits = self.actor.policy_head(trunk)
        wdl_logits = self.actor.wdl_head(trunk)

        # Masked Softmax over legal actions
        neg_large = torch.tensor(-1e9, dtype=policy_logits.dtype, device=policy_logits.device)
        is_legal = legal_mask > 0.5
        masked_logits = torch.where(is_legal, policy_logits, neg_large)
        probs = F.softmax(masked_logits, dim=-1)
        action_probs = torch.where(is_legal, probs, torch.zeros_like(probs))
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        wdl_probs = F.softmax(wdl_logits, dim=-1)

        return action_probs, wdl_probs


class BeliefEncExportWrapper(nn.Module):
    """WebGPU-safe export wrapper for UnoBeliefNet Encoder."""

    def __init__(self, belief: UnoBeliefNet):
        super().__init__()
        self.belief = belief

    def forward(
        self,
        card_features: torch.Tensor,   # [B, 54, 7] float32
        global_features: torch.Tensor, # [B, 10] float32
        history_tokens: torch.Tensor,  # [B, 48, 5] int64
    ) -> torch.Tensor:
        b = self.belief
        B = card_features.shape[0]

        norm_cards = torch.cat([card_features[:, :, :4] / 4.0, card_features[:, :, 4:7]], dim=-1)
        cards_flat = norm_cards.flatten(1)
        x_board = torch.cat([cards_flat, global_features], dim=-1)
        state_emb = F.silu(b.board_norm(b.board_proj(x_board)))

        actor_rel = history_tokens[..., 0].clamp(0, 2)
        action_id = history_tokens[..., 1].clamp(0, 61)
        color_before = history_tokens[..., 2].clamp(0, 4)
        stack_before = history_tokens[..., 3].clamp(0, 16)
        cards_drawn = history_tokens[..., 4].clamp(0, 16)

        hist_emb = torch.cat([
            b.emb_actor_rel(actor_rel),
            b.emb_action_id(action_id),
            b.emb_color(color_before),
            b.emb_stack(stack_before),
            b.emb_drawn(cards_drawn),
        ], dim=-1)

        valid_mask = (actor_rel != 2).float().unsqueeze(-1)
        h = (b.hist_proj(hist_emb) + b.hist_pos) * valid_mask

        h_conv = h.transpose(1, 2)
        vm_conv = valid_mask.transpose(1, 2)
        for conv, norm in zip(b.convs, b.conv_norms):
            res = h_conv
            out = F.silu(norm(conv(h_conv)))
            h_conv = (res + out) * vm_conv
        h = h_conv.transpose(1, 2)

        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        hist_mean = h.sum(dim=1) / valid_count
        has_valid = (valid_mask.sum(dim=1) > 0.0)
        hist_mean = torch.where(has_valid, hist_mean, torch.zeros_like(hist_mean))

        Q = b.q_proj(state_emb).view(B, 4, b.n_heads, b.head_dim).permute(0, 2, 1, 3)
        KV_in = torch.cat([b.null_kv.expand(B, -1, -1), h], dim=1)
        K = b.k_proj(KV_in).view(B, 49, b.n_heads, b.head_dim).permute(0, 2, 1, 3)
        V = b.v_proj(KV_in).view(B, 49, b.n_heads, b.head_dim).permute(0, 2, 1, 3)

        null_valid = torch.ones((B, 1), dtype=torch.bool, device=card_features.device)
        hist_valid_bool = (valid_mask.squeeze(-1) > 0.5)
        kv_valid = torch.cat([null_valid, hist_valid_bool], dim=1)
        float_mask = torch.where(kv_valid.unsqueeze(1).unsqueeze(2), 0.0, -1e9)

        scale = 1.0 / math.sqrt(b.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale + float_mask
        attn_probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_probs, V)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 4, b.d_hist)
        hist_attn_out = b.out_proj(attn_out).reshape(B, 4 * b.d_hist)

        fused = torch.cat([state_emb, hist_attn_out, hist_mean], dim=-1)
        enc_context = F.silu(b.fusion_norm(b.fusion(fused)))
        return enc_context


class BeliefDecExportWrapper(nn.Module):
    """WebGPU-safe export wrapper for UnoBeliefNet Autoregressive Decoder Step."""

    def __init__(self, belief: UnoBeliefNet):
        super().__init__()
        self.belief = belief

    def forward(
        self,
        enc_context: torch.Tensor,        # [B, 256] float32
        picked_so_far: torch.Tensor,      # [B, 54] float32
        remaining_eligible: torch.Tensor, # [B, 54] float32
        prev_card_token: torch.Tensor,    # [B] int64
        cards_still_needed: torch.Tensor, # [B] float32
        step_legal_mask: torch.Tensor,    # [B, 54] float32
    ) -> torch.Tensor:
        b = self.belief
        p_norm = picked_so_far / 4.0
        e_norm = remaining_eligible / 4.0
        prev_emb = b.prev_card_emb(prev_card_token.clamp(0, 54))
        needed_norm = cards_still_needed.unsqueeze(-1) / 20.0

        x_in = torch.cat([enc_context, p_norm, e_norm, prev_emb, needed_norm], dim=-1)
        h = F.silu(b.dec_in_proj(x_in))

        for norm, block in zip(b.dec_norms, b.dec_blocks):
            h = h + block(norm(h))

        h = b.dec_final_norm(h)
        logits = b.dec_out_proj(h)

        neg_large = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        is_legal = step_legal_mask > 0.5
        masked_logits = torch.where(is_legal, logits, neg_large)
        probs = F.softmax(masked_logits, dim=-1)
        step_probs = torch.where(is_legal, probs, torch.zeros_like(probs))
        step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return step_probs


class CriticExportWrapper(nn.Module):
    """WebGPU-safe export wrapper for UnoQCriticNet with dynamic batching."""

    def __init__(self, critic: UnoQCriticNet):
        super().__init__()
        self.critic = critic

    def forward(
        self,
        card_features: torch.Tensor,   # [B, 54, 7] float32
        opp_hand_counts: torch.Tensor, # [B, 54] float32
        global_features: torch.Tensor, # [B, 10] float32
        history_tokens: torch.Tensor,  # [B, 48, 5] int64
    ) -> torch.Tensor:
        bb = self.critic.backbone
        B = card_features.shape[0]

        card_features_critic = torch.cat([
            card_features,
            opp_hand_counts.unsqueeze(-1),
        ], dim=-1)  # [B, 54, 8]

        norm_cards = torch.cat([card_features_critic[:, :, :4] / 4.0, card_features_critic[:, :, 4:7], card_features_critic[:, :, 7:8] / 4.0], dim=-1)
        cards_flat = norm_cards.flatten(1)
        x_board = torch.cat([cards_flat, global_features], dim=-1)
        state_emb = F.silu(bb.board_norm(bb.board_proj(x_board)))

        actor_rel = history_tokens[..., 0].clamp(0, 2)
        action_id = history_tokens[..., 1].clamp(0, 61)
        color_before = history_tokens[..., 2].clamp(0, 4)
        stack_before = history_tokens[..., 3].clamp(0, 16)
        cards_drawn = history_tokens[..., 4].clamp(0, 16)

        hist_emb = torch.cat([
            bb.emb_actor_rel(actor_rel),
            bb.emb_action_id(action_id),
            bb.emb_color(color_before),
            bb.emb_stack(stack_before),
            bb.emb_drawn(cards_drawn),
        ], dim=-1)

        valid_mask = (actor_rel != 2).float().unsqueeze(-1)
        h = (bb.hist_proj(hist_emb) + bb.hist_pos) * valid_mask

        h_conv = h.transpose(1, 2)
        vm_conv = valid_mask.transpose(1, 2)
        for conv, norm in zip(bb.convs, bb.conv_norms):
            res = h_conv
            out = F.silu(norm(conv(h_conv)))
            h_conv = (res + out) * vm_conv
        h = h_conv.transpose(1, 2)

        valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
        hist_mean = h.sum(dim=1) / valid_count
        has_valid_hist = (actor_rel != 2).any(dim=1, keepdim=True)
        hist_mean = torch.where(has_valid_hist, hist_mean, torch.zeros_like(hist_mean))

        Q = bb.q_proj(state_emb).view(B, 4, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)
        KV_in = torch.cat([bb.null_kv.expand(B, -1, -1), h], dim=1)
        K = bb.k_proj(KV_in).view(B, 49, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)
        V = bb.v_proj(KV_in).view(B, 49, bb.n_heads, bb.head_dim).permute(0, 2, 1, 3)

        null_valid = torch.ones((B, 1), dtype=torch.bool, device=card_features.device)
        hist_valid_bool = (actor_rel != 2)
        kv_valid = torch.cat([null_valid, hist_valid_bool], dim=1)
        float_mask = torch.where(kv_valid.unsqueeze(1).unsqueeze(2), 0.0, -1e9)

        scale = 1.0 / math.sqrt(bb.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale + float_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 4, bb.d_hist)
        hist_attn_out = bb.out_proj(attn_out).reshape(B, 4 * bb.d_hist)

        fused = bb.fusion(torch.cat([state_emb, hist_attn_out, hist_mean], dim=-1))
        x = fused
        for norm, block in zip(bb.trunk_norms, bb.trunk_blocks):
            x = x + block(norm(x))
        trunk = bb.final_norm(x)

        q_values = torch.tanh(self.critic.q_head(trunk))
        return q_values


def run_critic_parity_test(onnx_path: str, critic_net: UnoQCriticNet, num_states: int = 100) -> None:
    """Run mandatory parity test asserting critic max diff < 1e-4."""
    print(f"\n[Critic Parity Test] Validating {onnx_path} across {num_states} real VecUnoEnv states...")
    sess = ort.InferenceSession(onnx_path)

    env = VecUnoEnv(num_envs=20, seed=123)
    obs = env.reset()

    states_tested = 0
    max_q_diff = 0.0

    while states_tested < num_states:
        for e in range(20):
            if states_tested >= num_states:
                break
            cf = obs["card_features_actor"][e : e + 1]
            oh = obs["opp_hand_counts"][e : e + 1]
            gf = obs["global_features"][e : e + 1]
            ht = obs["history_tokens"][e : e + 1].astype(np.int64)

            ort_out = sess.run(None, {
                "card_features": cf,
                "opp_hand_counts": oh,
                "global_features": gf,
                "history_tokens": ht,
            })
            ort_q = ort_out[0][0]

            with torch.no_grad():
                cf_t = torch.from_numpy(cf)
                oh_t = torch.from_numpy(oh)
                gf_t = torch.from_numpy(gf)
                ht_t = torch.from_numpy(ht)
                pt_q = critic_net(cf_t, oh_t, gf_t, ht_t).numpy()[0]

            diff = float(np.abs(ort_q - pt_q).max())
            max_q_diff = max(max_q_diff, diff)
            states_tested += 1

        actions = np.zeros(20, dtype=np.int32)
        for e in range(20):
            legal = np.where(obs["legal_mask"][e] > 0.5)[0]
            actions[e] = np.random.choice(legal)
        obs, _, _, _ = env.step(actions)

    print(f"[Critic Parity Result] States: {states_tested} | Max Q Diff: {max_q_diff:.8e}")
    assert max_q_diff < 1e-4, f"Critic Q diff too large! {max_q_diff} >= 1e-4"
    print(">>> CRITIC PARITY TEST PASSED! Max Q diff < 1e-4 verified.")


def run_actor_parity_test(onnx_path: str, actor_net: UnoActorNet, num_states: int = 100) -> None:
    """Run mandatory parity test asserting argmax match 100/100 and max diff < 1e-4."""
    print(f"\n[Parity Test] Validating {onnx_path} across {num_states} real VecUnoEnv states...")
    sess = ort.InferenceSession(onnx_path)

    env = VecUnoEnv(num_envs=20, seed=42)
    obs = env.reset()

    states_tested = 0
    argmax_matches = 0
    max_prob_diff = 0.0

    while states_tested < num_states:
        for e in range(20):
            if states_tested >= num_states:
                break
            cf = obs["card_features_actor"][e : e + 1]
            gf = obs["global_features"][e : e + 1]
            ht = obs["history_tokens"][e : e + 1].astype(np.int64)
            mask_np = obs["legal_mask"][e : e + 1].astype(np.float32)

            # ORT Inference
            ort_out = sess.run(None, {
                "card_features": cf,
                "global_features": gf,
                "history_tokens": ht,
                "legal_mask": mask_np,
            })
            ort_probs = ort_out[0][0]

            # PyTorch Reference Inference
            with torch.no_grad():
                cf_t = torch.from_numpy(cf)
                gf_t = torch.from_numpy(gf)
                ht_t = torch.from_numpy(ht)
                m_t = torch.from_numpy(mask_np)
                raw_logits, _ = actor_net(cf_t, gf_t, ht_t)
                pt_log_p = masked_log_softmax(raw_logits, m_t > 0.5)
                pt_probs = torch.where(m_t > 0.5, pt_log_p.exp(), torch.zeros_like(pt_log_p)).numpy()[0]
                pt_probs = pt_probs / pt_probs.sum().clip(min=1e-9)

            ort_arg = int(ort_probs.argmax())
            pt_arg = int(pt_probs.argmax())
            diff = float(np.abs(ort_probs - pt_probs).max())

            if ort_arg == pt_arg:
                argmax_matches += 1
            max_prob_diff = max(max_prob_diff, diff)
            states_tested += 1

        # Step environments with random legal actions
        actions = np.zeros(20, dtype=np.int32)
        for e in range(20):
            legal = np.where(obs["legal_mask"][e] > 0.5)[0]
            actions[e] = np.random.choice(legal)
        obs, _, _, _ = env.step(actions)

    print(f"[Parity Test Result] States: {states_tested} | Argmax Match: {argmax_matches}/{states_tested} | Max Diff: {max_prob_diff:.8e}")
    assert argmax_matches == num_states, f"Argmax mismatch! {argmax_matches}/{num_states}"
    assert max_prob_diff < 1e-4, f"Probability diff too large! {max_prob_diff} >= 1e-4"
    print(">>> 100/100 PARITY TEST PASSED! Max prob diff < 1e-4 verified.")


def main():
    parser = argparse.ArgumentParser(description="Export Ataraxos-UNO models to WebGPU ONNX")
    parser.add_argument("--blueprint", type=str, default="runs/blueprint_100m/latest.pt", help="Path to blueprint checkpoint")
    parser.add_argument("--belief", type=str, default="runs/belief_20m/best.pt", help="Path to belief checkpoint")
    parser.add_argument("--output-dir", type=str, default="web/model", help="Target output directory")
    parser.add_argument("--weight-int8", action="store_true", help="Also generate weight-only INT8 quantized actor ONNX")
    parser.add_argument("--skip-belief", action="store_true", help="Skip belief net export")
    parser.add_argument("--skip-parity", action="store_true", help="Skip 100-state parity test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Blueprint Actor Net
    print(f"\n1. Loading UnoActorNet from {args.blueprint}...")
    ckpt = torch.load(args.blueprint, map_location="cpu")
    actor = UnoActorNet()
    raw_actor = ckpt.get("actor_state_dict", ckpt.get("actor", ckpt.get("exploiter_actor", ckpt)))
    actor_state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in raw_actor.items()}
    actor.load_state_dict(actor_state)
    actor.eval()
    param_count = sum(p.numel() for p in actor.parameters())
    print(f"Actor loaded successfully ({param_count:,} params, ~{param_count * 4 / (1024 * 1024):.2f} MB FP32).")

    # 2. Export UnoActorNet
    actor_wrapper = ActorExportWrapper(actor)
    actor_wrapper.eval()

    cf_dummy = torch.randn(1, 54, 7, dtype=torch.float32)
    gf_dummy = torch.randn(1, 10, dtype=torch.float32)
    ht_dummy = torch.zeros(1, 48, 5, dtype=torch.int64)
    mask_dummy = torch.ones(1, 61, dtype=torch.float32)

    actor_onnx_path = out_dir / "ataraxos_actor.onnx"
    print(f"Exporting Actor ONNX to {actor_onnx_path} (opset 17)...")
    torch.onnx.export(
        actor_wrapper,
        (cf_dummy, gf_dummy, ht_dummy, mask_dummy),
        str(actor_onnx_path),
        input_names=["card_features", "global_features", "history_tokens", "legal_mask"],
        output_names=["action_probs", "wdl_probs"],
        dynamic_axes={
            "card_features": {0: "batch_size"},
            "global_features": {0: "batch_size"},
            "history_tokens": {0: "batch_size"},
            "legal_mask": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
            "wdl_probs": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    actor_size_mb = actor_onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported ataraxos_actor.onnx: {actor_size_mb:.2f} MB.")

    # 3. Export UnoQCriticNet
    if "critic_state_dict" in ckpt or "critic" in ckpt:
        print(f"\n2. Loading UnoQCriticNet from {args.blueprint}...")
        critic = UnoQCriticNet()
        raw_critic = ckpt.get("critic_state_dict", ckpt.get("critic", ckpt))
        critic_state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in raw_critic.items()}
        critic.load_state_dict(critic_state)
        critic.eval()
        c_params = sum(p.numel() for p in critic.parameters())
        print(f"Critic loaded successfully ({c_params:,} params, ~{c_params * 4 / (1024 * 1024):.2f} MB FP32).")

        critic_wrapper = CriticExportWrapper(critic)
        critic_wrapper.eval()

        oh_dummy = torch.randn(1, 54, dtype=torch.float32)
        critic_onnx_path = out_dir / "ataraxos_critic.onnx"
        print(f"Exporting Critic ONNX to {critic_onnx_path} (opset 17)...")
        torch.onnx.export(
            critic_wrapper,
            (cf_dummy, oh_dummy, gf_dummy, ht_dummy),
            str(critic_onnx_path),
            input_names=["card_features", "opp_hand_counts", "global_features", "history_tokens"],
            output_names=["q_values"],
            dynamic_axes={
                "card_features": {0: "batch_size"},
                "opp_hand_counts": {0: "batch_size"},
                "global_features": {0: "batch_size"},
                "history_tokens": {0: "batch_size"},
                "q_values": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        c_size_mb = critic_onnx_path.stat().st_size / (1024 * 1024)
        print(f"Exported ataraxos_critic.onnx: {c_size_mb:.2f} MB.")

        if not args.skip_parity:
            run_critic_parity_test(str(critic_onnx_path), critic, num_states=100)

    # 4. Optional INT8 Quantization
    if args.weight_int8:
        import onnxruntime.quantization as oq
        int8_path = out_dir / "ataraxos_actor_int8.onnx"
        print(f"Quantizing weights to INT8: {int8_path}...")
        oq.quantize_dynamic(
            str(actor_onnx_path),
            str(int8_path),
            weight_type=oq.QuantType.QInt8,
        )
        int8_size_mb = int8_path.stat().st_size / (1024 * 1024)
        print(f"Exported ataraxos_actor_int8.onnx: {int8_size_mb:.2f} MB.")

    # 4. Parity Test
    if not args.skip_parity:
        run_actor_parity_test(str(actor_onnx_path), actor, num_states=100)

    # 5. Export UnoBeliefNet (Encoder + Decoder)
    if not args.skip_belief and Path(args.belief).exists():
        print(f"\n2. Loading UnoBeliefNet from {args.belief}...")
        bel_ckpt = torch.load(args.belief, map_location="cpu")
        belief = UnoBeliefNet()
        bel_state = bel_ckpt.get("belief_net", bel_ckpt.get("model_state_dict", bel_ckpt))
        belief.load_state_dict(bel_state)
        belief.eval()
        bel_params = sum(p.numel() for p in belief.parameters())
        print(f"Belief loaded successfully ({bel_params:,} params).")

        enc_wrapper = BeliefEncExportWrapper(belief)
        dec_wrapper = BeliefDecExportWrapper(belief)
        enc_wrapper.eval()
        dec_wrapper.eval()

        enc_path = out_dir / "belief_enc.onnx"
        print(f"Exporting Belief Encoder to {enc_path}...")
        torch.onnx.export(
            enc_wrapper,
            (cf_dummy, gf_dummy, ht_dummy),
            str(enc_path),
            input_names=["card_features", "global_features", "history_tokens"],
            output_names=["enc_context"],
            dynamic_axes={
                "card_features": {0: "batch_size"},
                "global_features": {0: "batch_size"},
                "history_tokens": {0: "batch_size"},
                "enc_context": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        print(f"Exported belief_enc.onnx: {enc_path.stat().st_size / (1024 * 1024):.2f} MB.")

        dec_path = out_dir / "belief_dec.onnx"
        print(f"Exporting Belief Decoder to {dec_path}...")
        enc_ctx_dummy = torch.randn(1, 256, dtype=torch.float32)
        p_dummy = torch.zeros(1, 54, dtype=torch.float32)
        rem_dummy = torch.ones(1, 54, dtype=torch.float32) * 2.0
        prev_tok_dummy = torch.tensor([54], dtype=torch.int64)
        needed_dummy = torch.tensor([7.0], dtype=torch.float32)
        step_mask_dummy = torch.ones(1, 54, dtype=torch.float32)

        torch.onnx.export(
            dec_wrapper,
            (enc_ctx_dummy, p_dummy, rem_dummy, prev_tok_dummy, needed_dummy, step_mask_dummy),
            str(dec_path),
            input_names=["enc_context", "picked_so_far", "remaining_eligible", "prev_card_token", "cards_still_needed", "step_legal_mask"],
            output_names=["step_probs"],
            dynamic_axes={
                "enc_context": {0: "batch_size"},
                "picked_so_far": {0: "batch_size"},
                "remaining_eligible": {0: "batch_size"},
                "prev_card_token": {0: "batch_size"},
                "cards_still_needed": {0: "batch_size"},
                "step_legal_mask": {0: "batch_size"},
                "step_probs": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        print(f"Exported belief_dec.onnx: {dec_path.stat().st_size / (1024 * 1024):.2f} MB.")

    print("\n[Export Complete] All target ONNX models generated successfully!")


if __name__ == "__main__":
    main()
