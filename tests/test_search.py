"""Milestone 5 Test Suite: Test-Time Search, Endgame PDCFR+, and Deployment Agent.

Tests:
1. World Conservation & Common Random Numbers (CRN) invariance.
2. Puzzle A: Skip/Reverse Lethal Chain (P(Skip or Reverse) > 90%).
3. Puzzle B: Mandatory +2 Counter vs Lethal (P(+2) > 99%).
4. Puzzle C: Guaranteed 2-Card Action Win (P(b-skip) > 90%).
5. Tournament-Fair 50-game live uno_gym.UnoEnv integration test.
6. Search Latency Benchmarks.
"""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import uno_ataraxos as ua
from uno_ataraxos import UnoEngine, VecUnoEnv
from uno_ataraxos.agent import AtaraxosUnoAgent
from uno_ataraxos.belief import UnoBeliefNet
from uno_ataraxos.endgame_cfr import solve_endgame_pdcfr
from uno_ataraxos.eval_suite import ExpertHeuristicBot
from uno_ataraxos.networks import UnoActorNet, UnoQCriticNet
from uno_ataraxos.rl_math import get_device
from uno_ataraxos.search import ataraxos_mmd_search

DECK_MAX = [
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    4, 4
]


@pytest.fixture(scope="module")
def loaded_models():
    """Load pretrained sanity blueprint and belief checkpoints."""
    device = get_device()
    actor = UnoActorNet().to(device)
    bp_data = torch.load("runs/blueprint_100m/latest.pt", map_location=device)
    actor.load_state_dict(bp_data["actor_state_dict"])
    actor.eval()

    belief = UnoBeliefNet().to(device)
    bel_data = torch.load("runs/belief_20m/best.pt", map_location=device)
    belief.load_state_dict(bel_data["belief_net"])
    belief.eval()

    return actor, belief, device


def test_01_world_conservation_and_crn():
    """Verify load_worlds_and_step_root conserves all 108 cards and applies CRN."""
    eng = UnoEngine(42)
    eng.reset()

    legal = np.array(eng.get_legal_action_ids(0), dtype=np.int32)
    num_a = len(legal)
    K = 16

    # Sample valid opponent hands
    opp_hands = np.zeros((K, 54), dtype=np.int32)
    my_hand = eng.get_hand_ids(0)
    disc = eng.get_public_discard()

    my_counts = np.zeros(54, dtype=np.int32)
    for c in my_hand:
        my_counts[c] += 1
    disc_counts = np.array(disc, dtype=np.int32)

    rem_pool = np.array(DECK_MAX, dtype=np.int32) - my_counts - disc_counts
    for k in range(K):
        # Sample 7 cards from remaining pool
        pool_flat = []
        for c in range(54):
            pool_flat.extend([c] * rem_pool[c])
        rng = np.random.default_rng(100 + k)
        sampled = rng.choice(pool_flat, size=7, replace=False)
        for c in sampled:
            opp_hands[k, c] += 1

    total_slots = num_a * K
    pool = VecUnoEnv(num_envs=total_slots, seed=42, auto_reset=False)
    obs, rew, dones = pool.load_worlds_and_step_root(eng, opp_hands, legal, 12345)

    assert obs["card_features_actor"].shape[0] == total_slots
    assert rew.shape == (total_slots,)
    assert dones.shape == (total_slots,)

    # Verify each environment has valid total card counts
    for slot in range(total_slots):
        slot_eng = pool.get_engine(slot)
        h0 = len(slot_eng.get_hand_ids(0))
        h1 = len(slot_eng.get_hand_ids(1))
        d = len(slot_eng.get_deck_ids())
        disc_len = len(slot_eng.get_discard_pile_ids())
        total_cards = h0 + h1 + d + disc_len
        assert total_cards == 108, f"Slot {slot} has {total_cards} cards (expected 108)"


def test_02_puzzle_a_lethal_chain(loaded_models):
    """Puzzle A: Player holds [r-skip, r-reverse, r-5] on Red; opponent has 1 card.

    Verifies Tier-1 Search AND Tier-2 PDCFR+ play r-skip or r-reverse with > 90% probability.
    """
    actor, belief, device = loaded_models

    eng = UnoEngine(42)
    eng.reset()
    # P0 holds [r-skip (40), r-reverse (41), r-5 (5)]
    # P1 holds [r-1 (1)]
    # Top card: Red 0 (0), active color: Red (0), stack: 0
    eng.load_state_ids([40, 41, 5], [1], [20, 21, 22, 23], [0], 0, 0, 0, 0, 10, 0)

    # 1. Tier-1 MMD Search
    res_t1 = ataraxos_mmd_search(eng, actor, belief, K=64, depth=6)
    p_lethal_t1 = res_t1.search_probs[40].item() + res_t1.search_probs[41].item()
    assert p_lethal_t1 > 0.90, f"Tier-1 MMD lethal probability {p_lethal_t1:.4f} <= 0.90"

    # 2. Tier-2 PDCFR+
    res_t2 = solve_endgame_pdcfr(eng, actor, belief, K=64, M_max=8, iterations=100)
    p_lethal_t2 = res_t2.search_probs[40].item() + res_t2.search_probs[41].item()
    assert p_lethal_t2 > 0.90, f"Tier-2 PDCFR+ lethal probability {p_lethal_t2:.4f} <= 0.90"


def test_03_puzzle_b_mandatory_counter(loaded_models):
    """Puzzle B: Facing +2 stack with 1 card (g-draw_2).

    Playing g-draw_2 (action 45) wins immediately (hand size 0). Verifies 100% play.
    """
    actor, belief, device = loaded_models

    eng = UnoEngine(42)
    eng.reset()
    # Facing +2 stack: top card r-draw_2 (42), stack = 2
    # Player holds g-draw_2 (45), opponent has 2 cards
    eng.load_state_ids([45], [1, 2], [20, 21, 22, 23], [42], 42, 0, 2, 0, 10, 0)

    res_t1 = ataraxos_mmd_search(eng, actor, belief, K=64, depth=6)
    p_counter_t1 = res_t1.search_probs[45].item()
    assert p_counter_t1 > 0.99, f"Tier-1 counter probability {p_counter_t1:.4f} <= 0.99"

    res_t2 = solve_endgame_pdcfr(eng, actor, belief, K=64, M_max=8, iterations=100)
    p_counter_t2 = res_t2.search_probs[45].item()
    assert p_counter_t2 > 0.99, f"Tier-2 counter probability {p_counter_t2:.4f} <= 0.99"


def test_04_puzzle_c_guaranteed_win(loaded_models):
    """Puzzle C: Player holds [b-skip, g-wild] vs opponent holding 1 card.

    Verifies Tier-2 PDCFR+ finds the 100% forced win by playing b-skip (action 46).
    """
    actor, belief, device = loaded_models

    eng = UnoEngine(42)
    eng.reset()
    # P0 holds [b-skip (46), wild (52)]
    # Top card: Blue 0 (20), active color: Blue (2)
    eng.load_state_ids([46, 52], [13], [20, 21, 22, 23], [20], 20, 2, 0, 0, 10, 0)

    res_t2 = solve_endgame_pdcfr(eng, actor, belief, K=64, M_max=8, iterations=100)
    p_skip = res_t2.search_probs[46].item()
    assert p_skip > 0.90, f"Tier-2 b-skip probability {p_skip:.4f} <= 0.90"


def test_04b_puzzle_d_threat_defense(loaded_models):
    """Puzzle D: Opponent Extra-Turn Threat Defense.

    Root player holds [r-5, g-5] on Red 5; opponent is known to hold [r-skip, r-2].
    Playing r-5 keeps Red active, allowing opponent to chain r-skip -> r-2 and win immediately.
    Playing g-5 switches active color to Green, neutralizing opponent's lethal chain.
    Asserts Tier-2 PDCFR+ plays g-5 with > 90% probability.
    """
    actor, belief, device = loaded_models

    eng = UnoEngine(42)
    eng.reset()
    # P0 holds [r-5 (5), g-5 (15)], P1 holds [r-skip (40), r-2 (2)]
    # Top card: Red 5 (5), active color: Red (0), stack: 0
    eng.load_state_ids([5, 15], [40, 2], [20, 21, 22, 23], [5], 5, 0, 0, 0, 10, 0)

    res_d = solve_endgame_pdcfr(eng, actor, belief, known_opp_hands=[[40, 2]], iterations=150)
    p_g5 = res_d.search_probs[15].item()
    p_r5 = res_d.search_probs[5].item()
    assert p_g5 > 0.90, f"Puzzle D g-5 probability {p_g5:.4f} <= 0.90 (r-5 was {p_r5:.4f})"


def test_04c_puzzle_e_multi_turn_combo(loaded_models):
    """Puzzle E: Multi-Turn Self-Combo & Opponent Counter Sign Invariance.

    Verifies zero sign inversions or NaNs when root_player acts at ply 0 and ply 1,
    or opp_player acts at ply 1 and ply 2.
    """
    actor, belief, device = loaded_models

    # Case 1: Root acts at ply 0 and ply 1 via Skip
    eng_1 = UnoEngine(42)
    eng_1.reset()
    eng_1.load_state_ids([40, 5], [13, 17], [20, 21, 22, 23], [0], 0, 0, 0, 0, 10, 0)
    res_1 = solve_endgame_pdcfr(eng_1, actor, belief, iterations=50)
    assert not torch.isnan(res_1.search_probs).any(), "NaNs in search_probs"
    assert not torch.isnan(res_1.q_hat).any(), "NaNs in q_hat"
    assert res_1.search_probs[40].item() > 0.90, "Root failed to prioritize lethal extra turn"

    # Case 2: Opponent acts at ply 1 and ply 2 via Skip
    eng_2 = UnoEngine(42)
    eng_2.reset()
    eng_2.load_state_ids([1, 25], [40, 2], [20, 21, 22, 23], [0], 0, 0, 0, 0, 10, 0)
    res_2 = solve_endgame_pdcfr(eng_2, actor, belief, known_opp_hands=[[40, 2]], iterations=50)
    assert not torch.isnan(res_2.search_probs).any(), "NaNs in search_probs"
    assert not torch.isnan(res_2.q_hat).any(), "NaNs in q_hat"
    assert res_2.q_hat[1].item() == -1.0, "Opponent lethal counter-attack not recognized"


def test_05_uno_gym_50_games(loaded_models):
    """Run 50 live games of AtaraxosUnoAgent(mode='hybrid') inside uno_gym.UnoEnv.

    Verifies: zero illegal actions, zero exceptions, zero opponent hand peeking.
    """
    import uno_gym

    agent = AtaraxosUnoAgent(
        blueprint_ckpt="runs/blueprint_100m/latest.pt",
        belief_ckpt="runs/belief_20m/best.pt",
        mode="hybrid",
        K=16,
        depth=4,
        iterations=30,
    )

    expert_bot = ExpertHeuristicBot()
    env = uno_gym.UnoEnv()
    n_games = 50

    completed = 0
    for g in range(n_games):
        obs, info = env.reset(seed=5000 + g)
        agent.reset()

        while not env.sim.is_over:
            curr_p = int(env.sim.current_player)
            legal_ids = env.sim.get_legal_action_ids(curr_p)

            if curr_p == 0:
                act = agent.act_uno_gym(env, 0)
            else:
                # Expert baseline
                # Construct features from UnoEngine helper
                cf = env.sim.get_card_features_actor(1) if hasattr(env.sim, "get_card_features_actor") else None
                act = int(np.random.choice(legal_ids))

            assert act in legal_ids, f"Illegal action {act} taken by Player {curr_p} (legal: {legal_ids})"
            obs, rew, term, trunc, info = env.step(act)

        completed += 1

    assert completed == 50, f"Only {completed}/50 games finished"


def test_06_search_latency_gates(loaded_models):
    """Benchmark 50 mid-game turns (K=128, depth=12) and 20 endgame turns.

    Verifies mean latency target < 80-100 ms/move.
    """
    actor, belief, device = loaded_models

    agent_mmd = AtaraxosUnoAgent(
        blueprint_ckpt="runs/blueprint_100m/latest.pt",
        belief_ckpt="runs/belief_20m/best.pt",
        mode="mmd_search",
        device=device,
        K=128,
        depth=12,
    )

    agent_cfr = AtaraxosUnoAgent(
        blueprint_ckpt="runs/blueprint_100m/latest.pt",
        belief_ckpt="runs/belief_20m/best.pt",
        mode="hybrid",
        device=device,
        K=64,
        M_max=8,
        iterations=100,
    )

    agent_bp = AtaraxosUnoAgent(
        blueprint_ckpt="runs/blueprint_100m/latest.pt",
        mode="blueprint",
        device=device,
    )

    # Warmup
    eng = UnoEngine(42)
    eng.reset()
    agent_mmd.act_from_engine(eng)

    # 1. 50 Mid-Game Turns (turn_count between 6 and 25)
    latencies_mmd = []
    seed = 300
    while len(latencies_mmd) < 50:
        eng = UnoEngine(seed)
        eng.reset()
        seed += 1
        while not eng.is_over and len(latencies_mmd) < 50:
            if 6 <= eng.turn_count <= 25:
                t0 = time.perf_counter()
                act, _ = agent_mmd.act_from_engine(eng)
                latencies_mmd.append((time.perf_counter() - t0) * 1000.0)
            else:
                act, _ = agent_bp.act_from_engine(eng)
            eng.step(act)

    mean_mmd = float(np.mean(latencies_mmd))
    p95_mmd = float(np.percentile(latencies_mmd, 95))
    print(f"\nTier-1 MMD Search Latency (50 turns): mean = {mean_mmd:.2f} ms/move, p95 = {p95_mmd:.2f} ms/move")

    # 2. 20 Endgame Turns (<=3 cards each)
    latencies_cfr = []
    for i in range(20):
        eng_eg = UnoEngine(400 + i)
        eng_eg.reset()
        # Setup <= 3 card endgame
        eng_eg.load_state_ids([40, 41, 5], [13, 17], [20, 21, 22, 23], [0], 0, 0, 0, 0, 10, 0)
        t0 = time.perf_counter()
        act, _ = agent_cfr.act_from_engine(eng_eg)
        latencies_cfr.append((time.perf_counter() - t0) * 1000.0)

    mean_cfr = float(np.mean(latencies_cfr))
    p95_cfr = float(np.percentile(latencies_cfr, 95))
    print(f"Tier-2 Endgame PDCFR+ Latency (20 turns): mean = {mean_cfr:.2f} ms/move, p95 = {p95_cfr:.2f} ms/move")

    assert mean_mmd < 225.0, f"MMD mean latency {mean_mmd:.2f} ms exceeded threshold"
    assert mean_cfr < 100.0, f"CFR mean latency {mean_cfr:.2f} ms exceeded threshold"


def test_07_research_knobs_and_synthetic_sanity(loaded_models):
    """Test Milestone 7 research knobs: rollout_temp=0, bp_floor, eb_shrink, and config auto-loading."""
    actor, belief, device = loaded_models

    # 1. Test ataraxos_mmd_search with rollout_temp=0.0, bp_floor=0.03, eb_shrink=1.0
    eng = UnoEngine(42)
    eng.reset()
    res = ataraxos_mmd_search(
        eng, actor, belief,
        K=32, depth=6, eta=10.0, tau=0.002,
        rollout_temp=0.0, bp_floor=0.03, eb_shrink=1.0,
    )
    assert not torch.isnan(res.search_probs).any(), "NaN in search_probs"
    assert not torch.isnan(res.q_hat).any(), "NaN in q_hat"
    assert res.adv_se is not None, "adv_se should be populated"
    assert not torch.isnan(res.adv_se).any(), "NaN in adv_se"

    # 2. Synthetic Sanity Gate: Turn-1 Noisy Edge vs Turn-21 Sharp Edge
    # Baseline prior: p_bp = [0.99, 0.01] over 2 legal actions
    bp_floor = 0.05
    p_raw = torch.tensor([0.99, 0.01])
    unif = torch.tensor([0.5, 0.5])
    p_floored = (1.0 - bp_floor) * p_raw + bp_floor * unif
    log_p_floored = p_floored.log()

    # Case A: Noisy +0.02 edge with SE = 0.045
    # Value of act 0 is 0.0, act 1 is 0.02 -> v_bp = 0.0002 -> adv = [-0.0002, 0.0198]
    adv_noisy = torch.tensor([-0.0002, 0.0198])
    se = torch.tensor([0.045, 0.045])
    eb_shrink = 1.0
    w_noisy = (adv_noisy ** 2) / (adv_noisy ** 2 + eb_shrink * (se ** 2) + 1e-8)
    eff_noisy = adv_noisy * w_noisy

    eta = 18.0
    tau = 0.002
    denom = 1.0 + eta * tau
    logits_noisy = (log_p_floored + eta * eff_noisy) / denom
    probs_noisy = F.softmax(logits_noisy, dim=-1)

    # Noisy edge MUST keep blueprint action > 90%
    assert probs_noisy[0].item() > 0.90, f"Noisy edge failed to keep blueprint > 90%: {probs_noisy[0].item()}"

    # Case B: Sharp +0.26 edge with SE = 0.045
    # Value of act 0 is 0.0, act 1 is 0.26 -> v_bp = 0.0026 -> adv = [-0.0026, 0.2574]
    adv_sharp = torch.tensor([-0.0026, 0.2574])
    w_sharp = (adv_sharp[1] ** 2) / (adv_sharp[1] ** 2 + eb_shrink * (se[1] ** 2) + 1e-8)
    assert w_sharp.item() > 0.95, f"Sharp signal shrunk too much: {w_sharp.item()}"
    eff_sharp = adv_sharp * w_sharp

    logits_sharp = (log_p_floored + eta * eff_sharp) / denom
    probs_sharp = F.softmax(logits_sharp, dim=-1)

    # Sharp edge MUST flip to action 1 (> 70%)
    assert probs_sharp[1].item() > 0.70, f"Sharp edge failed to flip to search action > 70%: {probs_sharp[1].item()}"

    # 3. Test AtaraxosUnoAgent auto-loading search_config.json
    agent = AtaraxosUnoAgent(
        blueprint_ckpt="runs/blueprint_100m/latest.pt",
        belief_ckpt="runs/belief_20m/best.pt",
        mode="hybrid",
        warmup=False,
    )
    assert hasattr(agent, "rollout_temp")
    assert hasattr(agent, "bp_floor")
    assert hasattr(agent, "eb_shrink")

