#!/usr/bin/env python3
"""Export 20 full games from C++ UnoEngine with step-by-step tensor traces for JS Parity Testing.

Generates `tests/cpp_game_traces.json` containing:
- Initial game setup (hands, start card, active color)
- Step-by-step actions and exact C++ ground-truth tensors:
  - card_features_actor [54, 7]
  - global_features [10]
  - history_tokens [48, 5]
  - legal_mask [61]
- Detailed event metadata (cards_drawn, reshuffle events) for exact JS tracking verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from uno_ataraxos import UnoEngine, get_action_names, get_card_names


def export_traces(num_games: int = 20, output_file: str = "tests/cpp_game_traces.json"):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    action_names = list(get_action_names())
    card_names = list(get_card_names())

    games_data = []

    for game_idx in range(num_games):
        seed = 1000 + game_idx
        eng = UnoEngine(seed)
        eng.reset(seed)

        game_record = {
            "game_idx": game_idx,
            "seed": seed,
            "initial_hand_0": [int(c) for c in eng.get_hand_ids(0)],
            "initial_hand_1": [int(c) for c in eng.get_hand_ids(1)],
            "initial_target": int(eng.target),
            "initial_active_color": int(eng.active_color),
            "steps": [],
        }

        step_num = 0
        while not eng.is_over and step_num < 250:
            curr_p = int(eng.current_player)

            # Record expected ground truth tensors from acting player's perspective
            cf = eng.get_card_features_actor(curr_p).tolist()  # [54, 7]
            gf = eng.get_global_features(curr_p).tolist()       # [10]
            ht = eng.get_history_tokens(curr_p).tolist()        # [48, 5]
            mask = eng.get_action_mask(curr_p).astype(int).tolist() # [61]

            legal_ids = eng.get_legal_action_ids(curr_p)
            assert len(legal_ids) > 0, f"No legal actions for player {curr_p}!"

            chosen_action = int(np.random.choice(legal_ids))

            hand_0_before = [int(c) for c in eng.get_hand_ids(0)]
            hand_1_before = [int(c) for c in eng.get_hand_ids(1)]
            deck_before = [int(c) for c in eng.get_deck_ids()]
            discard_before = [int(c) for c in eng.get_discard_pile_ids()]
            target_before = int(eng.target)
            color_before = int(eng.active_color)
            stack_before = int(eng.stack)
            passes_before = int(eng.consecutive_passes)
            turns_before = int(eng.turn_count)
            avail_draw_before = int(eng.get_available_draw_count())
            reshuffle_pool_before = [int(x) for x in eng.get_last_reshuffle_pool()]

            # Step the engine
            eng.step(chosen_action)

            hand_0_after = [int(c) for c in eng.get_hand_ids(0)]
            hand_1_after = [int(c) for c in eng.get_hand_ids(1)]
            curr_hand_before = hand_0_before if curr_p == 0 else hand_1_before
            curr_hand_after = hand_0_after if curr_p == 0 else hand_1_after

            # Calculate cards drawn by acting player
            cards_drawn_count = max(0, len(curr_hand_after) - len(curr_hand_before))
            # Cards added to hand
            drawn_card_ids = []
            if cards_drawn_count > 0:
                drawn_card_ids = curr_hand_after[-cards_drawn_count:]

            reshuffle_pool_after = [int(x) for x in eng.get_last_reshuffle_pool()]
            reshuffled = (reshuffle_pool_before != reshuffle_pool_after)

            step_record = {
                "step_num": step_num,
                "current_player": curr_p,
                "target": target_before,
                "active_color": color_before,
                "stack": stack_before,
                "turn_count": turns_before,
                "consecutive_passes": passes_before,
                "available_draw_count": avail_draw_before,
                "hand_0": hand_0_before,
                "hand_1": hand_1_before,
                "deck_count": len(deck_before),
                "action_id": chosen_action,
                "action_name": action_names[chosen_action],
                "cards_drawn_count": cards_drawn_count,
                "drawn_card_ids": drawn_card_ids,
                "reshuffled": reshuffled,
                "reshuffle_pool": reshuffle_pool_after if reshuffled else None,
                "expected": {
                    "card_features": cf,
                    "global_features": gf,
                    "history_tokens": ht,
                    "legal_mask": mask,
                },
            }
            game_record["steps"].append(step_record)
            step_num += 1

        game_record["total_steps"] = step_num
        game_record["winner"] = -1 if eng.winner is None else int(eng.winner)
        games_data.append(game_record)

    total_steps = sum(g["total_steps"] for g in games_data)
    print(f"Exported {len(games_data)} full games with {total_steps} total decision steps to {out_path}.")

    with open(out_path, "w") as f:
        json.dump(games_data, f, indent=2)


if __name__ == "__main__":
    export_traces(20, "tests/cpp_game_traces.json")
