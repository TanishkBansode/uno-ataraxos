#pragma once

#include "uno_constants.hpp"

#include <vector>
#include <array>
#include <string>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <cstdint>
#include <utility>

namespace uno {

struct HistoryEvent {
    int8_t actor_id = 0;             // 0 or 1
    uint8_t action_id = 0;           // 0..60
    int8_t active_color_before = 0;  // 0..3 (r, g, b, y)
    uint8_t stack_before = 0;        // clamped 0..16
    uint8_t cards_drawn = 0;         // clamped 0..16
};

class UnoEngine {
public:
    // Core Game State
    std::vector<uint8_t> hands[2];             // Card IDs (0..53) in player hands
    std::vector<uint8_t> deck;                 // Draw pile of Card IDs (0..53)
    std::vector<uint8_t> discard_pile;         // Discard history of Action IDs (0..59)
    std::vector<uint8_t> played_cards;         // Discards since last reshuffle (Action IDs)
    uint8_t target = 0;                        // Current target: Action ID (0..59)
    int8_t active_color = 0;                   // Active color: 0..3 (RED, GREEN, BLUE, YELLOW)
    int32_t stack = 0;                         // Active penalty stack (+0, +2, +4, +6, ...)
    int8_t current_player = 0;                 // 0 or 1
    bool is_over = false;                      // Game over flag
    int8_t winner = -1;                        // -1: none/draw, 0: player 0, 1: player 1
    int32_t consecutive_passes = 0;            // Consecutive passes with 0 cards drawn
    int32_t turn_count = 0;                    // Total turns played (max 200)

    // Reshuffle Provenance Tracking (For Exact Card Counting & Belief Net)
    uint8_t public_discard[54];                // Counts of cards currently in discard pile
    uint8_t last_reshuffle_pool[54];           // Card multiset recycled on last reshuffle
    int32_t draws_since_reshuffle[2];          // Cards drawn per player since last reshuffle
    uint8_t known_drawn_from_recycled[2][54];  // Cards each player is known to have drawn from recycled pool

    // Move History Ring Buffer (Last 48 Events)
    std::vector<HistoryEvent> history_events;

    // RNG (64-bit Mersenne Twister for high quality & fast rollouts)
    std::mt19937_64 rng;

public:
    explicit UnoEngine(uint64_t seed = 42) : rng(seed) {
        history_events.reserve(48);
        reset(seed);
    }

    void reset(uint64_t seed = 0) {
        if (seed != 0) {
            rng.seed(seed);
        }

        std::vector<uint8_t> full_deck = build_108_deck_vector();
        std::shuffle(full_deck.begin(), full_deck.end(), rng);

        hands[0].clear();
        hands[1].clear();
        hands[0].reserve(32);
        hands[1].reserve(32);

        // Deal 7 cards to each player (popping from the back of full_deck)
        for (int i = 0; i < 7; ++i) {
            hands[0].push_back(full_deck.back());
            full_deck.pop_back();
        }
        for (int i = 0; i < 7; ++i) {
            hands[1].push_back(full_deck.back());
            full_deck.pop_back();
        }

        // Draw starting card: must be a standard number card 0-9
        std::vector<uint8_t> non_numbers;
        uint8_t start_card = 0;
        while (!full_deck.empty()) {
            uint8_t c = full_deck.back();
            full_deck.pop_back();
            if (card_is_number(c)) {
                start_card = c;
                break;
            }
            non_numbers.push_back(c);
        }

        // Return any buried non-number cards to deck and shuffle
        if (!non_numbers.empty()) {
            full_deck.insert(full_deck.end(), non_numbers.begin(), non_numbers.end());
            std::shuffle(full_deck.begin(), full_deck.end(), rng);
        }

        deck = std::move(full_deck);
        target = start_card; // Number card action ID equals card ID (0..39)
        active_color = card_color(start_card);
        discard_pile.clear();
        discard_pile.push_back(target);
        played_cards.clear();
        played_cards.push_back(target);

        stack = 0;
        current_player = 0;
        is_over = false;
        winner = -1;
        consecutive_passes = 0;
        turn_count = 0;

        // Reset card counting & provenance
        std::memset(public_discard, 0, sizeof(public_discard));
        public_discard[normalize_target_to_card(target)] = 1;

        std::memset(last_reshuffle_pool, 0, sizeof(last_reshuffle_pool));
        draws_since_reshuffle[0] = 0;
        draws_since_reshuffle[1] = 0;
        std::memset(known_drawn_from_recycled, 0, sizeof(known_drawn_from_recycled));

        history_events.clear();
    }

    UnoEngine clone() const {
        return *this;
    }

    int32_t get_available_draw_count() const {
        int32_t count = static_cast<int32_t>(deck.size());
        if (discard_pile.size() > 1) {
            count += static_cast<int32_t>(discard_pile.size() - 1);
        }
        return count;
    }

    void _recycle_discard_pile() {
        if (discard_pile.size() <= 1) {
            return;
        }
        uint8_t top = discard_pile.back();
        std::vector<uint8_t> recycled;
        recycled.reserve(discard_pile.size() - 1);

        // Normalize all discards except top back to neutral card types (0..53)
        for (size_t i = 0; i < discard_pile.size() - 1; ++i) {
            recycled.push_back(action_required_card(discard_pile[i]));
        }

        // Provenance tracking: capture recycled pool
        std::memset(last_reshuffle_pool, 0, sizeof(last_reshuffle_pool));
        for (uint8_t c : recycled) {
            if (c < NUM_CARD_TYPES) {
                last_reshuffle_pool[c]++;
            }
        }
        draws_since_reshuffle[0] = 0;
        draws_since_reshuffle[1] = 0;
        std::memset(known_drawn_from_recycled, 0, sizeof(known_drawn_from_recycled));

        std::shuffle(recycled.begin(), recycled.end(), rng);
        deck = std::move(recycled);
        discard_pile.clear();
        discard_pile.push_back(top);
        played_cards.clear();
        played_cards.push_back(top);

        // Discard pile now contains ONLY top card
        std::memset(public_discard, 0, sizeof(public_discard));
        public_discard[normalize_target_to_card(top)] = 1;
    }

    // Draw cards from deck with recycling mid-draw
    std::vector<uint8_t> draw_cards(int32_t count, int8_t player) {
        std::vector<uint8_t> drawn;
        drawn.reserve(count);
        for (int32_t i = 0; i < count; ++i) {
            if (deck.empty()) {
                _recycle_discard_pile();
            }
            if (deck.empty()) {
                break; // No cards left in deck or discards to recycle
            }
            uint8_t c = deck.back();
            deck.pop_back();
            hands[player].push_back(c);
            drawn.push_back(c);

            // Track provenance for cards drawn from recycled pool
            draws_since_reshuffle[player]++;
            if (c < NUM_CARD_TYPES) {
                known_drawn_from_recycled[player][c]++;
            }
        }
        return drawn;
    }

    void _remove_card_from_hand(int8_t player, uint8_t action_id) {
        uint8_t req_card = static_cast<uint8_t>(action_required_card(action_id));
        auto& hand = hands[player];
        for (auto it = hand.begin(); it != hand.end(); ++it) {
            if (*it == req_card) {
                hand.erase(it);
                return;
            }
        }
        throw std::runtime_error("Action " + std::to_string(action_id) + " card not in player hand");
    }

    void _end_by_card_count() {
        is_over = true;
        size_t c0 = hands[0].size();
        size_t c1 = hands[1].size();
        if (c0 < c1) {
            winner = 0;
        } else if (c1 < c0) {
            winner = 1;
        } else {
            winner = -1; // Draw
        }
    }

    // ========================================================================
    // Legal Actions & Mask Generation
    // ========================================================================

    void get_action_mask(uint8_t* out_mask, int8_t player = -1) const {
        std::memset(out_mask, 0, NUM_ACTIONS * sizeof(uint8_t));
        if (is_over) {
            return;
        }
        if (player < 0) {
            player = current_player;
        }

        const auto& hand = hands[player];
        int8_t target_tr = action_trait(target);

        if (stack > 0) {
            // Facing active penalty stack: strict same-type draw countering
            if (target_tr == DRAW_2) {
                for (uint8_t c : hand) {
                    if (c >= 40 && c < 52 && card_trait(c) == DRAW_2) {
                        out_mask[c] = 1; // Countering with any color draw_2
                    }
                }
            } else if (target_tr == WILD_DRAW_4) {
                bool has_wd4 = false;
                for (uint8_t c : hand) {
                    if (c == 53) {
                        has_wd4 = true;
                        break;
                    }
                }
                if (has_wd4) {
                    out_mask[56] = 1; // r-wild_draw_4
                    out_mask[57] = 1; // g-wild_draw_4
                    out_mask[58] = 1; // b-wild_draw_4
                    out_mask[59] = 1; // y-wild_draw_4
                }
            }
            out_mask[60] = 1; // "draw" is always available to absorb stack
            return;
        }

        // Standard turn (stack == 0)
        bool has_playable_card = false;
        for (uint8_t c : hand) {
            if (c == 52) {
                // Wild: declare any of 4 colors
                out_mask[52] = 1;
                out_mask[53] = 1;
                out_mask[54] = 1;
                out_mask[55] = 1;
                has_playable_card = true;
            } else if (c == 53) {
                // Wild Draw 4: always playable on normal turn
                out_mask[56] = 1;
                out_mask[57] = 1;
                out_mask[58] = 1;
                out_mask[59] = 1;
                has_playable_card = true;
            } else {
                int8_t col = card_color(c);
                int8_t tr = card_trait(c);
                if (col == active_color || tr == target_tr) {
                    out_mask[c] = 1;
                    has_playable_card = true;
                }
            }
        }

        int32_t cards_available = get_available_draw_count();
        // Deck Exhaustion Guard: voluntary draw disabled only if 0 cards available and playable cards exist
        if (cards_available > 0 || !has_playable_card) {
            out_mask[60] = 1;
        }
    }

    std::vector<int32_t> get_legal_action_ids(int8_t player = -1) const {
        uint8_t mask[NUM_ACTIONS];
        get_action_mask(mask, player);
        std::vector<int32_t> actions;
        actions.reserve(NUM_ACTIONS);
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            if (mask[i]) {
                actions.push_back(i);
            }
        }
        return actions;
    }

    std::vector<std::string> get_legal_actions(int8_t player = -1) const {
        auto ids = get_legal_action_ids(player);
        const auto& names = get_action_names();
        std::vector<std::string> actions;
        actions.reserve(ids.size());
        for (int id : ids) {
            actions.push_back(names[id]);
        }
        return actions;
    }

    // ========================================================================
    // Step Game State
    // Returns reward for the player who just took action_id (+1.0, -1.0, 0.0)
    // ========================================================================

    float step(uint8_t action_id) {
        if (is_over) {
            return 0.0f;
        }

        int8_t acting_player = current_player;
        int8_t opponent = 1 - acting_player;
        int8_t color_before = active_color;
        int32_t stack_before = stack;

        turn_count += 1;
        int32_t cards_available_before = get_available_draw_count();
        uint8_t actual_drawn = 0;

        if (action_id == 60) { // "draw"
            if (stack > 0) {
                auto drawn = draw_cards(stack, acting_player);
                actual_drawn = static_cast<uint8_t>(std::min(16, (int)drawn.size()));
                stack = 0;
            } else {
                auto drawn = draw_cards(1, acting_player);
                actual_drawn = static_cast<uint8_t>(std::min(16, (int)drawn.size()));
            }

            if (cards_available_before == 0) {
                consecutive_passes += 1;
                if (consecutive_passes >= 2) {
                    _end_by_card_count();
                }
            } else {
                consecutive_passes = 0;
            }

            if (turn_count >= MAX_TURNS && !is_over) {
                _end_by_card_count();
            }

            if (!is_over) {
                current_player = opponent;
            }
        } else {
            // Play card
            consecutive_passes = 0;
            _remove_card_from_hand(acting_player, action_id);
            discard_pile.push_back(action_id);
            played_cards.push_back(action_id);
            public_discard[action_required_card(action_id)]++;
            target = action_id;
            active_color = action_declared_color(action_id);

            // Immediate win condition: 0 cards left in hand
            // Wins immediately even if final card is +2 or +4
            if (hands[acting_player].empty()) {
                is_over = true;
                winner = acting_player;
            } else if (turn_count >= MAX_TURNS && !is_over) {
                _end_by_card_count();
            } else {
                int8_t tr = action_trait(action_id);
                if (tr == DRAW_2) {
                    stack += 2;
                    current_player = opponent;
                } else if (tr == WILD_DRAW_4) {
                    stack += 4;
                    current_player = opponent;
                } else if (tr == SKIP || tr == REVERSE) {
                    // In 1v1 official UNO: skip and reverse give acting player an immediate extra turn!
                    current_player = acting_player;
                } else {
                    current_player = opponent;
                }
            }
        }

        // Record history event in ring buffer
        HistoryEvent ev;
        ev.actor_id = acting_player;
        ev.action_id = action_id;
        ev.active_color_before = color_before;
        ev.stack_before = static_cast<uint8_t>(std::min(16, stack_before));
        ev.cards_drawn = actual_drawn;

        if (history_events.size() < 48) {
            history_events.push_back(ev);
        } else {
            // Shift ring buffer
            for (size_t i = 0; i < 47; ++i) {
                history_events[i] = history_events[i + 1];
            }
            history_events[47] = ev;
        }

        // Payoff reported strictly from the perspective of acting_player
        if (is_over) {
            if (winner == acting_player) return 1.0f;
            if (winner == opponent) return -1.0f;
            return 0.0f; // Tie / draw
        }
        return 0.0f;
    }

    std::pair<float, float> get_payoffs() const {
        if (!is_over || winner < 0) {
            return {0.0f, 0.0f};
        }
        return (winner == 0) ? std::make_pair(1.0f, -1.0f) : std::make_pair(-1.0f, 1.0f);
    }

    // ========================================================================
    // Card Features Tensor: [54, 7] float32 from current_player's view
    // [my_hand[c], public_discard[c], unseen_total[c], recycled_remaining_for_me[c],
    //  is_active_color, is_top_card, is_matching_rank]
    // ========================================================================
    void encode_card_features_actor(float* out_features, int8_t player = -1) const {
        if (player < 0) {
            player = current_player;
        }
        std::memset(out_features, 0, NUM_CARD_TYPES * 7 * sizeof(float));

        // Count my hand
        uint8_t my_hand_counts[NUM_CARD_TYPES] = {0};
        for (uint8_t c : hands[player]) {
            if (c < NUM_CARD_TYPES) {
                my_hand_counts[c]++;
            }
        }

        const auto& full_counts = get_full_108_deck_counts();
        uint8_t top_card_norm = normalize_target_to_card(target);
        int8_t top_trait = action_trait(target);

        for (int c = 0; c < NUM_CARD_TYPES; ++c) {
            float* row = out_features + c * 7;
            // 0: my_hand[c]
            row[0] = static_cast<float>(my_hand_counts[c]);
            // 1: public_discard[c]
            row[1] = static_cast<float>(public_discard[c]);
            // 2: unseen_total[c] = DECK_MAX[c] - my_hand[c] - public_discard[c]
            int unseen = (int)full_counts[c] - (int)my_hand_counts[c] - (int)public_discard[c];
            row[2] = static_cast<float>(std::max(0, unseen));
            // 3: recycled_remaining_for_me[c] = max(0, last_reshuffle_pool[c] - known_drawn_from_recycled[me][c])
            int rec_rem = (int)last_reshuffle_pool[c] - (int)known_drawn_from_recycled[player][c];
            row[3] = static_cast<float>(std::max(0, rec_rem));
            // 4: is_active_color (1.0 if card c has color == active_color)
            int8_t c_col = card_color(c);
            row[4] = (c_col == active_color && c_col != COLOR_NONE) ? 1.0f : 0.0f;
            // 5: is_top_card (1.0 if card c is exact normalized card type of top card)
            row[5] = (c == top_card_norm) ? 1.0f : 0.0f;
            // 6: is_matching_rank (1.0 if card c has same trait as top card)
            int8_t c_tr = card_trait(c);
            row[6] = (c_tr == top_trait && c_tr != TRAIT_NONE) ? 1.0f : 0.0f;
        }
    }

    // ========================================================================
    // Global Features Tensor: [10] float32
    // [my_hand_size/20.0, opp_hand_size/20.0, draw_pile_size/108.0, stack_count/16.0,
    //  is_stack_p2, is_stack_p4, turn_count/200.0, consecutive_empty_passes/2.0,
    //  opp_draws_since_reshuffle/20.0, can_draw_any]
    // ========================================================================
    void encode_global_features(float* out_features, int8_t player = -1) const {
        if (player < 0) {
            player = current_player;
        }
        int8_t opp = 1 - player;
        int8_t target_tr = action_trait(target);

        out_features[0] = static_cast<float>(hands[player].size()) / 20.0f;
        out_features[1] = static_cast<float>(hands[opp].size()) / 20.0f;
        out_features[2] = static_cast<float>(deck.size()) / 108.0f;
        out_features[3] = static_cast<float>(stack) / 16.0f;
        out_features[4] = (stack > 0 && target_tr == DRAW_2) ? 1.0f : 0.0f;
        out_features[5] = (stack > 0 && target_tr == WILD_DRAW_4) ? 1.0f : 0.0f;
        out_features[6] = static_cast<float>(turn_count) / 200.0f;
        out_features[7] = static_cast<float>(consecutive_passes) / 2.0f;
        out_features[8] = static_cast<float>(draws_since_reshuffle[opp]) / 20.0f;
        out_features[9] = (get_available_draw_count() > 0) ? 1.0f : 0.0f;
    }

    // ========================================================================
    // History Tokens: [48, 5] int32 (Oldest to Newest, padded at the start)
    // [actor_rel, action_id, active_color_before, stack_before_clamped, cards_drawn_clamped]
    // Padded tokens: actor_rel=2, action_id=61, active_color=4, stack=0, cards_drawn=0
    // ========================================================================
    void encode_history_tokens(int32_t* out_tokens, int8_t player = -1) const {
        if (player < 0) {
            player = current_player;
        }
        size_t num_events = history_events.size();
        size_t pad_count = (num_events < 48) ? (48 - num_events) : 0;

        // Leading padding tokens
        for (size_t i = 0; i < pad_count; ++i) {
            int32_t* tok = out_tokens + i * 5;
            tok[0] = 2;   // padded actor
            tok[1] = 61;  // padded action_id
            tok[2] = 4;   // padded color
            tok[3] = 0;   // padded stack
            tok[4] = 0;   // padded drawn
        }

        // Chronological events (oldest to newest)
        for (size_t i = 0; i < num_events; ++i) {
            const auto& ev = history_events[i];
            int32_t* tok = out_tokens + (pad_count + i) * 5;
            tok[0] = (ev.actor_id == player) ? 0 : 1;
            tok[1] = ev.action_id;
            tok[2] = ev.active_color_before;
            tok[3] = ev.stack_before;
            tok[4] = ev.cards_drawn;
        }
    }

    // ========================================================================
    // Privileged Opponent Hand Counts: [54] float32
    // Kept strictly separate for VRPO Centralized Critic
    // ========================================================================
    void encode_opp_hand_counts(float* out_counts, int8_t player = -1) const {
        if (player < 0) {
            player = current_player;
        }
        int8_t opp = 1 - player;
        std::memset(out_counts, 0, NUM_CARD_TYPES * sizeof(float));
        for (uint8_t c : hands[opp]) {
            if (c < NUM_CARD_TYPES) {
                out_counts[c] += 1.0f;
            }
        }
    }

    // ========================================================================
    // Backward-Compatible 167-dim observation vector matching uno-gym exactly
    // ========================================================================
    void encode_observation_vector(float* out_vec, int8_t player = -1) const {
        if (player < 0) {
            player = current_player;
        }
        int8_t opponent = 1 - player;

        std::memset(out_vec, 0, OBS_DIM * sizeof(float));

        // 1. Player's hand counts (54)
        for (uint8_t c : hands[player]) {
            if (c < NUM_CARD_TYPES) {
                out_vec[c] += 1.0f;
            }
        }
        for (int i = 0; i < NUM_CARD_TYPES; ++i) {
            out_vec[i] /= 2.0f;
        }

        // 2. Target card one-hot (54)
        uint8_t target_card = normalize_target_to_card(target);
        if (target_card < NUM_CARD_TYPES) {
            out_vec[NUM_CARD_TYPES + target_card] = 1.0f;
        }

        int offset = NUM_CARD_TYPES * 2; // 108

        // 3. Active stack (1)
        out_vec[offset++] = static_cast<float>(stack) / 16.0f; // 108

        // 4. Hand sizes (2)
        out_vec[offset++] = static_cast<float>(hands[player].size()) / 108.0f;   // 109
        out_vec[offset++] = static_cast<float>(hands[opponent].size()) / 108.0f; // 110

        // 5. Unseen card counts (54)
        const auto& full_counts = get_full_108_deck_counts();
        int16_t unseen[NUM_CARD_TYPES];
        for (int i = 0; i < NUM_CARD_TYPES; ++i) {
            unseen[i] = full_counts[i];
        }

        for (uint8_t c : hands[player]) {
            if (c < NUM_CARD_TYPES && unseen[c] > 0) {
                unseen[c]--;
            }
        }
        for (uint8_t act : played_cards) {
            uint8_t c = action_required_card(act);
            if (c < NUM_CARD_TYPES && unseen[c] > 0) {
                unseen[c]--;
            }
        }

        for (int i = 0; i < NUM_CARD_TYPES; ++i) {
            out_vec[offset + i] = static_cast<float>(unseen[i]) / 4.0f;
        }
        offset += NUM_CARD_TYPES; // 165

        // 6. Public discards since last reshuffle (1)
        out_vec[offset++] = static_cast<float>(played_cards.size()) / 108.0f; // 165

        // 7. Player index (1)
        out_vec[offset++] = static_cast<float>(player); // 166
    }

    // ========================================================================
    // Explicit State Loader (Powers Differential Tests & Milestone 5 Search)
    // ========================================================================
    void load_state(
        const std::vector<uint8_t>& h0,
        const std::vector<uint8_t>& h1,
        const std::vector<uint8_t>& draw_p,
        const std::vector<uint8_t>& discard_p,
        uint8_t top_card_act,
        int8_t act_color,
        int32_t stk,
        int8_t curr_player,
        int32_t turns,
        int32_t empty_passes
    ) {
        hands[0] = h0;
        hands[1] = h1;
        deck = draw_p;
        discard_pile = discard_p;
        target = top_card_act;
        active_color = act_color;
        stack = stk;
        current_player = curr_player;
        turn_count = turns;
        consecutive_passes = empty_passes;

        // Recompute public discard counts
        std::memset(public_discard, 0, sizeof(public_discard));
        for (uint8_t act : discard_pile) {
            uint8_t c = action_required_card(act);
            if (c < NUM_CARD_TYPES) {
                public_discard[c]++;
            }
        }
        played_cards = discard_pile;

        // Check if loaded state is already terminal
        is_over = (hands[0].empty() || hands[1].empty() || turn_count >= MAX_TURNS || consecutive_passes >= 2);
        if (is_over) {
            if (hands[0].empty()) winner = 0;
            else if (hands[1].empty()) winner = 1;
            else _end_by_card_count();
        } else {
            winner = -1;
        }
    }

    // World Setter for Test-Time Search & Endgame Solvers
    void set_world(int8_t opp_player, const std::vector<uint8_t>& opp_hand, const std::vector<uint8_t>& draw_pile) {
        hands[opp_player] = opp_hand;
        deck = draw_pile;
        is_over = (hands[0].empty() || hands[1].empty() || turn_count >= MAX_TURNS || consecutive_passes >= 2);
        if (is_over) {
            if (hands[0].empty()) winner = 0;
            else if (hands[1].empty()) winner = 1;
            else _end_by_card_count();
        } else {
            winner = -1;
        }
    }

    std::vector<int> get_hand_ids(int8_t player) const {
        return std::vector<int>(hands[player].begin(), hands[player].end());
    }

    std::vector<int> get_deck_ids() const {
        return std::vector<int>(deck.begin(), deck.end());
    }

    std::vector<int> get_discard_pile_ids() const {
        return std::vector<int>(discard_pile.begin(), discard_pile.end());
    }

    // String-based state loader for Python integration
    void load_state_from_strings(
        const std::vector<std::string>& h0,
        const std::vector<std::string>& h1,
        const std::vector<std::string>& draw_p,
        const std::vector<std::string>& discard_p,
        const std::string& top_act_str,
        int8_t act_color,
        int32_t stk,
        int8_t curr_player,
        int32_t turns,
        int32_t empty_passes
    ) {
        std::vector<uint8_t> h0_ids, h1_ids, deck_ids, disc_ids;
        h0_ids.reserve(h0.size());
        for (const auto& s : h0) h0_ids.push_back(card_name_to_id(s));

        h1_ids.reserve(h1.size());
        for (const auto& s : h1) h1_ids.push_back(card_name_to_id(s));

        deck_ids.reserve(draw_p.size());
        for (const auto& s : draw_p) deck_ids.push_back(card_name_to_id(s));

        disc_ids.reserve(discard_p.size());
        for (const auto& s : discard_p) disc_ids.push_back(action_name_to_id(s));

        uint8_t top_act = action_name_to_id(top_act_str);
        load_state(h0_ids, h1_ids, deck_ids, disc_ids, top_act, act_color, stk, curr_player, turns, empty_passes);
    }

    // ========================================================================
    // String Inspection Helpers
    // ========================================================================
    std::vector<std::string> get_hands_str(int8_t player) const {
        const auto& names = get_card_names();
        std::vector<std::string> res;
        res.reserve(hands[player].size());
        for (uint8_t c : hands[player]) {
            res.push_back(names[c]);
        }
        return res;
    }

    std::string get_target_str() const {
        return get_action_names()[target];
    }

    std::vector<std::string> get_deck_str() const {
        const auto& names = get_card_names();
        std::vector<std::string> res;
        res.reserve(deck.size());
        for (uint8_t c : deck) {
            res.push_back(names[c]);
        }
        return res;
    }

    std::vector<std::string> get_discard_pile_str() const {
        const auto& names = get_action_names();
        std::vector<std::string> res;
        res.reserve(discard_pile.size());
        for (uint8_t c : discard_pile) {
            res.push_back(names[c]);
        }
        return res;
    }

    std::vector<std::string> get_played_cards_str() const {
        const auto& names = get_action_names();
        std::vector<std::string> res;
        res.reserve(played_cards.size());
        for (uint8_t c : played_cards) {
            res.push_back(names[c]);
        }
        return res;
    }
    std::vector<int32_t> get_public_discard() const {
        std::vector<int32_t> res(NUM_CARD_TYPES);
        for (int i = 0; i < NUM_CARD_TYPES; ++i) res[i] = public_discard[i];
        return res;
    }

    std::vector<int32_t> get_last_reshuffle_pool() const {
        std::vector<int32_t> res(NUM_CARD_TYPES);
        for (int i = 0; i < NUM_CARD_TYPES; ++i) res[i] = last_reshuffle_pool[i];
        return res;
    }

    std::vector<int32_t> get_recycled_remaining_for_me(int8_t player = -1) const {
        if (player < 0) player = current_player;
        std::vector<int32_t> res(NUM_CARD_TYPES);
        for (int i = 0; i < NUM_CARD_TYPES; ++i) {
            int rem = (int)last_reshuffle_pool[i] - (int)known_drawn_from_recycled[player][i];
            res[i] = std::max(0, rem);
        }
        return res;
    }

    int32_t get_draws_since_reshuffle(int8_t player = -1) const {
        if (player < 0) player = current_player;
        return draws_since_reshuffle[player];
    }

    void set_history_events(const std::vector<std::vector<int32_t>>& events) {
        history_events.clear();
        for (const auto& ev_data : events) {
            if (ev_data.size() < 5) continue;
            HistoryEvent ev;
            ev.actor_id = static_cast<int8_t>(ev_data[0]);
            ev.action_id = static_cast<uint8_t>(ev_data[1]);
            ev.active_color_before = static_cast<int8_t>(ev_data[2]);
            ev.stack_before = static_cast<uint8_t>(ev_data[3]);
            ev.cards_drawn = static_cast<uint8_t>(ev_data[4]);
            if (history_events.size() < 48) {
                history_events.push_back(ev);
            } else {
                for (size_t i = 0; i < 47; ++i) {
                    history_events[i] = history_events[i + 1];
                }
                history_events[47] = ev;
            }
        }
    }

    void set_reshuffle_provenance(
        const std::vector<int32_t>& pool,
        int32_t d0,
        int32_t d1,
        const std::vector<int32_t>& k0,
        const std::vector<int32_t>& k1
    ) {
        for (int i = 0; i < NUM_CARD_TYPES && i < (int)pool.size(); ++i) {
            last_reshuffle_pool[i] = static_cast<uint8_t>(pool[i]);
        }
        draws_since_reshuffle[0] = d0;
        draws_since_reshuffle[1] = d1;
        for (int i = 0; i < NUM_CARD_TYPES && i < (int)k0.size(); ++i) {
            known_drawn_from_recycled[0][i] = static_cast<uint8_t>(k0[i]);
        }
        for (int i = 0; i < NUM_CARD_TYPES && i < (int)k1.size(); ++i) {
            known_drawn_from_recycled[1][i] = static_cast<uint8_t>(k1[i]);
        }
    }
};

} // namespace uno
