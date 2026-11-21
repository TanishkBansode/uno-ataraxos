#pragma once

#include <cstdint>
#include <string>
#include <array>
#include <vector>

namespace uno {

// ============================================================================
// Exact Card and Action ID Mappings MATCHING uno-gym 1-to-1
// ============================================================================

constexpr int NUM_COLORS = 4;
constexpr int NUM_NUMBERS = 10;
constexpr int NUM_ACTION_TRAITS = 3;
constexpr int NUM_CARD_TYPES = 54;
constexpr int NUM_ACTIONS = 61;
constexpr int TOTAL_DECK_CARDS = 108;
constexpr int OBS_DIM = 167;
constexpr int MAX_TURNS = 200;

// Color order matching uno-gym COLORS = ("r", "g", "b", "y")
enum Color : int8_t {
    RED = 0,
    GREEN = 1,
    BLUE = 2,
    YELLOW = 3,
    COLOR_NONE = -1
};

// Trait IDs
enum Trait : int8_t {
    NUM_0 = 0,
    NUM_1 = 1,
    NUM_2 = 2,
    NUM_3 = 3,
    NUM_4 = 4,
    NUM_5 = 5,
    NUM_6 = 6,
    NUM_7 = 7,
    NUM_8 = 8,
    NUM_9 = 9,
    SKIP = 10,
    REVERSE = 11,
    DRAW_2 = 12,
    WILD = 13,
    WILD_DRAW_4 = 14,
    TRAIT_NONE = -1
};

// ============================================================================
// Canonical 54 Card Names
// 0..39: {color}-{0..9}
// 40..51: {color}-{skip, reverse, draw_2}
// 52: wild
// 53: wild_draw_4
// ============================================================================
inline const std::array<std::string, 54>& get_card_names() {
    static const std::array<std::string, 54> names = []() {
        std::array<std::string, 54> arr;
        const char colors[] = {'r', 'g', 'b', 'y'};
        int idx = 0;
        // Numbers: 4 colors * 10 = 40
        for (int c = 0; c < 4; ++c) {
            for (int n = 0; n < 10; ++n) {
                arr[idx++] = std::string(1, colors[c]) + "-" + std::to_string(n);
            }
        }
        // Action traits: 4 colors * 3 = 12
        const char* traits[] = {"skip", "reverse", "draw_2"};
        for (int c = 0; c < 4; ++c) {
            for (int t = 0; t < 3; ++t) {
                arr[idx++] = std::string(1, colors[c]) + "-" + traits[t];
            }
        }
        // Wilds
        arr[idx++] = "wild";         // 52
        arr[idx++] = "wild_draw_4";  // 53
        return arr;
    }();
    return names;
}

// ============================================================================
// Canonical 61 Action Names
// 0..39: {color}-{0..9}
// 40..51: {color}-{skip, reverse, draw_2}
// 52..55: {color}-wild
// 56..59: {color}-wild_draw_4
// 60: draw
// ============================================================================
inline const std::array<std::string, 61>& get_action_names() {
    static const std::array<std::string, 61> names = []() {
        std::array<std::string, 61> arr;
        const char colors[] = {'r', 'g', 'b', 'y'};
        int idx = 0;
        // 1. Colored number cards: 4 colors * 10 numbers (0-9) = 40 actions
        for (int c = 0; c < 4; ++c) {
            for (int n = 0; n < 10; ++n) {
                arr[idx++] = std::string(1, colors[c]) + "-" + std::to_string(n);
            }
        }
        // 2. Colored action cards: 4 colors * 3 traits (skip, reverse, draw_2) = 12 actions
        const char* traits[] = {"skip", "reverse", "draw_2"};
        for (int c = 0; c < 4; ++c) {
            for (int t = 0; t < 3; ++t) {
                arr[idx++] = std::string(1, colors[c]) + "-" + traits[t];
            }
        }
        // 3. Wild card with color declaration: 4 colors = 4 actions
        for (int c = 0; c < 4; ++c) {
            arr[idx++] = std::string(1, colors[c]) + "-wild";
        }
        // 4. Wild Draw 4 card with color declaration: 4 colors = 4 actions
        for (int c = 0; c < 4; ++c) {
            arr[idx++] = std::string(1, colors[c]) + "-wild_draw_4";
        }
        // 5. Draw action = 1 action
        arr[idx++] = "draw";
        return arr;
    }();
    return names;
}

// ============================================================================
// Card properties and conversion helpers
// ============================================================================

inline int8_t card_color(uint8_t card_id) {
    if (card_id < 40) return static_cast<int8_t>(card_id / 10);
    if (card_id < 52) return static_cast<int8_t>((card_id - 40) / 3);
    return COLOR_NONE; // 52 (wild), 53 (wild_draw_4)
}

inline int8_t card_trait(uint8_t card_id) {
    if (card_id < 40) return static_cast<int8_t>(card_id % 10);
    if (card_id < 52) return static_cast<int8_t>(10 + ((card_id - 40) % 3));
    if (card_id == 52) return WILD;
    if (card_id == 53) return WILD_DRAW_4;
    return TRAIT_NONE;
}

inline bool card_is_number(uint8_t card_id) {
    return card_id < 40;
}

// ============================================================================
// Action properties and conversion helpers
// ============================================================================

inline int8_t action_declared_color(uint8_t action_id) {
    if (action_id < 40) return static_cast<int8_t>(action_id / 10);
    if (action_id < 52) return static_cast<int8_t>((action_id - 40) / 3);
    if (action_id < 56) return static_cast<int8_t>(action_id - 52);
    if (action_id < 60) return static_cast<int8_t>(action_id - 56);
    return COLOR_NONE; // 60: draw
}

inline int8_t action_trait(uint8_t action_id) {
    if (action_id < 40) return static_cast<int8_t>(action_id % 10);
    if (action_id < 52) return static_cast<int8_t>(10 + ((action_id - 40) % 3));
    if (action_id < 56) return WILD;
    if (action_id < 60) return WILD_DRAW_4;
    return TRAIT_NONE; // 60: draw
}

// The card type (0..53) that is consumed from the player's hand by this action
inline int8_t action_required_card(uint8_t action_id) {
    if (action_id < 52) return static_cast<int8_t>(action_id);
    if (action_id < 56) return 52; // wild
    if (action_id < 60) return 53; // wild_draw_4
    return -1; // 60: draw
}

// Target normalization: uno-gym maps "c-wild" -> "wild" (52), "c-wild_draw_4" -> "wild_draw_4" (53)
inline uint8_t normalize_target_to_card(uint8_t target_action_id) {
    if (target_action_id < 52) return target_action_id;
    if (target_action_id < 56) return 52; // wild
    if (target_action_id < 60) return 53; // wild_draw_4
    return 0;
}

// ============================================================================
// Full 108-Card Deck Composition
// ============================================================================
inline const std::array<uint8_t, 54>& get_full_108_deck_counts() {
    static const std::array<uint8_t, 54> counts = []() {
        std::array<uint8_t, 54> arr{};
        for (int c = 0; c < 4; ++c) {
            // One 0 per color
            arr[c * 10 + 0] = 1;
            // Two 1-9 per color
            for (int n = 1; n < 10; ++n) {
                arr[c * 10 + n] = 2;
            }
            // Two each of skip, reverse, draw_2 per color
            arr[40 + c * 3 + 0] = 2; // skip
            arr[40 + c * 3 + 1] = 2; // reverse
            arr[40 + c * 3 + 2] = 2; // draw_2
        }
        arr[52] = 4; // wild
        arr[53] = 4; // wild_draw_4
        return arr;
    }();
    return counts;
}

// List of all 108 cards in canonical deck generation order matching build_108_deck()
inline std::vector<uint8_t> build_108_deck_vector() {
    std::vector<uint8_t> deck;
    deck.reserve(TOTAL_DECK_CARDS);
    for (int c = 0; c < 4; ++c) {
        deck.push_back(c * 10 + 0);
        for (int n = 1; n < 10; ++n) {
            deck.push_back(c * 10 + n);
            deck.push_back(c * 10 + n);
        }
    }
    for (int c = 0; c < 4; ++c) {
        for (int t = 0; t < 3; ++t) {
            deck.push_back(40 + c * 3 + t);
            deck.push_back(40 + c * 3 + t);
        }
    }
    for (int i = 0; i < 4; ++i) {
        deck.push_back(52); // wild
        deck.push_back(53); // wild_draw_4
    }
    return deck;
}

// Lookups by string name
inline int card_name_to_id(const std::string& name) {
    const auto& names = get_card_names();
    for (int i = 0; i < 54; ++i) {
        if (names[i] == name) return i;
    }
    return -1;
}

inline int action_name_to_id(const std::string& name) {
    const auto& names = get_action_names();
    for (int i = 0; i < 61; ++i) {
        if (names[i] == name) return i;
    }
    return -1;
}

} // namespace uno
