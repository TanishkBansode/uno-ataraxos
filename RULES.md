# 1v1 UNO Rules Specification

Official rules specification for `Ataraxos-UNO`.

### Rule Set Classification & Standard Adherence

The rule set implemented in `Ataraxos-UNO` is **Classic 108-Card UNO (1v1 with Same-Type Draw Stacking)**:

- **Official Mattel Rules**:
  - Exactly 108 physical cards (colored 0–9, Skip, Reverse, Draw 2, Wild, Wild Draw 4).
  - Official 2-player rule: **`Reverse = Skip`** (in 1v1, playing a Reverse acts as a Skip, giving the player an immediate extra turn).
  - Official 1v1 initial card redrawing (non-numbers buried until a number card starts the game).
- **Competitive Rule Extensions**:
  - **Same-Type Draw Stacking**: Same-type stacking (`+2` on `+2`, `+4` on `+4`) is enabled. Cross-type stacking (`+4` on `+2` or `+2` on `+4`) is prohibited.
  - **Wild Draw 4 is always playable** on a normal turn, regardless of the other cards in hand (standard color-matching restrictions are omitted).

---

## 1. Deck Composition (108 Cards)

The deck consists of exactly **108 cards**:

- **Colored Cards (100 cards)** across 4 colors: `RED`, `GREEN`, `BLUE`, `YELLOW`.
  For each of the 4 colors:
  - One `0` (4 cards total)
  - Two each of `1` through `9` (72 cards total: $2 \times 9 \times 4$)
  - Two `BLOCK` (`skip`) (8 cards total)
  - Two `REVERSE` (`reverse`) (8 cards total)
  - Two `PLUS_TWO` (`draw_2`) (8 cards total)
- **Wild Cards (8 cards)**:
  - Four `COLOR_CHANGE` (`wild`)
  - Four `PLUS_FOUR` (`wild_draw_4`)

### Deck Reshuffling (Recycling Discards)

- Drawing draws cards directly from the finite draw pile.
- When the draw pile runs out of cards during a draw:
  - The current top active card remains in place as the active discard.
  - All other cards in the discard pile are collected.
  - Any wild cards (`COLOR_CHANGE`, `PLUS_FOUR`) have their declared color reset back to neutral.
  - The collected cards are shuffled using the Fisher-Yates algorithm into a new draw pile.

---

## 2. Game Setup

- **Default Player Count**: 2 players (1v1 head-to-head match).
- **Starting Hand Size**: 7 cards dealt to each player from the shuffled 108-card deck.
- **Initial Top Card**: Drawn from the top of the deck.
  - If the initial card drawn is not a standard number card (`0`–`9`), it is buried/redrawn until an initial number card (`0`–`9`) is placed as the starting target card.

---

## 3. Turn & Card Action Rules

### Standard Plays (When `stack == 0`)

- A player may play a card if it matches the current target card's **color** or **trait/type** (number, `BLOCK`, `REVERSE`, `PLUS_TWO`).
- Regular Wild (`COLOR_CHANGE`) can be played on any normal turn (`stack == 0`). The player designates the active color. Play then passes to the next player.
- `BLOCK` (`skip`): The opponent is skipped. In 1v1, the playing player gets another turn immediately.
- `REVERSE` (`reverse`): In 2-player rules, `REVERSE` acts identically to `BLOCK`. The playing player gets another turn immediately.

---

## 4. Same-Type Draw Countering (`+2` vs `+2`, `+4` vs `+4`)

Strict separation applies: **`+2` CANNOT counter `+4`**, and **`+4` CANNOT counter `+2`**.

### Draw 2 (`PLUS_TWO`)

- **Initiation (`stack == 0`)**:
  - Playing `PLUS_TWO` adds `+2` to `stack` and passes the turn to the opponent.
- **Facing an Active `PLUS_TWO` Stack (`stack > 0`)**:
  - When the top card is `PLUS_TWO`, the player may ONLY:
    1. **Counter**: Play another `PLUS_TWO` of any color. This increases `stack` by `+2` and passes the turn back to the opponent.
    2. **Draw & End Turn**: Choose `draw`. The player draws all `stack` cards from the finite 108-card deck, `stack` resets to `0`, and their turn ends.
  - A `PLUS_FOUR` **cannot** be played on an active `PLUS_TWO` stack. No other cards (numbers, skips, reverses, regular wilds) may be played.

### Wild Draw 4 (`PLUS_FOUR`)

- **Initiation (`stack == 0`)**:
  - `PLUS_FOUR` is always legal on a normal turn, even if the player holds cards matching the active target color.
  - Playing it adds `+4` to `stack`, allows the player to choose the new active color, and passes the turn to the opponent.
- **Facing an Active `PLUS_FOUR` Stack (`stack > 0`)**:
  - When the top card is `PLUS_FOUR`, the player may ONLY:
    1. **Counter**: Play another `PLUS_FOUR` (always legal to counter a `+4` stack, regardless of colors in hand). This increases `stack` by `+4`, lets the player declare a new active color, and passes the turn back to the opponent.
    2. **Draw & End Turn**: Choose `draw`. The player draws all `stack` cards from the finite 108-card deck, `stack` resets to `0`, and their turn ends.
  - A `PLUS_TWO` **cannot** be played on an active `PLUS_FOUR` stack.

---

## 5. Prohibited Arcade Rules

The following arcade/house rules are strictly disabled:

- **Cross-Type Stacking**: No stacking `+4` on `+2` or `+2` on `+4`.
- **No Jump-In**: Players cannot play out of turn even if holding an identical card.
- **No Multi-Card Chaining**: Players cannot play multiple identical cards in one turn.
- **No Infinite 50/50 Deck**: Cards are drawn from the real, finite 108-card deck with discard recycling.

---

## 6. Draw Rules (Always Keep on Draw)

- **Standard Draw (`stack == 0`)**:
  - When `stack == 0` and a player cannot or chooses not to play a card, they draw **1 card** from the finite 108-card deck directly into their hand.
  - **Always Keep & End Turn**: Drawing a card always keeps it in hand and immediately ends the player's turn. There is no secondary "place or save" prompt or auto-play.
- **Drawing on Penalty Stack (`stack > 0`)**:
  - If a player cannot or chooses not to counter an active penalty stack (`+2` or `+4`), they draw all `stack` cards from the finite deck into their hand, `stack` resets to `0`, and their turn ends immediately.

---

## 7. Deck Exhaustion & Stalemate Rules (Deadlock & Turn Limits)

To guarantee that games always terminate and prevent infinite loops:

1. **Deck Exhaustion Condition**:
   - The draw deck is empty AND the discard pile has $\le 1$ card (the active top card), meaning 0 cards can be drawn even after recycling discards.
2. **Voluntary Draw Disabled**:
   - When 0 cards are available to draw, voluntary `draw` is **disabled** if the player holds $\ge 1$ playable cards; they must play a card.
3. **Empty Draw / Pass on Inability to Play**:
   - If a player has 0 playable cards (or is forced to draw from an active penalty stack) and 0 cards exist to be drawn, they draw as many as exist (0 cards), any active penalty stack immediately resets to `0`, and their turn passes to the opponent.
4. **Consecutive Stalemate (Deadlock)**:
   - If both players consecutively cannot play or draw (2 consecutive passes with 0 cards drawn), the game deadlocks and terminates immediately.
5. **Turn Limit**:
   - If a game reaches $200$ turns without a player emptying their hand, the game terminates immediately.
6. **Card Count Resolution**:
   - Upon stalemate or hitting the 200-turn limit, the player with **fewer cards in hand** wins.
   - If both players hold the exact same number of cards, the game is declared a **draw (tie / 0 payoff)**.

---

## 8. UNO Declaration

In this simulation and reinforcement learning environment, UNO calling is automatic. When a player plays a card leaving them with 1 card in hand, or legally plays their final card to reach 0 cards, the simulator resolves the state directly. A player wins immediately upon legally playing their last card (0 cards remaining in hand).

---

## 9. Card-Counting & Public Information

Because the deck is a finite 108-card set and discard recycling preserves public history, agents can maintain an exact count of **Unseen Cards**:
$$\text{Unseen Cards} = \text{Initial 108 Cards} - \text{My Hand} - \text{Discards since last reshuffle}$$
The unseen cards represent the exact combined pool of cards in the draw pile and opponent's hand.
