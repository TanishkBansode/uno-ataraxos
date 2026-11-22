// In-Browser Authoritative 108-Card Official UNO Game Engine with Ataraxos Feature Tracking
import { CONFIG } from "./config.js";
import {
  AtaraxosFeatureTracker,
  CARD_NAME_TO_ID,
  ACTION_NAME_TO_ID,
  CARD_NAMES,
  ACTION_NAMES,
  normalizeTargetToCard,
  actionRequiredCard,
} from "./ataraxos-features.js";
import { FastUnoEnv } from "./fast-rollout.js";

export class UnoGameEngine {
  constructor() {
    this.deck = [];
    this.discardPile = [];
    this.hands = {
      human: {}, // card_id -> { color, type }
      bot: {},
    };
    this.currentCard = null; // { color, type }
    this.currentMove = "human"; // 'human' | 'bot'
    this.stack = 0;
    this.direction = 1; // 1 | -1 (in 2-player, reverse acts as skip)
    this.turnCount = 0;
    this.consecutivePasses = 0;
    this.isStarted = false;
    this.winner = null;
    this.isChoosingColor = false;
    this.pendingColorPicker = null; // 'human' | 'bot'
    this.unoCandidate = null;
    this.playedCards = []; // RL format history: ['r-0', 'g-3', ...] since last reshuffle

    // Ataraxos 108-card & History Ring Buffer Tracker matching C++ 1-to-1
    this.tracker = new AtaraxosFeatureTracker();

    this.drawEvidence = { human: {}, bot: {} };
    this.listeners = {};
  }

  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    this.listeners[event].push(callback);
  }

  emit(event, data) {
    if (this.listeners[event]) {
      for (const cb of this.listeners[event]) {
        cb(data);
      }
    }
  }

  static buildOfficialDeck() {
    const deck = [];
    const numberTypes = [
      "ZERO",
      "ONE",
      "TWO",
      "THREE",
      "FOUR",
      "FIVE",
      "SIX",
      "SEVEN",
      "EIGHT",
      "NINE",
    ];

    for (const color of CONFIG.COLORS) {
      deck.push({ color, type: "ZERO" });
      for (let i = 1; i <= 9; i++) {
        deck.push({ color, type: numberTypes[i] });
        deck.push({ color, type: numberTypes[i] });
      }
      deck.push({ color, type: "REVERSE" });
      deck.push({ color, type: "REVERSE" });
      deck.push({ color, type: "BLOCK" });
      deck.push({ color, type: "BLOCK" });
      deck.push({ color, type: "PLUS_TWO" });
      deck.push({ color, type: "PLUS_TWO" });
    }

    for (let i = 0; i < 4; i++) {
      deck.push({ color: "ANY", type: "COLOR_CHANGE" });
      deck.push({ color: "ANY", type: "PLUS_FOUR" });
    }

    return deck;
  }

  static shuffle(deck) {
    const arr = [...deck];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  static generateId() {
    return (
      "c_" + Math.random().toString(36).substring(2, 11) + "_" + Date.now()
    );
  }

  cardToRl(card) {
    if (!card) return "r-0";
    const color = card.color || "RED";
    const type = card.type || "ZERO";
    if (color === "ANY") {
      return CONFIG.TYPE_WEB_TO_RL[type] || "wild";
    }
    const rlCol = CONFIG.COLOR_WEB_TO_RL[color] || "r";
    const rlTrait = CONFIG.TYPE_WEB_TO_RL[type] || "0";
    return `${rlCol}-${rlTrait}`;
  }

  cardToId(card) {
    if (!card) return 0;
    const rlStr = this.cardToRl(card);
    if (rlStr === "wild") return 52;
    if (rlStr === "wild_draw_4") return 53;
    const cid = CARD_NAME_TO_ID[rlStr];
    return cid !== undefined ? cid : 0;
  }

  actionToId(card, chosenColor = null) {
    if (!card) return 60;
    const color = chosenColor || card.color || "RED";
    const rlCol = CONFIG.COLOR_WEB_TO_RL[color] || "r";
    if (card.type === "COLOR_CHANGE") {
      return ACTION_NAME_TO_ID[`${rlCol}-wild`] ?? 52;
    }
    if (card.type === "PLUS_FOUR") {
      return ACTION_NAME_TO_ID[`${rlCol}-wild_draw_4`] ?? 56;
    }
    const rlStr = this.cardToRl(card);
    return ACTION_NAME_TO_ID[rlStr] ?? 0;
  }

  startNewGame() {
    this.deck = UnoGameEngine.shuffle(UnoGameEngine.buildOfficialDeck());
    this.discardPile = [];
    this.hands = { human: {}, bot: {} };
    this.stack = 0;
    this.direction = 1;
    this.turnCount = 0;
    this.consecutivePasses = 0;
    this.isStarted = true;
    this.winner = null;
    this.isChoosingColor = false;
    this.pendingColorPicker = null;
    this.unoCandidate = null;
    this.playedCards = [];
    this.drawEvidence = { human: {}, bot: {} };
    this.currentCard = null;

    // Deal 7 cards to each player
    for (let i = 0; i < CONFIG.START_CARDS; i++) {
      const hCard = this.drawRawCard();
      const bCard = this.drawRawCard();
      this.hands.human[UnoGameEngine.generateId()] = hCard;
      this.hands.bot[UnoGameEngine.generateId()] = bCard;
    }

    // Draw starting non-special number card (0-9)
    const numberTypes = [
      "ZERO",
      "ONE",
      "TWO",
      "THREE",
      "FOUR",
      "FIVE",
      "SIX",
      "SEVEN",
      "EIGHT",
      "NINE",
    ];
    while (this.deck.length > 0) {
      const candidate = this.deck.pop();
      if (numberTypes.includes(candidate.type)) {
        this.currentCard = candidate;
        this.discardPile = [candidate];
        break;
      }
      this.deck.unshift(candidate);
    }

    if (!this.currentCard) {
      this.currentCard = { color: "RED", type: "ZERO" };
      this.discardPile = [this.currentCard];
    }

    this.playedCards = [this.cardToRl(this.currentCard)];
    this.currentMove = "human"; // Human always starts

    // Initialize Ataraxos Feature Tracker
    const startCardId = this.cardToId(this.currentCard);
    this.tracker.reset(startCardId);

    this.emit("game_started", {
      currentCard: this.currentCard,
      currentMove: this.currentMove,
      humanCards: this.hands.human,
      botCardCount: Object.keys(this.hands.bot).length,
      stack: this.stack,
    });
  }

  drawRawCard() {
    if (this.deck.length === 0) {
      this.recycleDiscardPile();
    }
    if (this.deck.length === 0) {
      return null;
    }
    return this.deck.pop();
  }

  recycleDiscardPile() {
    if (this.discardPile.length <= 1) {
      return;
    }
    const activeTop = this.discardPile[this.discardPile.length - 1];
    const recyclable = this.discardPile.slice(0, this.discardPile.length - 1);
    this.discardPile = [activeTop];

    // Reset wilds back to ANY
    for (const card of recyclable) {
      if (card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR") {
        card.color = "ANY";
      }
    }

    this.deck = UnoGameEngine.shuffle(recyclable);
    this.playedCards = [this.cardToRl(activeTop)];

    const activeTopActionId = this.actionToId(activeTop);
    const recycledCards = recyclable.map((c) => this.actionToId(c));
    this.tracker.recordReshuffle(activeTopActionId, recycledCards);

    this.emit("deck_reshuffled", {
      deckCount: this.deck.length,
      currentCard: this.currentCard,
    });
  }

  getAvailableDrawCount() {
    return this.deck.length + Math.max(0, this.discardPile.length - 1);
  }

  canPlayCard(card, playerId) {
    const curr = this.currentCard;
    if (!card || !curr) return false;

    // When facing an active stack, strictly identical draw type counters
    if (this.stack > 0) {
      if (curr.type === "PLUS_TWO") {
        return card.type === "PLUS_TWO";
      }
      if (curr.type === "PLUS_FOUR") {
        return card.type === "PLUS_FOUR";
      }
      return false;
    }

    // Wild Draw 4 is always playable on a normal turn.
    if (card.type === "PLUS_FOUR" || card.type === "COLOR_CHANGE") {
      return true;
    }

    return card.color === curr.color || card.type === curr.type;
  }

  getLegalActionsForPlayer(playerId) {
    const hand = this.hands[playerId];
    const items = []; // { cardId, actionStr, pickColor }

    for (const [cid, card] of Object.entries(hand)) {
      if (this.canPlayCard(card, playerId)) {
        if (card.type === "COLOR_CHANGE") {
          for (const clr of CONFIG.RL_COLORS) {
            items.push({
              cardId: cid,
              actionStr: `${clr}-wild`,
              pickColor: CONFIG.COLOR_RL_TO_WEB[clr],
            });
          }
        } else if (card.type === "PLUS_FOUR") {
          for (const clr of CONFIG.RL_COLORS) {
            items.push({
              cardId: cid,
              actionStr: `${clr}-wild_draw_4`,
              pickColor: CONFIG.COLOR_RL_TO_WEB[clr],
            });
          }
        } else {
          const rlCol = CONFIG.COLOR_WEB_TO_RL[card.color] || "r";
          const rlTrait = CONFIG.TYPE_WEB_TO_RL[card.type] || "0";
          items.push({
            cardId: cid,
            actionStr: `${rlCol}-${rlTrait}`,
            pickColor: null,
          });
        }
      }
    }

    const rawLegal = Array.from(new Set(items.map((it) => it.actionStr)));
    const availDraw = this.getAvailableDrawCount();
    if (availDraw > 0 || items.length === 0) {
      rawLegal.push("draw");
    }

    return { items, rawLegal };
  }

  playCard(playerId, cardId, chosenColor = null) {
    if (!this.isStarted || this.isChoosingColor) {
      return { success: false, reason: "game_not_active" };
    }
    if (this.currentMove !== playerId) {
      return { success: false, reason: "not_your_turn" };
    }

    const card = this.hands[playerId][cardId];
    if (!card) {
      return { success: false, reason: "card_not_in_hand" };
    }
    if (!this.canPlayCard(card, playerId)) {
      return { success: false, reason: "illegal_card" };
    }

    const actingPlayerIdx = playerId === "human" ? 0 : 1;
    const colorBefore = CONFIG.RL_COLORS.indexOf(
      CONFIG.COLOR_WEB_TO_RL[this.currentCard?.color] || "r",
    );
    const stackBefore = this.stack;

    // Remove card from hand
    delete this.hands[playerId][cardId];
    this.discardPile.push(card);
    this.currentCard = card;
    this.consecutivePasses = 0;
    this.turnCount++;

    // Add to public history
    this.playedCards.push(this.cardToRl(card));

    if (card.color !== "ANY") {
      const playedColor = CONFIG.COLOR_WEB_TO_RL[card.color];
      if (playedColor) {
        delete this.drawEvidence[playerId][playedColor];
      }
    }

    // Stack penalty accumulator
    if (card.type === "PLUS_TWO") {
      this.stack += 2;
    } else if (card.type === "PLUS_FOUR") {
      this.stack += 4;
    }

    const remainingCount = Object.keys(this.hands[playerId]).length;
    const isWild = card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR";
    const resolvedColor =
      isWild && chosenColor && CONFIG.COLORS.includes(chosenColor)
        ? chosenColor
        : null;

    if (resolvedColor) {
      this.applyChosenColor(resolvedColor);
    }

    // Determine action ID for tracker
    const actionId = this.actionToId(card, resolvedColor);
    this.tracker.recordPlay(actingPlayerIdx, actionId);
    this.tracker.pushHistoryEvent({
      actorId: actingPlayerIdx,
      actionId,
      activeColorBefore: colorBefore >= 0 ? colorBefore : 0,
      stackBefore,
      cardsDrawn: 0,
    });

    this.emit("card_placed", {
      playerId,
      card: this.currentCard,
      remainingCount,
      stack: this.stack,
      needsColor: isWild && !resolvedColor,
    });

    // Check for Win (Immediate Win upon emptying hand)
    if (remainingCount === 0) {
      this.winner = playerId;
      this.isStarted = false;
      this.emit("game_over", { winner: playerId, reason: "hand_cleared" });
      return { success: true, winner: playerId };
    }

    // Check for 200-turn limit
    if (this.turnCount >= CONFIG.MAX_TURNS) {
      this.endByCardCount();
      return { success: true, winner: this.winner };
    }

    // Auto-call UNO when 1 card left
    if (remainingCount === 1) {
      this.unoCandidate = null;
      this.emit("uno_called", { playerId, auto: true });
    } else {
      this.unoCandidate = null;
    }

    // Wild color selection check
    if (isWild) {
      if (resolvedColor) {
        this.isChoosingColor = false;
        this.pendingColorPicker = null;
        this.emit("color_changed", {
          color: resolvedColor,
          currentCard: this.currentCard,
        });
        this.advanceTurnAfterPlay(playerId, card.type);
      } else {
        this.isChoosingColor = true;
        this.pendingColorPicker = playerId;
        this.emit("need_color_choice", { playerId });
        return { success: true, needColor: true };
      }
    } else {
      this.advanceTurnAfterPlay(playerId, card.type);
    }

    return { success: true, card };
  }

  applyChosenColor(color) {
    if (!CONFIG.COLORS.includes(color)) return;
    this.currentCard = { color, type: this.currentCard.type };

    if (this.discardPile.length > 0) {
      this.discardPile[this.discardPile.length - 1] = { ...this.currentCard };
    }

    if (this.playedCards.length > 0) {
      const lastIdx = this.playedCards.length - 1;
      const rlColor = CONFIG.COLOR_WEB_TO_RL[color];
      this.playedCards[lastIdx] = `${rlColor}-${this.currentCard.type.toLowerCase()}`;
    }
  }

  executeColorChange(playerId, color) {
    if (!CONFIG.COLORS.includes(color)) return;
    this.isChoosingColor = false;
    this.pendingColorPicker = null;
    this.applyChosenColor(color);
    this.emit("color_changed", { color, currentCard: this.currentCard });
    this.advanceTurnAfterPlay(playerId, this.currentCard.type);
  }

  advanceTurnAfterPlay(playerId, cardType) {
    const opponentId = playerId === "human" ? "bot" : "human";

    // 1v1 UNO Rules: Reverse and Block give the current player another turn!
    if (cardType === "BLOCK" || cardType === "REVERSE") {
      this.currentMove = playerId; // Extra turn!
      this.emit("turn_changed", {
        currentMove: this.currentMove,
        extraTurn: true,
        currentCard: this.currentCard,
        stack: this.stack,
      });
    } else {
      this.currentMove = opponentId;
      this.emit("turn_changed", {
        currentMove: this.currentMove,
        extraTurn: false,
        currentCard: this.currentCard,
        stack: this.stack,
      });
    }
  }

  drawCard(playerId) {
    if (!this.isStarted || this.isChoosingColor) {
      return { success: false, reason: "game_not_active" };
    }
    if (this.currentMove !== playerId) {
      return { success: false, reason: "not_your_turn" };
    }

    const actingPlayerIdx = playerId === "human" ? 0 : 1;
    const colorBefore = CONFIG.RL_COLORS.indexOf(
      CONFIG.COLOR_WEB_TO_RL[this.currentCard?.color] || "r",
    );
    const stackBefore = this.stack;
    const isTakingStack = this.stack > 0;
    const takeCount = isTakingStack ? this.stack : 1;
    const cardsDrawn = {};

    const cardsAvailBefore = this.getAvailableDrawCount();
    const opponentId = playerId === "human" ? "bot" : "human";

    // Draw cards
    for (let i = 0; i < takeCount; i++) {
      if (Object.keys(this.hands[playerId]).length >= CONFIG.MAX_CARDS) {
        break;
      }
      const card = this.drawRawCard();
      if (!card) break;
      const cid = UnoGameEngine.generateId();
      this.hands[playerId][cid] = card;
      cardsDrawn[cid] = card;
    }

    const actualDrawnCount = Object.keys(cardsDrawn).length;
    const drawnCardIds = Object.values(cardsDrawn).map((c) => this.cardToId(c));

    this.stack = 0;
    this.turnCount++;

    // Track provenance in Ataraxos feature tracker
    this.tracker.recordDraws(actingPlayerIdx, drawnCardIds, actualDrawnCount);
    this.tracker.pushHistoryEvent({
      actorId: actingPlayerIdx,
      actionId: 60,
      activeColorBefore: colorBefore >= 0 ? colorBefore : 0,
      stackBefore,
      cardsDrawn: actualDrawnCount,
    });

    const drawColor = CONFIG.COLOR_WEB_TO_RL[this.currentCard?.color];
    if (drawColor && actualDrawnCount > 0) {
      this.drawEvidence[playerId][drawColor] = this.turnCount;
    }

    // Consecutive pass detection when deck is empty
    if (cardsAvailBefore === 0) {
      this.consecutivePasses++;
      if (this.consecutivePasses >= 2) {
        this.endByCardCount();
        return { success: true, cardsDrawn, gameOver: true };
      }
    } else {
      this.consecutivePasses = 0;
    }

    // Safety 200-turn limit check on draw
    if (this.turnCount >= CONFIG.MAX_TURNS) {
      this.endByCardCount();
      return { success: true, cardsDrawn, gameOver: true };
    }

    // Pass turn to opponent (standard UNO "keep on draw")
    this.currentMove = opponentId;

    this.emit("cards_drawn", {
      playerId,
      cardsDrawn,
      newCount: Object.keys(this.hands[playerId]).length,
      nextMove: this.currentMove,
      stack: 0,
    });

    this.emit("turn_changed", {
      currentMove: this.currentMove,
      currentCard: this.currentCard,
      stack: 0,
    });

    return { success: true, cardsDrawn };
  }

  getOpponentDrawRecency(playerId) {
    const opponentId = playerId === "human" ? "bot" : "human";
    const recency = {};
    for (const color of CONFIG.RL_COLORS) {
      const drawnAt = this.drawEvidence[opponentId][color];
      recency[color] = drawnAt === undefined ? -1 : this.turnCount - drawnAt;
    }
    return recency;
  }

  callUno(callingPlayerId) {
    if (!this.isStarted) return;
    this.emit("uno_called", { playerId: callingPlayerId, success: true });
  }

  endByCardCount() {
    this.isStarted = false;
    const humanCount = Object.keys(this.hands.human).length;
    const botCount = Object.keys(this.hands.bot).length;

    if (humanCount < botCount) {
      this.winner = "human";
    } else if (botCount < humanCount) {
      this.winner = "bot";
    } else {
      this.winner = "DRAW";
    }

    this.emit("game_over", {
      winner: this.winner,
      reason: "max_turns_or_deck_exhausted",
      humanCount,
      botCount,
    });
  }

  /**
   * Build 100% exact feature tensors and structured state payload matching C++ UnoEngine.
   */
  getAtaraxosFeatures(playerId = "bot") {
    const playerIdx = playerId === "human" ? 0 : 1;
    const handObj = this.hands[playerId];
    const handCardIds = Object.values(handObj).map((c) => this.cardToId(c));
    const targetActionId = this.actionToId(this.currentCard);
    const activeColor = CONFIG.RL_COLORS.indexOf(
      CONFIG.COLOR_WEB_TO_RL[this.currentCard?.color] || "r",
    );
    const availDraw = this.getAvailableDrawCount();

    const humanHandIds = Object.values(this.hands.human).map((c) => this.cardToId(c));
    const botHandIds = Object.values(this.hands.bot).map((c) => this.cardToId(c));

    const cardFeatures = this.tracker.encodeCardFeatures(
      playerIdx,
      handCardIds,
      targetActionId,
      activeColor >= 0 ? activeColor : 0,
    );

    const globalFeatures = this.tracker.encodeGlobalFeatures({
      player: playerIdx,
      hand0Size: humanHandIds.length,
      hand1Size: botHandIds.length,
      deckSize: this.deck.length,
      stack: this.stack,
      consecutivePasses: this.consecutivePasses,
      turnCount: this.turnCount,
      targetActionId,
      availableDrawCount: availDraw,
    });

    const historyTokens = this.tracker.encodeHistoryTokens(playerIdx, true); // BigInt64Array

    const legalMask = this.tracker.encodeLegalMask({
      handCardIds,
      targetActionId,
      activeColor: activeColor >= 0 ? activeColor : 0,
      stack: this.stack,
      availableDrawCount: availDraw,
    });

    const legalActionIds = [];
    const legalActionNames = [];
    for (let i = 0; i < 61; i++) {
      if (legalMask[i] > 0.5) {
        legalActionIds.push(i);
        legalActionNames.push(ACTION_NAMES[i]);
      }
    }

    const { items, rawLegal } = this.getLegalActionsForPlayer(playerId);

    const statePayload = {
      hands_0: humanHandIds,
      hands_1: botHandIds,
      deck_count: this.deck.length,
      discard_pile: this.discardPile.map((c) => this.actionToId(c)),
      target: targetActionId,
      active_color: activeColor >= 0 ? activeColor : 0,
      stack: this.stack,
      current_player: playerIdx,
      turn_count: this.turnCount,
      consecutive_passes: this.consecutivePasses,
      history_events: this.tracker.historyEvents.map((ev) => [
        ev.actorId,
        ev.actionId,
        ev.activeColorBefore,
        ev.stackBefore,
        ev.cardsDrawn,
      ]),
      reshuffle_provenance: {
        last_reshuffle_pool: Array.from(this.tracker.lastReshufflePool),
        draws_since_reshuffle: [...this.tracker.drawsSinceReshuffle],
        known_drawn_from_recycled: [
          Array.from(this.tracker.knownDrawnFromRecycled[0]),
          Array.from(this.tracker.knownDrawnFromRecycled[1]),
        ],
      },
    };

    return {
      cardFeatures,
      globalFeatures,
      historyTokens,
      legalMask,
      legalActionIds,
      legalActionNames,
      items,
      rawLegal,
      state: statePayload,
      fastEnv: this.toFastUnoEnv(),
    };
  }

  toFastUnoEnv() {
    const env = new FastUnoEnv();
    const humanHandIds = Object.values(this.hands.human).map((c) => this.cardToId(c));
    const botHandIds = Object.values(this.hands.bot).map((c) => this.cardToId(c));
    env.hands = [humanHandIds, botHandIds];
    env.deck = this.deck.map((c) => this.cardToId(c));
    env.discardPile = this.discardPile.map((c) => this.actionToId(c));
    env.target = this.actionToId(this.currentCard);
    const activeColor = CONFIG.RL_COLORS.indexOf(
      CONFIG.COLOR_WEB_TO_RL[this.currentCard?.color] || "r"
    );
    env.activeColor = activeColor >= 0 ? activeColor : 0;
    env.stack = this.stack;
    env.currentPlayer = this.currentMove === "human" ? 0 : 1;
    env.isOver = this.winner !== null;
    env.winner = this.winner === "human" ? 0 : (this.winner === "bot" ? 1 : -1);
    env.consecutivePasses = this.consecutivePasses;
    env.turnCount = this.turnCount;
    env.tracker = this.tracker.clone();
    return env;
  }
}
