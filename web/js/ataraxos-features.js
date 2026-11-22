// Exact Ataraxos UNO Feature Encoder & Provenance Tracker matching cpp/uno_engine.hpp 1-to-1
// Provides:
// - Exact 48-move ring buffer with standard [2, 61, 4, 0, 0] leading padding
// - Full card counting and discard recycling provenance tracking
// - card_features [54, 7], global_features [10], history_tokens [48, 5], legal_mask [61]

export const NUM_COLORS = 4;
export const NUM_CARD_TYPES = 54;
export const NUM_ACTIONS = 61;
export const TOTAL_DECK_CARDS = 108;
export const MAX_TURNS = 200;

export const COLOR = {
  RED: 0,
  GREEN: 1,
  BLUE: 2,
  YELLOW: 3,
  NONE: -1,
};

export const TRAIT = {
  NUM_0: 0,
  NUM_1: 1,
  NUM_2: 2,
  NUM_3: 3,
  NUM_4: 4,
  NUM_5: 5,
  NUM_6: 6,
  NUM_7: 7,
  NUM_8: 8,
  NUM_9: 9,
  SKIP: 10,
  REVERSE: 11,
  DRAW_2: 12,
  WILD: 13,
  WILD_DRAW_4: 14,
  NONE: -1,
};

export const CARD_NAMES = (() => {
  const arr = new Array(54);
  const colors = ["r", "g", "b", "y"];
  let idx = 0;
  for (let c = 0; c < 4; c++) {
    for (let n = 0; n < 10; n++) {
      arr[idx++] = `${colors[c]}-${n}`;
    }
  }
  const traits = ["skip", "reverse", "draw_2"];
  for (let c = 0; c < 4; c++) {
    for (let t = 0; t < 3; t++) {
      arr[idx++] = `${colors[c]}-${traits[t]}`;
    }
  }
  arr[52] = "wild";
  arr[53] = "wild_draw_4";
  return arr;
})();

export const ACTION_NAMES = (() => {
  const arr = new Array(61);
  const colors = ["r", "g", "b", "y"];
  let idx = 0;
  for (let c = 0; c < 4; c++) {
    for (let n = 0; n < 10; n++) {
      arr[idx++] = `${colors[c]}-${n}`;
    }
  }
  const traits = ["skip", "reverse", "draw_2"];
  for (let c = 0; c < 4; c++) {
    for (let t = 0; t < 3; t++) {
      arr[idx++] = `${colors[c]}-${traits[t]}`;
    }
  }
  for (let c = 0; c < 4; c++) {
    arr[idx++] = `${colors[c]}-wild`;
  }
  for (let c = 0; c < 4; c++) {
    arr[idx++] = `${colors[c]}-wild_draw_4`;
  }
  arr[60] = "draw";
  return arr;
})();

export const CARD_NAME_TO_ID = (() => {
  const map = {};
  for (let i = 0; i < CARD_NAMES.length; i++) {
    map[CARD_NAMES[i]] = i;
  }
  return map;
})();

export const ACTION_NAME_TO_ID = (() => {
  const map = {};
  for (let i = 0; i < ACTION_NAMES.length; i++) {
    map[ACTION_NAMES[i]] = i;
  }
  return map;
})();

export function cardColor(cardId) {
  if (cardId < 40) return Math.floor(cardId / 10);
  if (cardId < 52) return Math.floor((cardId - 40) / 3);
  return COLOR.NONE;
}

export function cardTrait(cardId) {
  if (cardId < 40) return cardId % 10;
  if (cardId < 52) return 10 + ((cardId - 40) % 3);
  if (cardId === 52) return TRAIT.WILD;
  if (cardId === 53) return TRAIT.WILD_DRAW_4;
  return TRAIT.NONE;
}

export function actionDeclaredColor(actionId) {
  if (actionId < 40) return Math.floor(actionId / 10);
  if (actionId < 52) return Math.floor((actionId - 40) / 3);
  if (actionId < 56) return actionId - 52;
  if (actionId < 60) return actionId - 56;
  return COLOR.NONE;
}

export function actionTrait(actionId) {
  if (actionId < 40) return actionId % 10;
  if (actionId < 52) return 10 + ((actionId - 40) % 3);
  if (actionId < 56) return TRAIT.WILD;
  if (actionId < 60) return TRAIT.WILD_DRAW_4;
  return TRAIT.NONE;
}

export function actionRequiredCard(actionId) {
  if (actionId < 52) return actionId;
  if (actionId < 56) return 52;
  if (actionId < 60) return 53;
  return -1;
}

export function normalizeTargetToCard(targetActionId) {
  if (targetActionId < 52) return targetActionId;
  if (targetActionId < 56) return 52;
  if (targetActionId < 60) return 53;
  return 0;
}

export function getFull108DeckCounts() {
  const arr = new Uint8Array(54);
  for (let c = 0; c < 4; c++) {
    arr[c * 10 + 0] = 1;
    for (let n = 1; n < 10; n++) {
      arr[c * 10 + n] = 2;
    }
    arr[40 + c * 3 + 0] = 2; // skip
    arr[40 + c * 3 + 1] = 2; // reverse
    arr[40 + c * 3 + 2] = 2; // draw_2
  }
  arr[52] = 4; // wild
  arr[53] = 4; // wild_draw_4
  return arr;
}

const FULL_108_DECK_COUNTS = getFull108DeckCounts();

/**
 * Exact Provenance and History Ring Buffer Tracker matching C++ UnoEngine 1-to-1.
 */
export class AtaraxosFeatureTracker {
  constructor() {
    this.publicDiscard = new Uint8Array(54);
    this.lastReshufflePool = new Uint8Array(54);
    this.drawsSinceReshuffle = [0, 0];
    this.knownDrawnFromRecycled = [new Uint8Array(54), new Uint8Array(54)];
    this.historyEvents = []; // max 48 items
  }

  reset(startCardId = 0) {
    this.publicDiscard.fill(0);
    const startCardNorm = normalizeTargetToCard(startCardId);
    this.publicDiscard[startCardNorm] = 1;

    this.lastReshufflePool.fill(0);
    this.drawsSinceReshuffle = [0, 0];
    this.knownDrawnFromRecycled[0].fill(0);
    this.knownDrawnFromRecycled[1].fill(0);
    this.historyEvents = [];
  }

  clone() {
    const copy = new AtaraxosFeatureTracker();
    copy.publicDiscard.set(this.publicDiscard);
    copy.lastReshufflePool.set(this.lastReshufflePool);
    copy.drawsSinceReshuffle = [this.drawsSinceReshuffle[0], this.drawsSinceReshuffle[1]];
    copy.knownDrawnFromRecycled[0].set(this.knownDrawnFromRecycled[0]);
    copy.knownDrawnFromRecycled[1].set(this.knownDrawnFromRecycled[1]);
    copy.historyEvents = this.historyEvents.map((ev) => ({ ...ev }));
    return copy;
  }

  recordPlay(actingPlayer, actionId) {
    const reqCard = actionRequiredCard(actionId);
    if (reqCard >= 0 && reqCard < 54) {
      this.publicDiscard[reqCard]++;
    }
  }

  recordDraws(player, drawnCardIds = [], count = 1) {
    const n = drawnCardIds.length > 0 ? drawnCardIds.length : count;
    this.drawsSinceReshuffle[player] += n;
    for (const c of drawnCardIds) {
      if (c >= 0 && c < 54) {
        this.knownDrawnFromRecycled[player][c]++;
      }
    }
  }

  recordReshuffle(activeTopActionId, recycledCardsOrCounts = null) {
    this.lastReshufflePool.fill(0);
    if (recycledCardsOrCounts && Array.isArray(recycledCardsOrCounts)) {
      if (recycledCardsOrCounts.length === 54) {
        for (let i = 0; i < 54; i++) {
          this.lastReshufflePool[i] = recycledCardsOrCounts[i];
        }
      } else {
        for (const c of recycledCardsOrCounts) {
          const norm = typeof c === "number" ? actionRequiredCard(c) : c;
          if (norm >= 0 && norm < 54) {
            this.lastReshufflePool[norm]++;
          }
        }
      }
    } else {
      // If recycled cards/counts not directly passed, recycled pool equals all public discards except active top
      for (let i = 0; i < 54; i++) {
        this.lastReshufflePool[i] = this.publicDiscard[i];
      }
      const topNorm = normalizeTargetToCard(activeTopActionId);
      if (this.lastReshufflePool[topNorm] > 0) {
        this.lastReshufflePool[topNorm]--;
      }
    }

    this.drawsSinceReshuffle[0] = 0;
    this.drawsSinceReshuffle[1] = 0;
    this.knownDrawnFromRecycled[0].fill(0);
    this.knownDrawnFromRecycled[1].fill(0);

    // Discard pile now contains ONLY top card
    this.publicDiscard.fill(0);
    const topCardNorm = normalizeTargetToCard(activeTopActionId);
    this.publicDiscard[topCardNorm] = 1;
  }

  pushHistoryEvent({
    actorId,
    actionId,
    activeColorBefore,
    stackBefore,
    cardsDrawn,
  }) {
    const ev = {
      actorId: Number(actorId),
      actionId: Number(actionId),
      activeColorBefore: Number(activeColorBefore),
      stackBefore: Math.min(16, Number(stackBefore)),
      cardsDrawn: Math.min(16, Number(cardsDrawn)),
    };

    if (this.historyEvents.length < 48) {
      this.historyEvents.push(ev);
    } else {
      this.historyEvents.shift();
      this.historyEvents.push(ev);
    }
  }

  /**
   * Encode Card Features [54, 7] float32 from `player`'s view.
   */
  encodeCardFeatures(player, handCardIds, targetActionId, activeColor) {
    const out = new Float32Array(54 * 7);
    const handCounts = new Uint8Array(54);
    for (const c of handCardIds) {
      if (c >= 0 && c < 54) {
        handCounts[c]++;
      }
    }

    const topNorm = normalizeTargetToCard(targetActionId);
    const topTrait = actionTrait(targetActionId);

    for (let c = 0; c < 54; c++) {
      const offset = c * 7;
      // 0: my_hand[c]
      out[offset + 0] = handCounts[c];
      // 1: public_discard[c]
      out[offset + 1] = this.publicDiscard[c];
      // 2: unseen_total[c] = DECK_MAX[c] - my_hand[c] - public_discard[c]
      const unseen = FULL_108_DECK_COUNTS[c] - handCounts[c] - this.publicDiscard[c];
      out[offset + 2] = Math.max(0, unseen);
      // 3: recycled_remaining_for_me[c] = max(0, last_reshuffle_pool[c] - known_drawn_from_recycled[player][c])
      const recRem = this.lastReshufflePool[c] - this.knownDrawnFromRecycled[player][c];
      out[offset + 3] = Math.max(0, recRem);
      // 4: is_active_color
      const cCol = cardColor(c);
      out[offset + 4] = cCol === activeColor && cCol !== COLOR.NONE ? 1.0 : 0.0;
      // 5: is_top_card
      out[offset + 5] = c === topNorm ? 1.0 : 0.0;
      // 6: is_matching_rank
      const cTr = cardTrait(c);
      out[offset + 6] = cTr === topTrait && cTr !== TRAIT.NONE ? 1.0 : 0.0;
    }

    return out;
  }

  /**
   * Encode Global Features [10] float32.
   */
  encodeGlobalFeatures({
    player,
    hand0Size,
    hand1Size,
    deckSize,
    stack,
    consecutivePasses,
    turnCount,
    targetActionId,
    availableDrawCount,
  }) {
    const out = new Float32Array(10);
    const myHandSize = player === 0 ? hand0Size : hand1Size;
    const oppHandSize = player === 0 ? hand1Size : hand0Size;
    const opp = 1 - player;
    const targetTr = actionTrait(targetActionId);

    out[0] = myHandSize / 20.0;
    out[1] = oppHandSize / 20.0;
    out[2] = deckSize / 108.0;
    out[3] = stack / 16.0;
    out[4] = stack > 0 && targetTr === TRAIT.DRAW_2 ? 1.0 : 0.0;
    out[5] = stack > 0 && targetTr === TRAIT.WILD_DRAW_4 ? 1.0 : 0.0;
    out[6] = turnCount / 200.0;
    out[7] = consecutivePasses / 2.0;
    out[8] = this.drawsSinceReshuffle[opp] / 20.0;
    out[9] = availableDrawCount > 0 ? 1.0 : 0.0;

    return out;
  }

  /**
   * Encode History Tokens [48, 5] as BigInt64Array (or Int32Array) with leading padding.
   */
  encodeHistoryTokens(player, asBigInt = true) {
    const numEvents = this.historyEvents.length;
    const padCount = numEvents < 48 ? 48 - numEvents : 0;
    const totalTokens = 48 * 5;

    if (asBigInt) {
      const out = new BigInt64Array(totalTokens);
      // Leading padding tokens: [2, 61, 4, 0, 0]
      for (let i = 0; i < padCount; i++) {
        const off = i * 5;
        out[off + 0] = 2n;
        out[off + 1] = 61n;
        out[off + 2] = 4n;
        out[off + 3] = 0n;
        out[off + 4] = 0n;
      }
      // Events oldest to newest
      for (let i = 0; i < numEvents; i++) {
        const ev = this.historyEvents[i];
        const off = (padCount + i) * 5;
        out[off + 0] = ev.actorId === player ? 0n : 1n;
        out[off + 1] = BigInt(ev.actionId);
        out[off + 2] = BigInt(ev.activeColorBefore);
        out[off + 3] = BigInt(ev.stackBefore);
        out[off + 4] = BigInt(ev.cardsDrawn);
      }
      return out;
    } else {
      const out = new Int32Array(totalTokens);
      for (let i = 0; i < padCount; i++) {
        const off = i * 5;
        out[off + 0] = 2;
        out[off + 1] = 61;
        out[off + 2] = 4;
        out[off + 3] = 0;
        out[off + 4] = 0;
      }
      for (let i = 0; i < numEvents; i++) {
        const ev = this.historyEvents[i];
        const off = (padCount + i) * 5;
        out[off + 0] = ev.actorId === player ? 0 : 1;
        out[off + 1] = ev.actionId;
        out[off + 2] = ev.activeColorBefore;
        out[off + 3] = ev.stackBefore;
        out[off + 4] = ev.cardsDrawn;
      }
      return out;
    }
  }

  /**
   * Encode Legal Action Mask [61] float32.
   */
  encodeLegalMask({
    handCardIds,
    targetActionId,
    activeColor,
    stack,
    availableDrawCount,
  }) {
    const mask = new Float32Array(61);
    const targetTr = actionTrait(targetActionId);

    if (stack > 0) {
      if (targetTr === TRAIT.DRAW_2) {
        for (const c of handCardIds) {
          if (c >= 40 && c < 52 && cardTrait(c) === TRAIT.DRAW_2) {
            mask[c] = 1.0;
          }
        }
      } else if (targetTr === TRAIT.WILD_DRAW_4) {
        let hasWd4 = false;
        for (const c of handCardIds) {
          if (c === 53) {
            hasWd4 = true;
            break;
          }
        }
        if (hasWd4) {
          mask[56] = 1.0;
          mask[57] = 1.0;
          mask[58] = 1.0;
          mask[59] = 1.0;
        }
      }
      mask[60] = 1.0; // Draw action is always legal to absorb stack
      return mask;
    }

    // Normal turn (stack == 0)
    let hasPlayable = false;
    for (const c of handCardIds) {
      if (c === 52) {
        mask[52] = 1.0;
        mask[53] = 1.0;
        mask[54] = 1.0;
        mask[55] = 1.0;
        hasPlayable = true;
      } else if (c === 53) {
        mask[56] = 1.0;
        mask[57] = 1.0;
        mask[58] = 1.0;
        mask[59] = 1.0;
        hasPlayable = true;
      } else {
        const col = cardColor(c);
        const tr = cardTrait(c);
        if (col === activeColor || tr === targetTr) {
          mask[c] = 1.0;
          hasPlayable = true;
        }
      }
    }

    if (availableDrawCount > 0 || !hasPlayable) {
      mask[60] = 1.0;
    }

    return mask;
  }
}
