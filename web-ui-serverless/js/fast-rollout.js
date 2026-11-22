// Fast In-Browser Typed-Array Vectorized UNO Rollout Engine matching C++ VecUnoEnv 1-to-1
import {
  COLOR,
  TRAIT,
  CARD_NAMES,
  ACTION_NAMES,
  cardColor,
  cardTrait,
  actionDeclaredColor,
  actionTrait,
  actionRequiredCard,
  normalizeTargetToCard,
  getFull108DeckCounts,
  AtaraxosFeatureTracker,
} from "./ataraxos-features.js";

/**
 * 32-bit PRNG (Mulberry32) matching fast deterministic random streams.
 */
export function mulberry32(a) {
  let s = a >>> 0;
  return function () {
    let t = (s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates array shuffle using supplied PRNG.
 */
export function shuffleArray(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/**
 * Single Authoritative Fast UNO Game Simulation State.
 */
export class FastUnoEnv {
  constructor(seed = 42) {
    this.rng = mulberry32(seed);
    this.hands = [[], []]; // Card IDs (0..53)
    this.deck = []; // Card IDs (0..53)
    this.discardPile = []; // Action IDs (0..59)
    this.target = 0; // Action ID (0..59)
    this.activeColor = 0; // 0..3 (r, g, b, y)
    this.stack = 0; // Penalty stack
    this.currentPlayer = 0; // 0 or 1
    this.isOver = false;
    this.winner = -1; // -1: draw, 0: player 0, 1: player 1
    this.consecutivePasses = 0;
    this.turnCount = 0;
    this.tracker = new AtaraxosFeatureTracker();
  }

  clone() {
    const env = new FastUnoEnv();
    env.rng = this.rng;
    env.hands = [this.hands[0].slice(), this.hands[1].slice()];
    env.deck = this.deck.slice();
    env.discardPile = this.discardPile.slice();
    env.target = this.target;
    env.activeColor = this.activeColor;
    env.stack = this.stack;
    env.currentPlayer = this.currentPlayer;
    env.isOver = this.isOver;
    env.winner = this.winner;
    env.consecutivePasses = this.consecutivePasses;
    env.turnCount = this.turnCount;
    env.tracker = this.tracker.clone();
    return env;
  }

  getAvailableDrawCount() {
    let count = this.deck.length;
    if (this.discardPile.length > 1) {
      count += this.discardPile.length - 1;
    }
    return count;
  }

  _recycleDiscardPile() {
    if (this.discardPile.length <= 1) return;
    const top = this.discardPile[this.discardPile.length - 1];
    const recycled = [];
    for (let i = 0; i < this.discardPile.length - 1; i++) {
      recycled.push(actionRequiredCard(this.discardPile[i]));
    }
    this.tracker.recordReshuffle(top, recycled);
    shuffleArray(recycled, this.rng);
    this.deck = recycled;
    this.discardPile = [top];
  }

  drawCards(count, player) {
    const drawn = [];
    for (let i = 0; i < count; i++) {
      if (this.deck.length === 0) {
        this._recycleDiscardPile();
      }
      if (this.deck.length === 0) break;
      const c = this.deck.pop();
      this.hands[player].push(c);
      drawn.push(c);
    }
    this.tracker.recordDraws(player, drawn);
    return drawn;
  }

  _removeCardFromHand(player, actionId) {
    const reqCard = actionRequiredCard(actionId);
    const hand = this.hands[player];
    const idx = hand.indexOf(reqCard);
    if (idx !== -1) {
      hand.splice(idx, 1);
    } else {
      throw new Error(`Action ${actionId} (card ${reqCard}) not in player ${player} hand`);
    }
  }

  _endByCardCount() {
    this.isOver = true;
    const c0 = this.hands[0].length;
    const c1 = this.hands[1].length;
    if (c0 < c1) this.winner = 0;
    else if (c1 < c0) this.winner = 1;
    else this.winner = -1;
  }

  getLegalMask(player = this.currentPlayer) {
    if (this.isOver) return new Float32Array(61);
    return this.tracker.encodeLegalMask({
      handCardIds: this.hands[player],
      targetActionId: this.target,
      activeColor: this.activeColor,
      stack: this.stack,
      availableDrawCount: this.getAvailableDrawCount(),
    });
  }

  getCardFeatures(player = this.currentPlayer) {
    return this.tracker.encodeCardFeatures(
      player,
      this.hands[player],
      this.target,
      this.activeColor,
    );
  }

  getGlobalFeatures(player = this.currentPlayer) {
    return this.tracker.encodeGlobalFeatures({
      player,
      hand0Size: this.hands[0].length,
      hand1Size: this.hands[1].length,
      deckSize: this.deck.length,
      stack: this.stack,
      consecutivePasses: this.consecutivePasses,
      turnCount: this.turnCount,
      targetActionId: this.target,
      availableDrawCount: this.getAvailableDrawCount(),
    });
  }

  getHistoryTokens(player = this.currentPlayer, asBigInt = false) {
    return this.tracker.encodeHistoryTokens(player, asBigInt);
  }

  step(actionId) {
    if (this.isOver) return 0.0;

    const actingPlayer = this.currentPlayer;
    const opponent = 1 - actingPlayer;
    const colorBefore = this.activeColor;
    const stackBefore = this.stack;

    this.turnCount += 1;
    const cardsAvailableBefore = this.getAvailableDrawCount();
    let actualDrawn = 0;

    if (actionId === 60) {
      // "draw"
      if (this.stack > 0) {
        const drawn = this.drawCards(this.stack, actingPlayer);
        actualDrawn = Math.min(16, drawn.length);
        this.stack = 0;
      } else {
        const drawn = this.drawCards(1, actingPlayer);
        actualDrawn = Math.min(16, drawn.length);
      }

      if (cardsAvailableBefore === 0) {
        this.consecutivePasses += 1;
        if (this.consecutivePasses >= 2) {
          this._endByCardCount();
        }
      } else {
        this.consecutivePasses = 0;
      }

      if (this.turnCount >= 200 && !this.isOver) {
        this._endByCardCount();
      }

      if (!this.isOver) {
        this.currentPlayer = opponent;
      }
    } else {
      // Play card
      this.consecutivePasses = 0;
      this._removeCardFromHand(actingPlayer, actionId);
      this.discardPile.push(actionId);
      this.tracker.recordPlay(actingPlayer, actionId);
      this.target = actionId;
      this.activeColor = actionDeclaredColor(actionId);

      if (this.hands[actingPlayer].length === 0) {
        this.isOver = true;
        this.winner = actingPlayer;
      } else if (this.turnCount >= 200 && !this.isOver) {
        this._endByCardCount();
      } else {
        const tr = actionTrait(actionId);
        if (tr === TRAIT.DRAW_2) {
          this.stack += 2;
          this.currentPlayer = opponent;
        } else if (tr === TRAIT.WILD_DRAW_4) {
          this.stack += 4;
          this.currentPlayer = opponent;
        } else if (tr === TRAIT.SKIP || tr === TRAIT.REVERSE) {
          this.currentPlayer = actingPlayer; // In 1v1 UNO, skip/reverse gives immediate extra turn
        } else {
          this.currentPlayer = opponent;
        }
      }
    }

    // Push history event in ring buffer
    this.tracker.pushHistoryEvent({
      actorId: actingPlayer,
      actionId,
      activeColorBefore: colorBefore,
      stackBefore: Math.min(16, stackBefore),
      cardsDrawn: actualDrawn,
    });

    if (this.isOver) {
      if (this.winner === actingPlayer) return 1.0;
      if (this.winner === opponent) return -1.0;
      return 0.0;
    }
    return 0.0;
  }
}

/**
 * Fast Vectorized Rollout Pool for Ataraxos Test-Time Search in Browser.
 */
export class FastVecUnoEnv {
  constructor(numEnvs = 1) {
    this.numEnvs = numEnvs;
    this.envs = [];
    for (let i = 0; i < numEnvs; i++) {
      this.envs.push(new FastUnoEnv());
    }
  }

  resize(newNumEnvs) {
    while (this.envs.length < newNumEnvs) {
      this.envs.push(new FastUnoEnv());
    }
    this.envs.length = newNumEnvs;
    this.numEnvs = newNumEnvs;
  }

  loadWorldsAndStepRoot(rootEnv, sampledOppHands, legalActions, seed = 42) {
    const K = sampledOppHands.length;
    const numA = legalActions.length;
    const totalSlots = numA * K;
    if (this.numEnvs !== totalSlots) {
      this.resize(totalSlots);
    }

    const rootP = rootEnv.currentPlayer;
    const oppP = 1 - rootP;
    const fullCounts = getFull108DeckCounts();

    const myCounts = new Uint8Array(54);
    for (const c of rootEnv.hands[rootP]) {
      myCounts[c]++;
    }

    // Step 1: Pre-generate K sampled worlds (oppHand and drawPile)
    const worlds = new Array(K);
    for (let k = 0; k < K; k++) {
      const oppHandCounts = sampledOppHands[k];
      const oppHand = [];
      const drawPile = [];

      for (let c = 0; c < 54; c++) {
        const cnt = oppHandCounts[c];
        for (let i = 0; i < cnt; i++) {
          oppHand.push(c);
        }
        let rem = fullCounts[c] - myCounts[c] - rootEnv.tracker.publicDiscard[c] - cnt;
        if (rem > 0) {
          for (let i = 0; i < rem; i++) {
            drawPile.push(c);
          }
        }
      }

      const rngK = mulberry32(seed + k * 10007 + 1);
      shuffleArray(drawPile, rngK);
      worlds[k] = { oppHand, drawPile, rngK };
    }

    // Step 2: Initialize slots and step root actions
    const rewards = new Float32Array(totalSlots);
    const dones = new Uint8Array(totalSlots);
    const actingPlayer = new Int32Array(totalSlots);

    for (let aIdx = 0; aIdx < numA; aIdx++) {
      const act = legalActions[aIdx];
      for (let k = 0; k < K; k++) {
        const slot = aIdx * K + k;
        const env = this.envs[slot];
        env.rng = mulberry32(seed + slot * 10007 + 1);
        env.hands[rootP] = rootEnv.hands[rootP].slice();
        env.hands[oppP] = worlds[k].oppHand.slice();
        env.deck = worlds[k].drawPile.slice();
        env.discardPile = rootEnv.discardPile.slice();
        env.target = rootEnv.target;
        env.activeColor = rootEnv.activeColor;
        env.stack = rootEnv.stack;
        env.currentPlayer = rootP;
        env.isOver = false;
        env.winner = -1;
        env.consecutivePasses = rootEnv.consecutivePasses;
        env.turnCount = rootEnv.turnCount;
        env.tracker = rootEnv.tracker.clone();

        actingPlayer[slot] = rootP;
        const r = env.step(act);
        rewards[slot] = r;
        dones[slot] = env.isOver ? 1 : 0;
      }
    }

    return { rewards, dones, actingPlayer };
  }

  stepIndices(activeIndices, actions) {
    const rewards = new Float32Array(activeIndices.length);
    const dones = new Uint8Array(activeIndices.length);
    const actingPlayer = new Int32Array(activeIndices.length);

    for (let i = 0; i < activeIndices.length; i++) {
      const slot = activeIndices[i];
      const env = this.envs[slot];
      actingPlayer[i] = env.currentPlayer;
      const r = env.step(actions[i]);
      rewards[i] = r;
      dones[i] = env.isOver ? 1 : 0;
    }
    return { rewards, dones, actingPlayer };
  }

  getTensorsForIndices(activeIndices, rootPlayer) {
    const B = activeIndices.length;
    const cf = new Float32Array(B * 54 * 7);
    const gf = new Float32Array(B * 10);
    const ht = new BigInt64Array(B * 48 * 5);
    const mask = new Float32Array(B * 61);
    const currentP = new Int32Array(B);
    const oppHandCounts = new Float32Array(B * 54);

    for (let i = 0; i < B; i++) {
      const slot = activeIndices[i];
      const env = this.envs[slot];
      const p = env.currentPlayer;
      currentP[i] = p;

      const cf_i = env.getCardFeatures(p);
      cf.set(cf_i, i * 54 * 7);

      const gf_i = env.getGlobalFeatures(p);
      gf.set(gf_i, i * 10);

      const ht_i = env.tracker.encodeHistoryTokens(p, true);
      ht.set(ht_i, i * 48 * 5);

      const m_i = env.getLegalMask(p);
      mask.set(m_i, i * 61);

      const opp = 1 - p;
      for (const c of env.hands[opp]) {
        oppHandCounts[i * 54 + c]++;
      }
    }

    return { cf, gf, ht, mask, currentP, oppHandCounts, B };
  }
}
