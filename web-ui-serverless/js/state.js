// Port of unolaya/state.py: 9-line strategic text summary with 108-card counting
import { CONFIG } from "./config.js";

export function getFull108DeckCounts() {
  const deck = {};
  for (const c of CONFIG.RL_COLORS) {
    deck[`${c}-0`] = 1;
    for (let num = 1; num <= 9; num++) {
      deck[`${c}-${num}`] = 2;
    }
    deck[`${c}-skip`] = 2;
    deck[`${c}-reverse`] = 2;
    deck[`${c}-draw_2`] = 2;
  }
  deck["wild"] = 4;
  deck["wild_draw_4"] = 4;
  return deck;
}

export function normalizeCardKey(cardStr) {
  if (cardStr.endsWith("wild_draw_4") || cardStr === "wild_draw_4") {
    return "wild_draw_4";
  }
  if (cardStr.endsWith("wild") || cardStr === "wild") {
    return "wild";
  }
  return cardStr;
}

export function computeUnseenCards(hand, playedCards) {
  const unseen = getFull108DeckCounts();
  for (const card of hand) {
    const key = normalizeCardKey(card);
    if (unseen[key] && unseen[key] > 0) {
      unseen[key] -= 1;
    }
  }
  for (const card of playedCards) {
    const key = normalizeCardKey(card);
    if (unseen[key] && unseen[key] > 0) {
      unseen[key] -= 1;
    }
  }
  return unseen;
}

export function formatUnoState({
  hand,
  target = "r-0",
  stack = 0,
  playedCards = [],
  oppCardCount = 7,
  oppDrawRecency = {},
}) {
  const activeColor = target.includes("-") ? target.split("-")[0] : target;

  // Hand color counts
  const myColorCounts = {};
  for (const c of CONFIG.RL_COLORS) {
    myColorCounts[c] = hand.filter((card) => card.startsWith(`${c}-`)).length;
  }
  const colorCountStr = `r:${myColorCounts["r"]} g:${myColorCounts["g"]} b:${myColorCounts["b"]} y:${myColorCounts["y"]}`;
  const sortedHand = [...hand].sort().join(", ");

  // Opponent cards & UNO danger
  const unoDanger = oppCardCount <= 2 ? "yes" : "no";

  // Unseen cards computation
  const unseen = computeUnseenCards(hand, playedCards);
  let totalUnseen = 0;
  for (const count of Object.values(unseen)) {
    totalUnseen += count;
  }
  const drawPileEst = Math.max(0, totalUnseen - oppCardCount);

  // Unseen wilds
  const uWild = unseen["wild"] || 0;
  const uWd4 = unseen["wild_draw_4"] || 0;

  // Unseen draw_2
  const rD2 = unseen["r-draw_2"] || 0;
  const gD2 = unseen["g-draw_2"] || 0;
  const bD2 = unseen["b-draw_2"] || 0;
  const yD2 = unseen["y-draw_2"] || 0;
  const uD2Total = rD2 + gD2 + bD2 + yD2;

  // Unseen by color & skip+rev
  const uByColor = {};
  const srByColor = {};
  for (const c of CONFIG.RL_COLORS) {
    let colSum = 0;
    for (const [k, count] of Object.entries(unseen)) {
      if (k.startsWith(`${c}-`)) {
        colSum += count;
      }
    }
    uByColor[c] = colSum;
    srByColor[c] = (unseen[`${c}-skip`] || 0) + (unseen[`${c}-reverse`] || 0);
  }

  // Number matches for numbers in hand
  const numberSet = new Set();
  for (const card of hand) {
    if (card.includes("-")) {
      const trait = card.split("-")[1];
      if (/^\d+$/.test(trait)) {
        numberSet.add(trait);
      }
    }
  }
  const numbersInHand = Array.from(numberSet).sort(
    (a, b) => parseInt(a, 10) - parseInt(b, 10),
  );

  let numBridgesStr = "none";
  if (numbersInHand.length > 0) {
    const matches = numbersInHand.map((num) => {
      let sum = 0;
      for (const c of CONFIG.RL_COLORS) {
        sum += unseen[`${c}-${num}`] || 0;
      }
      return `${num}:${sum}`;
    });
    numBridgesStr = matches.join(" ");
  }

  const recencyStr = CONFIG.RL_COLORS.map((c) => {
    const value = oppDrawRecency ? oppDrawRecency[c] : undefined;
    return `${c}:${value === undefined || value === null || value < 0 ? "-" : String(value)}`;
  }).join(" ");

  const lines = [
    `Target card: ${target} (active color: ${activeColor}) | Active penalty stack: +${stack}`,
    `My hand (${hand.length}): ${sortedHand} | My color counts: ${colorCountStr}`,
    `Opponent cards: ${oppCardCount} (UNO danger: ${unoDanger})`,
    `Opponent draw recency: ${recencyStr}`,
    `Unseen total: ${totalUnseen} cards (draw pile: ${drawPileEst}, opponent: ${oppCardCount})`,
    `Unseen wilds: wild:${uWild}/4, wild_draw_4:${uWd4}/4`,
    `Unseen draw_2: ${uD2Total}/8 (r:${rD2}/2 g:${gD2}/2 b:${bD2}/2 y:${yD2}/2)`,
    `Unseen by color: r:${uByColor["r"]}/25 g:${uByColor["g"]}/25 b:${uByColor["b"]}/25 y:${uByColor["y"]}/25 | Unseen skip+rev: r:${srByColor["r"]}/4 g:${srByColor["g"]}/4 b:${srByColor["b"]}/4 y:${srByColor["y"]}/4`,
    `Unseen matches for my numbers: ${numBridgesStr}`,
  ];

  return lines.join("\n");
}
