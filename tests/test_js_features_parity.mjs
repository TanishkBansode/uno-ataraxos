// Mandatory Parity Test: Assert JS AtaraxosFeatureTracker reproduces C++ UnoEngine ground truth across 20 full games.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  AtaraxosFeatureTracker,
  CARD_NAMES,
  ACTION_NAMES,
} from "../web-ui-serverless/js/ataraxos-features.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const tracesPath = path.join(__dirname, "cpp_game_traces.json");
if (!fs.existsSync(tracesPath)) {
  console.error(`Trace file ${tracesPath} does not exist. Run scripts/export_cpp_game_traces.py first.`);
  process.exit(1);
}

const games = JSON.parse(fs.readFileSync(tracesPath, "utf-8"));
console.log(`\n=== [JS vs C++ PARITY TEST] Validating ${games.length} full games ===`);

let totalStepsChecked = 0;
let maxDiffCardFeatures = 0.0;
let maxDiffGlobalFeatures = 0.0;
let maxDiffHistoryTokens = 0;
let maxDiffLegalMask = 0.0;

for (const game of games) {
  const tracker = new AtaraxosFeatureTracker();
  tracker.reset(game.initial_target);

  for (const step of game.steps) {
    const currP = step.current_player;
    const currHand = currP === 0 ? step.hand_0 : step.hand_1;

    // 1. Compute JS Tensors
    const cf = tracker.encodeCardFeatures(
      currP,
      currHand,
      step.target,
      step.active_color,
    );

    const gf = tracker.encodeGlobalFeatures({
      player: currP,
      hand0Size: step.hand_0.length,
      hand1Size: step.hand_1.length,
      deckSize: step.deck_count,
      stack: step.stack,
      consecutivePasses: step.consecutive_passes,
      turnCount: step.turn_count,
      targetActionId: step.target,
      availableDrawCount: step.available_draw_count,
    });

    const ht = tracker.encodeHistoryTokens(currP, false); // Int32Array [48 * 5]

    const mask = tracker.encodeLegalMask({
      handCardIds: currHand,
      targetActionId: step.target,
      activeColor: step.active_color,
      stack: step.stack,
      availableDrawCount: step.available_draw_count,
    });

    // 2. Validate against expected C++ tensors
    const expCf = step.expected.card_features.flat();
    for (let i = 0; i < cf.length; i++) {
      const diff = Math.abs(cf[i] - expCf[i]);
      if (diff > maxDiffCardFeatures) maxDiffCardFeatures = diff;
      if (diff > 1e-5) {
        throw new Error(
          `Game ${game.game_idx} Step ${step.step_num}: card_features mismatch at idx ${i} (card ${Math.floor(i / 7)} channel ${i % 7}): JS=${cf[i]} vs CPP=${expCf[i]} (diff=${diff})`,
        );
      }
    }

    const expGf = step.expected.global_features;
    for (let i = 0; i < gf.length; i++) {
      const diff = Math.abs(gf[i] - expGf[i]);
      if (diff > maxDiffGlobalFeatures) maxDiffGlobalFeatures = diff;
      if (diff > 1e-5) {
        throw new Error(
          `Game ${game.game_idx} Step ${step.step_num}: global_features mismatch at idx ${i}: JS=${gf[i]} vs CPP=${expGf[i]} (diff=${diff})`,
        );
      }
    }

    const expHt = step.expected.history_tokens.flat();
    for (let i = 0; i < ht.length; i++) {
      const diff = Math.abs(ht[i] - expHt[i]);
      if (diff > maxDiffHistoryTokens) maxDiffHistoryTokens = diff;
      if (diff !== 0) {
        throw new Error(
          `Game ${game.game_idx} Step ${step.step_num}: history_tokens mismatch at token ${Math.floor(i / 5)} field ${i % 5}: JS=${ht[i]} vs CPP=${expHt[i]}`,
        );
      }
    }

    const expMask = step.expected.legal_mask;
    for (let i = 0; i < mask.length; i++) {
      const diff = Math.abs(mask[i] - expMask[i]);
      if (diff > maxDiffLegalMask) maxDiffLegalMask = diff;
      if (diff > 1e-5) {
        throw new Error(
          `Game ${game.game_idx} Step ${step.step_num}: legal_mask mismatch at action ${i} (${ACTION_NAMES[i]}): JS=${mask[i]} vs CPP=${expMask[i]}`,
        );
      }
    }

    totalStepsChecked++;

    // 3. Advance tracker state matching C++ UnoEngine transition
    if (step.action_id === 60) {
      if (step.reshuffled) {
        const drawnBefore = Math.min(step.deck_count, step.cards_drawn_count);
        const drawnAfter = step.cards_drawn_count - drawnBefore;
        tracker.recordReshuffle(step.target, step.reshuffle_pool);
        const cardsAfter = step.drawn_card_ids.slice(drawnBefore);
        tracker.recordDraws(currP, cardsAfter, drawnAfter);
      } else {
        tracker.recordDraws(
          currP,
          step.drawn_card_ids,
          step.cards_drawn_count,
        );
      }
    } else {
      tracker.recordPlay(currP, step.action_id);
    }

    tracker.pushHistoryEvent({
      actorId: currP,
      actionId: step.action_id,
      activeColorBefore: step.active_color,
      stackBefore: step.stack,
      cardsDrawn: step.cards_drawn_count,
    });
  }
}

console.log(`\n>>> SUCCESS: All ${totalStepsChecked} steps across ${games.length} full games matched C++ ground truth!`);
console.log(`- Max card_features diff:    ${maxDiffCardFeatures.toExponential(4)} (< 1e-5)`);
console.log(`- Max global_features diff:  ${maxDiffGlobalFeatures.toExponential(4)} (< 1e-5)`);
console.log(`- Max history_tokens diff:   ${maxDiffHistoryTokens} (exact integer match)`);
console.log(`- Max legal_mask diff:       ${maxDiffLegalMask.toExponential(4)} (exact integer match)`);
