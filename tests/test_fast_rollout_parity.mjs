// Test parity of FastUnoEnv against tests/cpp_game_traces.json
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { FastUnoEnv } from "../web/js/fast-rollout.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const tracesPath = path.join(__dirname, "cpp_game_traces.json");
const games = JSON.parse(fs.readFileSync(tracesPath, "utf-8"));

console.log(`\n=== [FAST ROLLOUT PARITY TEST] Validating ${games.length} full games against C++ traces ===`);

let totalSteps = 0;

for (const game of games) {
  const env = new FastUnoEnv(game.seed);
  env.hands[0] = game.initial_hand_0.slice();
  env.hands[1] = game.initial_hand_1.slice();
  env.target = game.initial_target;
  env.activeColor = game.initial_active_color;
  env.discardPile = [game.initial_target];
  env.tracker.reset(game.initial_target);

  for (let sIdx = 0; sIdx < game.steps.length; sIdx++) {
    const step = game.steps[sIdx];
    totalSteps++;

    // Populate deck count if trace specifies it
    if (env.deck.length === 0 && step.deck_count > 0) {
      env.deck = new Array(step.deck_count).fill(0);
    }

    // Assert pre-step state matches C++ ground truth
    if (env.currentPlayer !== step.current_player) {
      throw new Error(`Game ${game.game_idx} Step ${step.step_num}: currentPlayer ${env.currentPlayer} != ${step.current_player}`);
    }
    if (env.target !== step.target) {
      throw new Error(`Game ${game.game_idx} Step ${step.step_num}: target ${env.target} != ${step.target}`);
    }
    if (env.activeColor !== step.active_color) {
      throw new Error(`Game ${game.game_idx} Step ${step.step_num}: activeColor ${env.activeColor} != ${step.active_color}`);
    }
    if (env.stack !== step.stack) {
      throw new Error(`Game ${game.game_idx} Step ${step.step_num}: stack ${env.stack} != ${step.stack}`);
    }
    if (env.turnCount !== step.turn_count) {
      throw new Error(`Game ${game.game_idx} Step ${step.step_num}: turnCount ${env.turnCount} != ${step.turn_count}`);
    }

    // Assert legal mask matches
    const mask = env.getLegalMask(step.current_player);
    for (let a = 0; a < 61; a++) {
      if ((mask[a] > 0.5 ? 1 : 0) !== step.expected.legal_mask[a]) {
        throw new Error(`Game ${game.game_idx} Step ${step.step_num}: legalMask at ${a} mismatch! JS=${mask[a]} vs CPP=${step.expected.legal_mask[a]}`);
      }
    }

    // Step environment
    // For draws in C++ trace, if drawn_card_ids are recorded, ensure deck has them or set them
    if (step.action_id === 60 && step.cards_drawn_count > 0 && step.drawn_card_ids.length > 0) {
      // Ensure the top cards of deck match trace's drawn cards
      // In FastUnoEnv.drawCards, it pops from back, so reverse drawn_card_ids
      env.deck = step.drawn_card_ids.slice().reverse();
    }

    env.step(step.action_id);

    // If step specifies game over
    if (sIdx === game.steps.length - 1 && game.winner !== -1) {
      if (env.winner !== game.winner) {
        throw new Error(`Game ${game.game_idx} final winner ${env.winner} != ${game.winner}`);
      }
    }
  }
}

console.log(`>>> SUCCESS: Verified ${totalSteps} steps across ${games.length} games with 100% parity against C++!`);
