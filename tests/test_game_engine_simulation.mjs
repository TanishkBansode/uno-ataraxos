import { UnoGameEngine } from "../web/js/game-engine.js";

console.log("\n=== [SIMULATION TEST] Running 50 Full Games in UnoGameEngine ===");

let humanWins = 0;
let botWins = 0;
let draws = 0;
let totalTurns = 0;

for (let g = 0; g < 50; g++) {
  const engine = new UnoGameEngine();
  engine.startNewGame();

  let safety = 0;
  while (engine.isStarted && safety < 500) {
    safety++;
    const curPlayer = engine.currentMove;
    const { items } = engine.getLegalActionsForPlayer(curPlayer);

    if (items.length === 0) {
      engine.drawCard(curPlayer);
    } else {
      // Pick an item
      const chosen = items[Math.floor(Math.random() * items.length)];
      const res = engine.playCard(curPlayer, chosen.cardId, chosen.pickColor || null);
      if (!res.success) {
        throw new Error(`Game ${g} turn ${safety}: move rejected! reason=${res.reason}`);
      }
    }
  }

  totalTurns += engine.turnCount;
  if (engine.winner === "human") humanWins++;
  else if (engine.winner === "bot") botWins++;
  else draws++;

  // Verify feature tracker integrity at the end of every game
  const botFeatures = engine.getAtaraxosFeatures("bot");
  if (botFeatures.cardFeatures.length !== 54 * 7) {
    throw new Error(`Corrupted card_features length: ${botFeatures.cardFeatures.length}`);
  }
  if (botFeatures.globalFeatures.length !== 10) {
    throw new Error(`Corrupted global_features length: ${botFeatures.globalFeatures.length}`);
  }
  if (botFeatures.historyTokens.length !== 48 * 5) {
    throw new Error(`Corrupted history_tokens length: ${botFeatures.historyTokens.length}`);
  }
  if (botFeatures.legalMask.length !== 61) {
    throw new Error(`Corrupted legal_mask length: ${botFeatures.legalMask.length}`);
  }
}

console.log(`✓ 50 games completed safely without errors!`);
console.log(`- Human wins: ${humanWins}`);
console.log(`- Bot wins:   ${botWins}`);
console.log(`- Draws:      ${draws}`);
console.log(`- Avg turns:  ${(totalTurns / 50).toFixed(1)}`);
console.log(`>>> ENGINE SIMULATION PASSED! <<<\n`);
