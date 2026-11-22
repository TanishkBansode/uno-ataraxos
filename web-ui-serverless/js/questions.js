// Port of unolaya/questions.py: Typed choice question builder for Laya
import { CONFIG } from "./config.js";

export const QUESTION_ID = "play";

export const INSTRUCTIONS =
  "Choose the best play under official 1v1 UNO rules. " +
  "Counter active +2 with draw_2 or active +4 with wild_draw_4, else draw penalty. " +
  "When stack is 0: skip and reverse give an extra turn; draw_2 and wild_draw_4 build penalty stacks; " +
  "match active color or number.";

export function describeAction(action) {
  if (action === "draw") {
    return "draw from deck (takes active penalty stack or draws 1 card and ends turn)";
  }

  if (!action.includes("-")) {
    return `play ${action}`;
  }

  const [col, trait] = action.split("-");
  const fullColor = CONFIG.COLOR_NAMES[col] || col;

  if (/^\d+$/.test(trait)) {
    return `play ${action} (${fullColor} number ${trait})`;
  }
  if (trait === "skip") {
    return `play ${action} (${fullColor} skip, opponent loses turn, you play again)`;
  }
  if (trait === "reverse") {
    return `play ${action} (${fullColor} reverse, acts as skip in 1v1, you play again)`;
  }
  if (trait === "draw_2") {
    return `play ${action} (${fullColor} draw_2, adds +2 to penalty stack)`;
  }
  if (trait === "wild") {
    return `play ${action} (wild card, declares active color ${fullColor})`;
  }
  if (trait === "wild_draw_4") {
    return `play ${action} (wild +4 stack, declares active color ${fullColor})`;
  }

  return `play ${action} (${fullColor} ${trait})`;
}

export function buildQuestions(legalActions) {
  const criteria = {};
  for (const action of legalActions) {
    criteria[action] = describeAction(action);
  }

  return {
    [QUESTION_ID]: {
      type: "choice",
      instructions: INSTRUCTIONS,
      criteria: criteria,
    },
  };
}
