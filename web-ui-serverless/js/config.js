// UNO Configuration & Rule Definitions
const LOCAL_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

// True only when the page URL explicitly forces an API target (?api=...).
// An injected <meta name="uno-api"> or same-origin detection is a *fallback*
// hint, not an override, so a WebGPU-capable browser still prefers local GPU
// inference over a possibly-CPU-only server.
function resolveExplicitApiOverride() {
  if (typeof window === "undefined" || !window.location) {
    return false;
  }
  try {
    const params = new URLSearchParams(window.location.search);
    return params.has("api") && Boolean(params.get("api").trim());
  } catch (_) {
    return false;
  }
}

function resolveDefaultApiUrl() {
  if (typeof window === "undefined" || !window.location) {
    return null;
  }

  const { hostname, search } = window.location;

  // Allow overriding via query param (?api=https://...)
  try {
    const params = new URLSearchParams(search);
    if (params.has("api")) {
      const value = params.get("api").trim();
      if (value === "same-origin") {
        return window.location.origin;
      }
      if (value) {
        return value.replace(/\/+$/, "");
      }
    }
  } catch (_) {}

  // Deployment-specific override, e.g. the Hugging Face Docker Space bundle
  // injects <meta name="uno-api" content="same-origin">.
  const meta = document.querySelector('meta[name="uno-api"]');
  if (meta) {
    const value = (meta.getAttribute("content") || "").trim();
    if (value === "same-origin") {
      return window.location.origin;
    }
    if (value) {
      return value.replace(/\/+$/, "");
    }
  }

  // Local development: scripts/serve_model.py serves the UI and the API from
  // the same origin, regardless of which port was chosen.
  if (LOCAL_HOSTNAMES.has(hostname)) {
    return window.location.origin;
  }

  // On static cloud deployments (Hugging Face Static Spaces, GitHub Pages),
  // there is no backend daemon by default; the neural model runs 100% in-browser.
  return null;
}

export const CONFIG = {
  // Endpoints
  DEFAULT_API_URL: resolveDefaultApiUrl(),
  // A user-typed ?api= wins over the in-browser GPU engine; deployment hints do not.
  EXPLICIT_API_OVERRIDE: resolveExplicitApiOverride(),

  // Timing constants (ms)
  BOT_TURN_DELAY: 80, // Snappy rhythm between bot turns (speculative engine eliminates stalls)
  UNO_REACTION_WINDOW: 1200, // Fair human reaction window before bot challenges UNO
  NEXT_GAME_TIMEOUT: 7, // Seconds before next round auto-starts
  INFERENCE_TIMEOUT_MS: 60000, // Abort a stalled neural request instead of freezing the game
  CARD_FLY_MS: 260, // Human card flight duration (crisp & responsive)
  BOT_CARD_FLY_MS: 260, // Bot card flight duration

  // UNO Deck Constants
  COLORS: ["RED", "GREEN", "BLUE", "YELLOW"],
  RL_COLORS: ["r", "g", "b", "y"],
  START_CARDS: 7,
  MAX_CARDS: 100,
  MAX_TURNS: 200, // Safety ceiling to prevent infinite loops

  COLOR_WEB_TO_RL: {
    RED: "r",
    GREEN: "g",
    BLUE: "b",
    YELLOW: "y",
  },

  COLOR_RL_TO_WEB: {
    r: "RED",
    g: "GREEN",
    b: "BLUE",
    y: "YELLOW",
  },

  TYPE_WEB_TO_RL: {
    ZERO: "0",
    ONE: "1",
    TWO: "2",
    THREE: "3",
    FOUR: "4",
    FIVE: "5",
    SIX: "6",
    SEVEN: "7",
    EIGHT: "8",
    NINE: "9",
    BLOCK: "skip",
    REVERSE: "reverse",
    PLUS_TWO: "draw_2",
    COLOR_CHANGE: "wild",
    PLUS_FOUR: "wild_draw_4",
  },

  TYPE_RL_TO_WEB: {
    0: "ZERO",
    1: "ONE",
    2: "TWO",
    3: "THREE",
    4: "FOUR",
    5: "FIVE",
    6: "SIX",
    7: "SEVEN",
    8: "EIGHT",
    9: "NINE",
    skip: "BLOCK",
    reverse: "REVERSE",
    draw_2: "PLUS_TWO",
    wild: "COLOR_CHANGE",
    wild_draw_4: "PLUS_FOUR",
  },

  COLOR_NAMES: {
    r: "red",
    g: "green",
    b: "blue",
    y: "yellow",
  },
};
