// Serverless UI Coordinator: Connects DOM, In-Browser UnoGameEngine & Strict AtaraxosModelClient
import { CONFIG } from "./config.js";
import { UnoGameEngine } from "./game-engine.js";
import { AtaraxosModelClient, FatalNeuralModelException } from "./model-client.js";
import {
  initAnalytics,
  trackEvent,
  recordMatchResult,
  recordAiDecision,
  getStatsSummary,
  initStatsDialog,
} from "./stats.js";
import {
  playSound,
  setOverlay,
  preloadAssets,
  prefersReducedMotion,
} from "./utils.js";

// ---------------------------------------------------------------------------
// Instances & state
// ---------------------------------------------------------------------------

const engine = new UnoGameEngine();
const modelClient = new AtaraxosModelClient();
if (typeof window !== "undefined") {
  window.engine = engine;
  window.modelClient = modelClient;
}

initAnalytics();

const humanPlayerId = "human";
const botPlayerId = "bot";

let humanUsername = readStorage("uno_username", "Player");
const humanAvatarUrl = "resources/defaultAvatar.png";
let isBotThinking = false;
let isHumanActing = false;
let pendingHumanCardId = null;
let pendingHumanCardImg = null;
let deskCardEl = null;
let nextGameInterval = null;
let winnerShowTimeout = null;
let resizeFrame = null;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const loginContainer = document.getElementById("login-container");
const loginForm = document.getElementById("login-form");
const gameContainer = document.getElementById("game-container");
const usernameInput = document.getElementById("username");
const connectBtn = document.getElementById("connect");
const aiRetryBtn = document.getElementById("ai-retry-btn");
const loginStatsBtn = document.getElementById("login-stats-btn");
const hudStatsBtn = document.getElementById("hud-stats-btn");
const winnerStatsBtn = document.getElementById("winner-stats-btn");
const cardsContainer = document.getElementById("cards");
const cardDesk = document.getElementById("cards-desk");
const cardDeckBtn = document.getElementById("carddeck");
const colorSelect = document.getElementById("color-select");
const playersContainer = document.getElementById("players");
const stackingContainer = document.getElementById("stacking-container");
const stackingCount = document.getElementById("stacking-count");
const arrowImg = document.getElementById("arrow");
const winnerContainer = document.getElementById("winner-container");
const winnerWrapper = document.getElementById("winner-wrapper");
const winnerUsername = document.getElementById("winner-username");
const winnerSummary = document.getElementById("winner-summary");
const winnerAvatar = document.getElementById("winner-avatar");
const winnerTimeout = document.getElementById("winner-timeout");
const winnerReturnBtn = document.getElementById("winner-return");
const botCardsContainer = document.getElementById("bot-cards");

// Clean Consumer HUD elements
const winProbVal = document.getElementById("win-prob-val");
const winProbBarFill = document.getElementById("win-prob-bar-fill");
const xrayToggleBtn = document.getElementById("xray-toggle-btn");
const xrayPanel = document.getElementById("belief-xray-panel");
const xrayCloseBtn = document.getElementById("xray-close-btn");
const xrayFooterNote = document.getElementById("xray-footer-note");

// Stats modal & analytics click tracking
const statsDialog = initStatsDialog();

for (const btn of [loginStatsBtn, hudStatsBtn, winnerStatsBtn]) {
  if (btn) {
    btn.addEventListener("click", () => {
      if (btn === winnerStatsBtn && nextGameInterval) {
        clearNextGameInterval();
        if (winnerTimeout) winnerTimeout.textContent = "Auto-next paused";
      }
      statsDialog.open();
    });
  }
}

for (const trackedEl of document.querySelectorAll("[data-track-event]")) {
  trackedEl.addEventListener("click", () => {
    const evt = trackedEl.dataset.trackEvent;
    if (evt) trackEvent(evt, trackedEl.title || trackedEl.textContent.trim());
  });
}

function setXrayOpen(open) {
  if (!xrayPanel || !xrayToggleBtn) return;
  xrayPanel.classList.toggle("hidden", !open);
  xrayToggleBtn.classList.toggle("active", open);
  xrayToggleBtn.setAttribute("aria-expanded", open ? "true" : "false");

  if (open) {
    trackEvent("xray-open", "Opened Belief X-Ray");
    if (modelClient.isInBrowserLoaded && !modelClient.inBrowserEngine.beliefLoaded) {
      if (xrayFooterNote) {
        xrayFooterNote.textContent = "Loading neural belief model...";
      }
      modelClient
        .initInBrowserBelief()
        .then(() => {
          if (xrayFooterNote) {
            xrayFooterNote.textContent = "Neural Belief Net active — updates after AI's turn";
          }
        })
        .catch((err) => {
          console.warn("[Ataraxos] Belief init error:", err);
          if (xrayFooterNote) {
            xrayFooterNote.textContent = "Updates after AI's turn";
          }
        });
    }
  }
}

if (xrayToggleBtn && xrayPanel) {
  xrayToggleBtn.addEventListener("click", () => {
    const isHidden = xrayPanel.classList.contains("hidden");
    setXrayOpen(isHidden);
  });
}

if (xrayCloseBtn && xrayPanel) {
  xrayCloseBtn.addEventListener("click", () => {
    setXrayOpen(false);
  });
}

function updateWinProbabilityDisplay(prob) {
  const p = Math.max(0, Math.min(1, Number(prob) || 0.5));
  const pct = (p * 100).toFixed(1);
  if (winProbVal) winProbVal.textContent = `${pct}%`;
  if (winProbBarFill) {
    winProbBarFill.style.width = `${pct}%`;
    if (p >= 0.55) {
      winProbBarFill.style.background = "#00e676";
    } else if (p <= 0.45) {
      winProbBarFill.style.background = "#ff3b30";
    } else {
      winProbBarFill.style.background = "#ffd60a";
    }
  }
}

function updateBeliefXrayDisplay(xray) {
  if (!xray) return;
  const setEl = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  const p_r = xray.p_red ?? xray.red;
  const p_g = xray.p_green ?? xray.green;
  const p_b = xray.p_blue ?? xray.blue;
  const p_y = xray.p_yellow ?? xray.yellow;
  const p_p2 = xray.p_plus_2 ?? xray.plus_two;
  const p_p4 = xray.p_plus_4 ?? xray.plus_four;

  if (p_r !== undefined) setEl("xray-red", `${(p_r * 100).toFixed(0)}%`);
  if (p_g !== undefined) setEl("xray-green", `${(p_g * 100).toFixed(0)}%`);
  if (p_b !== undefined) setEl("xray-blue", `${(p_b * 100).toFixed(0)}%`);
  if (p_y !== undefined) setEl("xray-yellow", `${(p_y * 100).toFixed(0)}%`);
  if (p_p2 !== undefined) setEl("xray-p2", `${(p_p2 * 100).toFixed(0)}%`);
  if (p_p4 !== undefined) setEl("xray-p4", `${(p_p4 * 100).toFixed(0)}%`);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readStorage(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch (_) {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) {
    // Storage can be unavailable in private mode / when full.
  }
}

function motionDuration(ms) {
  return prefersReducedMotion() ? 1 : ms;
}

function cardImageSrc(card, useColorGif = false) {
  if (!card) return "resources/cards/UNO_CARD.png";
  const isWild = card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR";
  if (useColorGif && isWild && card.color && card.color !== "ANY") {
    return `resources/cards/gifs/${card.color}_${card.type}.gif`;
  }
  return `resources/cards/${card.color}_${card.type}.png`;
}

function formatCardAriaLabel(card) {
  if (!card) return "UNO card";
  const colorMap = {
    RED: "Red",
    GREEN: "Green",
    BLUE: "Blue",
    YELLOW: "Yellow",
    ANY: "Wild",
  };
  const typeMap = {
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
    BLOCK: "Skip",
    REVERSE: "Reverse",
    PLUS_TWO: "Draw Two",
    COLOR_CHANGE: "Wild",
    PLUS_FOUR: "Wild Draw Four",
  };
  const col = colorMap[card.color] || "";
  const typ = typeMap[card.type] || card.type;
  return card.color === "ANY" ? typ : `${col} ${typ}`;
}

function playCardTypeSound(type) {
  switch (type) {
    case "REVERSE":
      playSound("resources/sounds/reverse.mp3");
      break;
    case "BLOCK":
      playSound("resources/sounds/block.mp3");
      break;
    case "PLUS_TWO":
      playSound("resources/sounds/plus_two.mp3");
      break;
    case "PLUS_FOUR":
      playSound("resources/sounds/plus_four.mp3");
      break;
    default:
      playSound("resources/sounds/card_place.mp3");
  }
}

function canHumanAct() {
  return (
    engine.isStarted &&
    engine.currentMove === humanPlayerId &&
    !isBotThinking &&
    !isHumanActing &&
    !engine.isChoosingColor &&
    !pendingHumanCardId
  );
}

// ---------------------------------------------------------------------------
// Login & AI Initialization
// ---------------------------------------------------------------------------

usernameInput.value = humanUsername;
modelClient.setEndpointUrl(CONFIG.DEFAULT_API_URL);

preloadAssets();

const aiStatusText = document.getElementById("ai-status-text");
const aiProgressBarContainer = document.getElementById(
  "ai-progress-bar-container",
);
const aiProgressBar = document.getElementById("ai-progress-bar");

let useInBrowser = false;
let inBrowserInitPromise = null;

function startInBrowserInit() {
  if (modelClient.isInBrowserLoaded) {
    if (aiStatusText) {
      const dev = (modelClient.inBrowserDevice || "wasm").toUpperCase();
      aiStatusText.classList.remove("error");
      aiStatusText.textContent = `AI Ready (${dev})`;
    }
    if (aiProgressBarContainer) aiProgressBarContainer.style.display = "none";
    if (aiRetryBtn) aiRetryBtn.classList.add("hidden");
    return Promise.resolve();
  }

  if (inBrowserInitPromise) return inBrowserInitPromise;

  if (aiStatusText) {
    aiStatusText.classList.remove("error");
    aiStatusText.textContent = "Loading AI...";
  }
  if (aiRetryBtn) aiRetryBtn.classList.add("hidden");

  inBrowserInitPromise = modelClient
    .initInBrowser((pct, msg) => {
      if (aiStatusText) {
        aiStatusText.classList.remove("error");
        aiStatusText.textContent = msg || `Loading AI (${pct}%)...`;
      }
      if (aiProgressBarContainer) aiProgressBarContainer.style.display = "block";
      if (aiProgressBar) aiProgressBar.style.width = `${pct}%`;
    })
    .then(() => {
      const dev = (modelClient.inBrowserDevice || "wasm").toUpperCase();
      if (aiStatusText) {
        aiStatusText.classList.remove("error");
        aiStatusText.textContent = `AI Ready (${dev})`;
      }
      if (aiProgressBarContainer) aiProgressBarContainer.style.display = "none";
      if (aiRetryBtn) aiRetryBtn.classList.add("hidden");
    })
    .catch((err) => {
      console.warn("[ataraxos-uno] In-browser init failed:", err);
      inBrowserInitPromise = null;
      if (aiStatusText) {
        aiStatusText.classList.add("error");
        aiStatusText.textContent = `AI failed to load: ${err.message || "Network error"}`;
      }
      if (aiProgressBarContainer) aiProgressBarContainer.style.display = "none";
      if (aiRetryBtn) aiRetryBtn.classList.remove("hidden");
      throw err;
    });

  return inBrowserInitPromise;
}

const backendReady = (async () => {
  const isServerAvailable = await modelClient.checkHealth();

  if (isServerAvailable) {
    useInBrowser = false;
    modelClient.setPreferInBrowser(false);
    modelClient.setMode("auto");
    if (aiStatusText) {
      aiStatusText.classList.remove("error");
      aiStatusText.textContent = "AI Ready (Server)";
    }
    if (aiProgressBarContainer) aiProgressBarContainer.style.display = "none";
  } else {
    useInBrowser = true;
    modelClient.setPreferInBrowser(true);
    modelClient.setMode("auto");
    startInBrowserInit().catch(() => {});
  }
})();

if (aiRetryBtn) {
  aiRetryBtn.addEventListener("click", () => {
    startInBrowserInit().catch(() => {});
  });
}

async function handleLoginSubmit() {
  humanUsername = (usernameInput.value || "").trim().slice(0, 24) || "Player";
  writeStorage("uno_username", humanUsername);

  connectBtn.textContent = "Loading AI...";
  connectBtn.disabled = true;

  try {
    await backendReady;

    if (useInBrowser && !modelClient.isInBrowserLoaded) {
      await startInBrowserInit();
    }
  } catch (initErr) {
    console.error("[ataraxos-uno] In-browser init failed on play click:", initErr);
    connectBtn.textContent = "Play!";
    connectBtn.disabled = false;
    return;
  }

  connectBtn.textContent = "Play!";
  connectBtn.disabled = false;

  trackEvent("play-clicked", "Entered Game Lobby");

  loginContainer.classList.add("hidden");
  gameContainer.classList.remove("hidden");

  setupLobby();
}

if (loginForm) {
  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleLoginSubmit();
  });
} else {
  connectBtn.addEventListener("click", handleLoginSubmit);
}

// ---------------------------------------------------------------------------
// Lobby / player badges
// ---------------------------------------------------------------------------

function setupLobby() {
  playersContainer.innerHTML = "";
  renderPlayerBadge(humanPlayerId, humanUsername, humanAvatarUrl, 0);
  renderPlayerBadge(
    botPlayerId,
    "ATARAXOS AI",
    "resources/defaultAvatar.png",
    0,
  );

  cardsContainer.innerHTML = "";
  botCardsContainer.innerHTML = "";
  deskCardEl = null;
  cardDesk.innerHTML = `
    <img draggable="false" src="resources/cards/UNO_CARD.png" class="card-desk" id="START_CARD" alt="Start game deck" style="cursor: pointer;" />
    <button type="button" id="start-game-btn" class="start-deal-btn">Deal Cards</button>
  `;

  winnerContainer.classList.add("hidden");
  winnerWrapper.classList.remove("ScaleUp");
  if (arrowImg) arrowImg.style.visibility = "hidden";
  updateStackDisplay(0);
  setInteractionForTurn(null);

  const startBtn = document.getElementById("start-game-btn");
  if (startBtn) startBtn.focus();
}

cardDesk.addEventListener("click", (e) => {
  const trigger =
    e.target &&
    (e.target.id === "START_CARD" || e.target.id === "start-game-btn");
  if (trigger && !engine.isStarted) {
    engine.startNewGame();
  }
});

function renderPlayerBadge(pid, name, avatar, count) {
  let div = document.getElementById(pid);
  if (!div) {
    div = document.createElement("div");
    div.className = "player";
    div.id = pid;

    const uLabel = document.createElement("label");
    uLabel.className = "username";
    uLabel.id = `username_${pid}`;
    uLabel.innerText = name;

    const aImg = document.createElement("img");
    aImg.className = "avatar";
    aImg.id = `avatar_${pid}`;
    aImg.src = avatar;
    aImg.alt = "";
    aImg.width = 64;
    aImg.height = 64;
    aImg.draggable = false;

    const oImg = document.createElement("img");
    oImg.className = "overlay";
    oImg.id = `overlay_${pid}`;
    oImg.alt = "";
    oImg.width = 64;
    oImg.height = 64;
    oImg.draggable = false;

    const cLabel = document.createElement("label");
    cLabel.className = "count";
    cLabel.id = `count_${pid}`;
    cLabel.innerText = count;

    div.appendChild(uLabel);
    div.appendChild(aImg);
    div.appendChild(oImg);
    div.appendChild(cLabel);
    playersContainer.appendChild(div);
  } else {
    const cLabel = document.getElementById(`count_${pid}`);
    if (cLabel) cLabel.innerText = count;
  }
}

// ---------------------------------------------------------------------------
// Card rendering (diff-based, so existing cards are never rebuilt / re-animated)
// ---------------------------------------------------------------------------

function layoutFan(container, selector, fallbackWidth, maxFanWidth) {
  if (!container) return;
  const cards = Array.from(container.querySelectorAll(selector));
  if (!cards.length) return;

  const firstWidth = cards[0].getBoundingClientRect().width || fallbackWidth;
  const fallbackAvailable =
    selector === ".bot-card"
      ? window.innerWidth * 0.6
      : window.innerWidth * 0.75;
  const available = container.clientWidth || fallbackAvailable;
  const fanWidth = Math.min(available, maxFanWidth);
  const step =
    cards.length > 1
      ? Math.min(
          firstWidth,
          Math.max(2, (fanWidth - firstWidth) / (cards.length - 1)),
        )
      : firstWidth;
  const margin = step - firstWidth;

  cards.forEach((el, i) => {
    el.style.marginLeft = i === 0 ? "0px" : `${margin}px`;
  });
}

function captureCardRects(container, selector) {
  const rects = new Map();
  if (!container) return rects;
  for (const el of container.querySelectorAll(selector)) {
    rects.set(el, el.getBoundingClientRect());
  }
  return rects;
}

function playFlip(container, selector, beforeRects, duration = 220) {
  if (!beforeRects || beforeRects.size === 0) return;
  const ms = motionDuration(duration);
  for (const el of container.querySelectorAll(selector)) {
    const before = beforeRects.get(el);
    if (!before || (el.matches && el.matches(":hover"))) continue;

    const now = el.getBoundingClientRect();
    const dx = before.left - now.left;
    const dy = before.top - now.top;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) continue;
    if (ms <= 1 || typeof el.animate !== "function") continue;

    const anim = el.animate(
      [
        { transform: `translate3d(${dx}px, ${dy}px, 0)` },
        { transform: "translate3d(0, 0, 0)" },
      ],
      {
        duration: ms,
        easing: "cubic-bezier(0.2, 0.8, 0.25, 1)",
      },
    );
    if (anim.finished && typeof anim.finished.catch === "function") {
      anim.finished.catch(() => {});
    }
  }
}

function renderHumanCards(cards) {
  const hand = cards || engine.hands.human || {};
  const entries = Object.entries(hand);
  const beforeRects = captureCardRects(cardsContainer, ".card");
  const existing = new Map();
  for (const el of cardsContainer.querySelectorAll(".card")) {
    existing.set(el.dataset.cardId, el);
  }

  for (const [cid, el] of existing) {
    if (!Object.prototype.hasOwnProperty.call(hand, cid)) {
      el.remove();
    }
  }

  entries.forEach(([cid, card], index) => {
    let img = existing.get(cid);
    if (!img) {
      img = document.createElement("img");
      img.className = "card";
      img.dataset.cardId = cid;
      img.draggable = false;
      img.alt = formatCardAriaLabel(card);
      img.setAttribute("role", "button");
      img.src = cardImageSrc(card);
      img.addEventListener(
        "animationend",
        () => img.classList.remove("dealt"),
        { once: true },
      );
      img.classList.add("dealt");
      existing.set(cid, img);
    } else {
      const src = cardImageSrc(card);
      if (img.getAttribute("src") !== src) img.setAttribute("src", src);
      img.alt = formatCardAriaLabel(card);
    }

    const canPlay =
      engine.isStarted &&
      engine.currentMove === humanPlayerId &&
      !isBotThinking &&
      !isHumanActing &&
      engine.canPlayCard(card, humanPlayerId);
    img.classList.toggle("playable", canPlay);
    img.classList.toggle("unplayable", !canPlay);
    img.tabIndex = canPlay ? 0 : -1;
    img.setAttribute("aria-disabled", canPlay ? "false" : "true");

    const ref = cardsContainer.children[index] || null;
    if (ref !== img) cardsContainer.insertBefore(img, ref);
  });

  layoutFan(cardsContainer, ".card", 90, 680);
  playFlip(cardsContainer, ".card", beforeRects);
}

function renderBotCards(count) {
  if (!botCardsContainer) return;
  const beforeRects = captureCardRects(botCardsContainer, ".bot-card");
  const cards = Array.from(botCardsContainer.querySelectorAll(".bot-card"));

  while (cards.length > count) {
    cards.pop().remove();
  }
  while (cards.length < count) {
    const img = document.createElement("img");
    img.className = "bot-card dealt";
    img.src = "resources/cards/UNO_CARD.png";
    img.draggable = false;
    img.alt = "";
    img.addEventListener("animationend", () => img.classList.remove("dealt"), {
      once: true,
    });
    botCardsContainer.appendChild(img);
    cards.push(img);
  }

  layoutFan(botCardsContainer, ".bot-card", 76, 460);
  playFlip(botCardsContainer, ".bot-card", beforeRects);
}

function refreshPlayableCards() {
  const hand = engine.hands.human || {};
  for (const el of cardsContainer.querySelectorAll(".card")) {
    const card = hand[el.dataset.cardId];
    const canPlay =
      !!card &&
      engine.isStarted &&
      engine.currentMove === humanPlayerId &&
      !isBotThinking &&
      !isHumanActing &&
      engine.canPlayCard(card, humanPlayerId);
    el.classList.toggle("playable", canPlay);
    el.classList.toggle("unplayable", !canPlay);
    el.tabIndex = canPlay ? 0 : -1;
    el.setAttribute("aria-disabled", canPlay ? "false" : "true");
  }
}

function updatePlayerCount(pid, count) {
  const el = document.getElementById(`count_${pid}`);
  if (el) el.innerText = count;
}

function clearPlayerUnoOverlay(pid) {
  const overlay = document.getElementById(`overlay_${pid}`);
  if (!overlay) return;
  overlay.classList.remove("popup");
  overlay.removeAttribute("src");
}

function clearUnoOverlays() {
  for (const pid of [humanPlayerId, botPlayerId]) {
    clearPlayerUnoOverlay(pid);
  }
}

function updateActivePlayerHighlight(activePlayerId) {
  document
    .querySelectorAll(".glow")
    .forEach((el) => el.classList.remove("glow"));
  if (!activePlayerId) return;
  const activeLabel = document.getElementById(`username_${activePlayerId}`);
  if (activeLabel) activeLabel.classList.add("glow");
}

function setInteractionForTurn(playerId = engine.currentMove) {
  const canAct =
    engine.isStarted &&
    playerId === humanPlayerId &&
    !isBotThinking &&
    !isHumanActing &&
    !engine.isChoosingColor &&
    !pendingHumanCardId;

  cardDeckBtn.classList.toggle("disabled", !canAct);
  cardDeckBtn.setAttribute("aria-disabled", canAct ? "false" : "true");
  cardsContainer.classList.toggle("disabled", !canAct);
  updateActivePlayerHighlight(playerId);
  refreshPlayableCards();
}

function updateStackDisplay(stack) {
  const value = Number(stack) || 0;
  const drawText = value > 0 ? `Draw ${value} cards` : "Draw card";
  cardDeckBtn.title = drawText;
  cardDeckBtn.setAttribute("aria-label", drawText);

  if (value > 0) {
    stackingCount.innerText = `+${value}`;
    if (!stackingContainer.classList.contains("PopIn")) {
      stackingContainer.classList.remove("PopOut");
      stackingContainer.classList.add("PopIn");
    }
  } else if (stackingContainer.classList.contains("PopIn")) {
    stackingContainer.classList.remove("PopIn");
    stackingContainer.classList.add("PopOut");
  }
}

function renderDeskCard(
  card,
  { colorChange = false, reset = false, silent = false, animate = false } = {},
) {
  if (reset) {
    cardDesk.innerHTML = "";
    deskCardEl = null;
  }
  if (!card) return null;

  const src = cardImageSrc(card, colorChange);

  let img = deskCardEl;
  if (!img || !img.isConnected) {
    img = document.createElement("img");
    img.className = "card-desk";
    img.draggable = false;
    img.alt = formatCardAriaLabel(card);
    deskCardEl = img;
    cardDesk.appendChild(img);
  } else {
    img.alt = formatCardAriaLabel(card);
  }

  if (img.getAttribute("src") !== src) {
    img.setAttribute("src", src);
  }

  for (const oldCard of Array.from(cardDesk.querySelectorAll(".card-desk"))) {
    if (oldCard !== img) oldCard.remove();
  }

  const animationClass = colorChange ? "color-reveal" : "settle";
  img.classList.remove("settle", "color-reveal");
  if (animate) {
    void img.offsetWidth;
    img.classList.add(animationClass);
    img.addEventListener(
      "animationend",
      () => {
        img.classList.remove(animationClass);
      },
      { once: true },
    );
  }

  if (!silent) {
    if (colorChange) {
      playSound("resources/sounds/change_color.mp3");
    } else {
      playCardTypeSound(card.type);
    }
  }
  return img;
}

// ---------------------------------------------------------------------------
// Card flight animations (transform/opacity only, handed off to the real DOM)
// ---------------------------------------------------------------------------

function deskTargetRect() {
  const rect = cardDesk.getBoundingClientRect();
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height: rect.height,
  };
}

function flyCard({ src, fromRect, toRect, duration }) {
  const fromWidth = fromRect.width || toRect.width;
  const fromHeight = fromRect.height || toRect.height;
  const ghost = document.createElement("img");
  ghost.className = "flying-card";
  ghost.src = src;
  ghost.draggable = false;
  ghost.alt = "";
  ghost.style.left = `${fromRect.left}px`;
  ghost.style.top = `${fromRect.top}px`;
  ghost.style.width = `${fromWidth}px`;
  ghost.style.height = `${fromHeight}px`;
  document.body.appendChild(ghost);

  const dx = toRect.left + toRect.width / 2 - (fromRect.left + fromWidth / 2);
  const dy = toRect.top + toRect.height / 2 - (fromRect.top + fromHeight / 2);
  const sx = toRect.width / fromWidth;
  const sy = toRect.height / fromHeight;
  const ms = motionDuration(duration);

  let anim = null;
  if (ms > 1 && typeof ghost.animate === "function") {
    anim = ghost.animate(
      [
        {
          transform: "translate3d(0, 0, 0) scale(1, 1) rotate(0deg)",
          opacity: 1,
          offset: 0,
        },
        {
          transform: `translate3d(${dx * 0.55}px, ${dy * 0.55 - 18}px, 0) scale(${(1 + sx) / 2}, ${(1 + sy) / 2}) rotate(-4deg)`,
          opacity: 1,
          offset: 0.55,
        },
        {
          transform: `translate3d(${dx}px, ${dy}px, 0) scale(${sx}, ${sy}) rotate(0deg)`,
          opacity: 1,
          offset: 1,
        },
      ],
      {
        duration: ms,
        easing: "cubic-bezier(0.22, 0.85, 0.25, 1)",
        fill: "forwards",
      },
    );
  } else {
    ghost.style.transform = `translate3d(${dx}px, ${dy}px, 0) scale(${sx}, ${sy})`;
  }

  return { ghost, anim, duration: ms };
}

async function waitForFlight(flight) {
  if (!flight) return;
  if (!flight.anim) return;
  await Promise.race([
    flight.anim.finished.catch(() => {}),
    new Promise((resolve) => setTimeout(resolve, flight.duration + 450)),
  ]);
}

function removeGhost(flight) {
  if (flight && flight.ghost) flight.ghost.remove();
}

async function waitForDeskImage() {
  const img = cardDesk.querySelector(".card-desk");
  if (!img || typeof img.decode !== "function") return;
  try {
    await img.decode();
  } catch (_) {
    // A failed decode still leaves the src assigned; never block the turn.
  }
}

async function animateHumanPlayCard(img, srcOverride = null) {
  const fromRect = img.getBoundingClientRect();
  img.style.visibility = "hidden";
  const flight = flyCard({
    src: srcOverride || img.src,
    fromRect,
    toRect: deskTargetRect(),
    duration: CONFIG.CARD_FLY_MS,
  });
  await waitForFlight(flight);
  return flight;
}

async function animateBotPlayCard(card, chosenColor) {
  const botCards = botCardsContainer
    ? botCardsContainer.querySelectorAll(".bot-card")
    : [];
  let sourceEl = null;
  let fromRect;

  if (botCards.length > 0) {
    sourceEl = botCards[botCards.length - 1];
    const rect = sourceEl.getBoundingClientRect();
    fromRect = {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
    };
    sourceEl.style.visibility = "hidden";
  } else {
    const width = 76;
    const height = 118;
    fromRect = {
      left: window.innerWidth / 2 - width / 2,
      top: 20,
      width,
      height,
    };
  }

  const useGif =
    (card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR") &&
    !!chosenColor;
  const src = useGif
    ? `resources/cards/gifs/${chosenColor}_${card.type}.gif`
    : `resources/cards/${card.color}_${card.type}.png`;

  const flight = flyCard({
    src,
    fromRect,
    toRect: deskTargetRect(),
    duration: CONFIG.BOT_CARD_FLY_MS,
  });
  await waitForFlight(flight);
  return { flight, sourceEl };
}

function deckSourceRect(cardWidth, cardHeight) {
  const rect = cardDeckBtn.getBoundingClientRect();
  return {
    left: rect.left + (rect.width - cardWidth) / 2,
    top: rect.top + (rect.height - cardHeight) / 2,
    width: cardWidth,
    height: cardHeight,
  };
}

async function animateBotDrawCard() {
  const firstCard = botCardsContainer
    ? botCardsContainer.querySelector(".bot-card")
    : null;
  const width = firstCard ? firstCard.getBoundingClientRect().width : 76;
  const height = firstCard ? firstCard.getBoundingClientRect().height : 118;
  const containerRect = botCardsContainer
    ? botCardsContainer.getBoundingClientRect()
    : { left: window.innerWidth / 2 - width / 2, top: 20, width, height };

  const flight = flyCard({
    src: "resources/cards/UNO_CARD.png",
    fromRect: deckSourceRect(width, height),
    toRect: {
      left: containerRect.left + Math.max(0, (containerRect.width - width) / 2),
      top: containerRect.top + (containerRect.height - height) / 2,
      width,
      height,
    },
    duration: CONFIG.CARD_FLY_MS,
  });
  await waitForFlight(flight);
  return flight;
}

async function animateHumanDrawCard() {
  const firstCard = cardsContainer.querySelector(".card");
  const width = firstCard ? firstCard.getBoundingClientRect().width : 90;
  const height = firstCard ? firstCard.getBoundingClientRect().height : 140;
  const containerRect = cardsContainer.getBoundingClientRect();

  const flight = flyCard({
    src: "resources/cards/UNO_CARD.png",
    fromRect: deckSourceRect(width, height),
    toRect: {
      left: containerRect.left + Math.max(0, (containerRect.width - width) / 2),
      top: containerRect.top + Math.max(0, containerRect.height - height),
      width,
      height,
    },
    duration: CONFIG.CARD_FLY_MS,
  });
  await waitForFlight(flight);
  return flight;
}

// ---------------------------------------------------------------------------
// Turn / action handlers & Speculative Neural Prefetching
// ---------------------------------------------------------------------------

let botPrefetchPromise = null;

function prefetchBotMoveIfTurn() {
  if (!engine.isStarted || engine.currentMove !== botPlayerId) return;
  const features = engine.getAtaraxosFeatures(botPlayerId);
  botPrefetchPromise = modelClient.predictAction(features);
}

async function playHumanCard(cid, card, img, chosenColor) {
  if (!card || !engine.isStarted || engine.currentMove !== humanPlayerId) {
    if (img) img.style.visibility = "";
    return;
  }

  isHumanActing = true;
  setInteractionForTurn();

  const isWild = card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR";
  const srcOverride =
    isWild && chosenColor
      ? `resources/cards/gifs/${chosenColor}_${card.type}.gif`
      : null;

  let flight = null;
  try {
    let flightPromise = null;
    if (img && img.isConnected) {
      flightPromise = animateHumanPlayCard(img, srcOverride);
    }
    const result = engine.playCard(humanPlayerId, cid, chosenColor);
    if (!result.success) {
      console.warn(
        `[ataraxos-uno] Card play rejected: ${result.reason || "unknown reason"}`,
      );
    }

    // Speculatively fire bot neural inference concurrently with card flight animation
    prefetchBotMoveIfTurn();

    if (flightPromise) {
      flight = await flightPromise;
    }
    await waitForDeskImage();
  } catch (err) {
    console.error("[ataraxos-uno] Failed to place card:", err);
  } finally {
    removeGhost(flight);
    if (img) img.style.visibility = "";
    isHumanActing = false;
    setInteractionForTurn();
  }
}

function activateHumanCard(img) {
  if (!img || !cardsContainer.contains(img)) return;
  if (!canHumanAct()) return;

  const cid = img.dataset.cardId;
  const card = engine.hands.human[cid];
  if (!card || !engine.canPlayCard(card, humanPlayerId)) return;

  if (card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR") {
    pendingHumanCardId = cid;
    pendingHumanCardImg = img;
    isHumanActing = true;
    setInteractionForTurn();
    showColorSelect();
    return;
  }

  playHumanCard(cid, card, img, null);
}

cardsContainer.addEventListener("click", (e) => {
  const img = e.target && e.target.closest ? e.target.closest(".card") : null;
  activateHumanCard(img);
});

cardsContainer.addEventListener("keydown", (e) => {
  const img = e.target && e.target.closest ? e.target.closest(".card") : null;
  if (!img) return;
  if (e.key === "Enter") {
    e.preventDefault();
    activateHumanCard(img);
  } else if (e.key === " ") {
    e.preventDefault();
  }
});

cardsContainer.addEventListener("keyup", (e) => {
  const img = e.target && e.target.closest ? e.target.closest(".card") : null;
  if (!img) return;
  if (e.key === " ") {
    e.preventDefault();
    activateHumanCard(img);
  }
});

async function handleDeckClick() {
  if (!canHumanAct()) return;
  isHumanActing = true;
  setInteractionForTurn();

  let flight = null;
  try {
    const flightPromise = animateHumanDrawCard();
    engine.drawCard(humanPlayerId);
    prefetchBotMoveIfTurn();
    flight = await flightPromise;
  } catch (err) {
    console.error("[ataraxos-uno] Failed to draw card:", err);
  } finally {
    removeGhost(flight);
    isHumanActing = false;
    setInteractionForTurn();
  }
}

cardDeckBtn.addEventListener("click", handleDeckClick);

function showColorSelect() {
  colorSelect.classList.remove("PopOut");
  void colorSelect.offsetWidth;
  colorSelect.classList.add("PopIn");
  playSound("resources/sounds/choose_color.mp3");
  const firstColorBtn = colorSelect.querySelector(".color");
  if (firstColorBtn) firstColorBtn.focus();
}

function hideColorSelect() {
  if (!colorSelect.classList.contains("PopIn")) return;
  colorSelect.classList.remove("PopIn");
  colorSelect.classList.add("PopOut");
}

function cancelColorSelect() {
  if (!pendingHumanCardId) return;
  const prevImg = pendingHumanCardImg;
  pendingHumanCardId = null;
  pendingHumanCardImg = null;
  hideColorSelect();
  isHumanActing = false;
  setInteractionForTurn();
  if (prevImg && prevImg.isConnected) prevImg.focus();
}

colorSelect.addEventListener("click", (e) => {
  const btn =
    e.target && e.target.closest ? e.target.closest(".color") : null;
  if (!btn || !pendingHumanCardId) return;

  const chosenColor = btn.id;
  const cid = pendingHumanCardId;
  const img = pendingHumanCardImg;
  const card = engine.hands.human[cid];

  pendingHumanCardId = null;
  pendingHumanCardImg = null;
  hideColorSelect();
  playSound("resources/sounds/color_press.mp3");

  if (!card) {
    isHumanActing = false;
    setInteractionForTurn();
    return;
  }

  playHumanCard(cid, card, img, chosenColor);
});

// Clicking away or pressing Escape cancels a pending wild colour choice (or closes X-Ray).
document.addEventListener(
  "pointerdown",
  (e) => {
    if (!pendingHumanCardId) return;
    if (colorSelect.contains(e.target)) return;
    cancelColorSelect();
  },
  true,
);

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (pendingHumanCardId) {
      cancelColorSelect();
    } else if (xrayPanel && !xrayPanel.classList.contains("hidden")) {
      setXrayOpen(false);
    }
  }
});

// ---------------------------------------------------------------------------
// Engine events
// ---------------------------------------------------------------------------

engine.on("game_started", (data) => {
  clearNextGameInterval();
  clearWinnerShowTimeout();
  hideColorSelect();
  pendingHumanCardId = null;
  pendingHumanCardImg = null;
  isHumanActing = false;
  isBotThinking = false;
  modelClient.resetGame();
  updateWinProbabilityDisplay(0.5);

  if (arrowImg) arrowImg.style.visibility = "visible";
  winnerContainer.classList.add("hidden");
  winnerWrapper.classList.remove("ScaleUp");
  clearUnoOverlays();

  renderDeskCard(data.currentCard, { reset: true, silent: true });
  playSound("resources/sounds/card_place.mp3");

  renderHumanCards(data.humanCards);
  renderBotCards(data.botCardCount);
  updatePlayerCount(humanPlayerId, Object.keys(data.humanCards).length);
  updatePlayerCount(botPlayerId, data.botCardCount);
  updateStackDisplay(data.stack);
  setInteractionForTurn(data.currentMove);
  trackEvent("game-started", "Started UNO Match");
});

engine.on("card_placed", (data) => {
  const { playerId, card, remainingCount } = data;
  const isWild =
    card && (card.type === "COLOR_CHANGE" || card.type === "PLUS_FOUR");

  if (playerId === humanPlayerId) {
    renderHumanCards(engine.hands.human);
    updatePlayerCount(humanPlayerId, remainingCount);
  } else {
    renderBotCards(remainingCount);
    updatePlayerCount(botPlayerId, remainingCount);
  }
  if (remainingCount !== 1) clearPlayerUnoOverlay(playerId);

  renderDeskCard(card, { silent: !!isWild });
  updateStackDisplay(data.stack);
});

engine.on("turn_changed", (data) => {
  updateStackDisplay(data.stack);
  setInteractionForTurn(data.currentMove);

  if (data.currentMove === botPlayerId) {
    triggerBotTurn();
  }
});

engine.on("color_changed", (data) => {
  renderDeskCard(data.currentCard, {
    colorChange: true,
    silent: true,
    animate: true,
  });
  playSound("resources/sounds/change_color.mp3");
});

engine.on("cards_drawn", (data) => {
  playSound("resources/sounds/card_pickup.mp3");
  updatePlayerCount(data.playerId, data.newCount);
  updateStackDisplay(data.stack);

  if (data.playerId === humanPlayerId) {
    renderHumanCards(engine.hands.human);
  } else {
    renderBotCards(data.newCount);
  }
  if (data.newCount !== 1) clearPlayerUnoOverlay(data.playerId);
});

engine.on("uno_called", (data) => {
  setOverlay(`overlay_${data.playerId}`, "resources/UNO.png");
});

engine.on("game_over", (data) => {
  handleGameOver(data);
});

// ---------------------------------------------------------------------------
// Bot turn: strictly neural, no fallbacks
// ---------------------------------------------------------------------------

async function triggerBotTurn() {
  if (!engine.isStarted || engine.currentMove !== botPlayerId || isBotThinking)
    return;
  isBotThinking = true;
  setInteractionForTurn(botPlayerId);

  try {
    // The while loop supports 1v1 extra turns (Skip/Reverse) without stalling.
    while (engine.isStarted && engine.currentMove === botPlayerId) {
      if (CONFIG.BOT_TURN_DELAY > 0) {
        await new Promise((resolve) =>
          setTimeout(resolve, CONFIG.BOT_TURN_DELAY),
        );
      }
      if (!engine.isStarted || engine.currentMove !== botPlayerId) break;

      const { items } = engine.getLegalActionsForPlayer(botPlayerId);

      // Consume speculative neural prediction if already in flight, else run inference
      let prediction;
      if (botPrefetchPromise) {
        prediction = await botPrefetchPromise;
        botPrefetchPromise = null;
      } else {
        const features = engine.getAtaraxosFeatures(botPlayerId);
        prediction = await modelClient.predictAction(features);
      }

      if (!engine.isStarted || engine.currentMove !== botPlayerId) break;

      if (prediction.win_prob !== undefined) {
        updateWinProbabilityDisplay(prediction.win_prob);
      }
      if (prediction.belief_xray) {
        updateBeliefXrayDisplay(prediction.belief_xray);
      }
      recordAiDecision({
        latencyMs: prediction.latencyMs,
        backend: prediction.engine_backend,
      });

      // If no legal cards, the bot must draw (or take the active penalty stack).
      if (items.length === 0 || prediction.action === "draw") {
        while (isHumanActing) {
          await new Promise((r) => setTimeout(r, 16));
        }
        const flight = await animateBotDrawCard();
        engine.drawCard(botPlayerId);
        removeGhost(flight);
        break;
      }

      const chosenAction = prediction.action;
      const match = items.find((it) => it.actionStr === chosenAction);
      if (!match) {
        throw new FatalNeuralModelException(
          `Neural model selected '${chosenAction}' but no matching legal card was found.`,
        );
      }

      const cardObj = engine.hands.bot[match.cardId];
      if (!cardObj) {
        throw new FatalNeuralModelException(
          `Neural model selected a card that is no longer in the bot hand.`,
        );
      }

      while (isHumanActing) {
        await new Promise((r) => setTimeout(r, 16));
      }

      const chosenColor = match.pickColor || null;
      const { flight, sourceEl } = await animateBotPlayCard(
        cardObj,
        chosenColor,
      );

      try {
        const result = engine.playCard(botPlayerId, match.cardId, chosenColor);
        if (!result.success) {
          throw new FatalNeuralModelException(
            `Neural action rejected by the engine: ${result.reason || "unknown reason"}.`,
          );
        }
        await waitForDeskImage();
      } finally {
        removeGhost(flight);
        if (sourceEl) sourceEl.style.visibility = "";
      }
    }
  } catch (err) {
    showFatalNeuralError(err);
  } finally {
    botPrefetchPromise = null;
    isBotThinking = false;
    setInteractionForTurn(engine.currentMove);
  }
}

// ---------------------------------------------------------------------------
// Fatal error / game over UI
// ---------------------------------------------------------------------------

function showFatalNeuralError(err) {
  engine.isStarted = false;
  isBotThinking = false;
  isHumanActing = false;
  pendingHumanCardId = null;
  pendingHumanCardImg = null;
  hideColorSelect();
  setInteractionForTurn(null);

  if (document.getElementById("fatal-neural-modal")) return;

  const modal = document.createElement("div");
  modal.id = "fatal-neural-modal";
  Object.assign(modal.style, {
    position: "fixed",
    inset: "0",
    backgroundColor: "rgba(0, 0, 0, 0.85)",
    backdropFilter: "blur(10px)",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    zIndex: "99999",
    color: "#ff4d4f",
    fontFamily: "monospace",
    padding: "20px",
    textAlign: "center",
  });

  const title = document.createElement("h1");
  title.textContent = "AI Engine Error";
  Object.assign(title.style, {
    fontSize: "2.2rem",
    color: "#ff4d4f",
    marginBottom: "15px",
  });

  const message = document.createElement("p");
  message.textContent = err && err.message ? err.message : String(err);
  Object.assign(message.style, {
    fontSize: "1.2rem",
    color: "#fff",
    maxWidth: "650px",
    lineHeight: "1.6",
    marginBottom: "25px",
  });

  const policy = document.createElement("p");
  policy.textContent =
    "The game halted because the AI model failed to produce a valid move.";
  Object.assign(policy.style, {
    fontSize: "1rem",
    color: "#faad14",
    marginBottom: "30px",
  });

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Restart Game";
  Object.assign(button.style, {
    padding: "12px 28px",
    fontSize: "1.1rem",
    fontWeight: "bold",
    backgroundColor: "#1890ff",
    color: "#fff",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
  });
  button.addEventListener("click", () => window.location.reload());

  modal.appendChild(title);
  modal.appendChild(message);
  modal.appendChild(policy);
  modal.appendChild(button);
  document.body.appendChild(modal);
}

function clearNextGameInterval() {
  if (nextGameInterval) {
    clearInterval(nextGameInterval);
    nextGameInterval = null;
  }
}

function clearWinnerShowTimeout() {
  if (winnerShowTimeout) {
    clearTimeout(winnerShowTimeout);
    winnerShowTimeout = null;
  }
}

function handleGameOver(data) {
  clearNextGameInterval();
  clearWinnerShowTimeout();
  cancelColorSelect();
  setInteractionForTurn(null);

  const summary = recordMatchResult({
    winner: data.winner,
    turnCount: engine.turnCount,
  });
  const outcomeTag =
    data.winner === humanPlayerId
      ? "human-win"
      : data.winner === botPlayerId
        ? "ai-win"
        : "draw";
  trackEvent(`game-over-${outcomeTag}`, `Game Over: ${outcomeTag}`);

  winnerShowTimeout = setTimeout(() => {
    winnerShowTimeout = null;
    winnerContainer.classList.remove("hidden");
    winnerWrapper.classList.add("ScaleUp");

    if (data.winner === humanPlayerId) {
      winnerUsername.textContent = `${humanUsername} WINS!`;
      winnerAvatar.src = humanAvatarUrl;
    } else if (data.winner === botPlayerId) {
      winnerUsername.textContent = "ATARAXOS AI WINS!";
      winnerAvatar.src = "resources/defaultAvatar.png";
    } else {
      winnerUsername.textContent = "DRAW";
      winnerAvatar.src = "resources/defaultAvatar.png";
    }

    if (winnerSummary) {
      winnerSummary.textContent = `Finished in ${engine.turnCount} turns • Record: ${summary.humanWins}W – ${summary.botWins}L (${summary.winRate}%)`;
    }
    if (winnerReturnBtn) winnerReturnBtn.focus();
  }, motionDuration(500));

  let sec = CONFIG.NEXT_GAME_TIMEOUT;
  winnerTimeout.textContent = `Next round in ${sec}s`;
  nextGameInterval = setInterval(() => {
    sec -= 1;
    if (sec > 0) {
      winnerTimeout.textContent = `Next round in ${sec}s`;
    } else {
      startPlayAgain();
    }
  }, 1000);
}

function startPlayAgain() {
  clearNextGameInterval();
  clearWinnerShowTimeout();
  winnerContainer.classList.add("hidden");
  winnerWrapper.classList.remove("ScaleUp");
  modelClient.resetGame();
  engine.startNewGame();
}

winnerReturnBtn.addEventListener("click", startPlayAgain);

// Re-layout fans after viewport changes without rebuilding any cards.
window.addEventListener("resize", () => {
  if (resizeFrame) return;
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = null;
    layoutFan(cardsContainer, ".card", 90, 680);
    layoutFan(botCardsContainer, ".bot-card", 76, 460);
  });
});
