// Persistent Match/AI Stats Tracker & Privacy-Friendly GoatCounter Analytics
import { CONFIG } from "./config.js";

const STATS_STORAGE_KEY = "ataraxos_uno_stats_v1";
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
const MAX_RECENT_MATCHES = 8;

function defaultStats() {
  return {
    gamesPlayed: 0,
    humanWins: 0,
    botWins: 0,
    draws: 0,
    currentStreak: 0,
    bestStreak: 0,
    totalTurns: 0,
    fastestWinTurns: null,
    totalAiDecisions: 0,
    totalAiLatencyMs: 0,
    lastAiLatencyMs: null,
    lastBackend: "",
    recentMatches: [],
  };
}

function loadStats() {
  try {
    const raw = localStorage.getItem(STATS_STORAGE_KEY);
    if (!raw) return defaultStats();
    const parsed = JSON.parse(raw);
    return { ...defaultStats(), ...parsed, recentMatches: Array.isArray(parsed.recentMatches) ? parsed.recentMatches : [] };
  } catch (_) {
    return defaultStats();
  }
}

function saveStats(stats) {
  try {
    localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify(stats));
  } catch (_) {
    // Ignore storage errors in private browsing / quota limits.
  }
}

let statsState = loadStats();

export function recordMatchResult({ winner, turnCount = 0 }) {
  const turns = Math.max(0, Number(turnCount) || 0);
  statsState.gamesPlayed += 1;
  statsState.totalTurns += turns;

  if (winner === "human") {
    statsState.humanWins += 1;
    statsState.currentStreak += 1;
    if (statsState.currentStreak > statsState.bestStreak) {
      statsState.bestStreak = statsState.currentStreak;
    }
    if (
      turns > 0 &&
      (statsState.fastestWinTurns === null || turns < statsState.fastestWinTurns)
    ) {
      statsState.fastestWinTurns = turns;
    }
  } else if (winner === "bot") {
    statsState.botWins += 1;
    statsState.currentStreak = 0;
  } else {
    statsState.draws += 1;
  }

  statsState.recentMatches.unshift({
    winner,
    turns,
    ts: Date.now(),
  });
  if (statsState.recentMatches.length > MAX_RECENT_MATCHES) {
    statsState.recentMatches.length = MAX_RECENT_MATCHES;
  }

  saveStats(statsState);
  return getStatsSummary();
}

export function recordAiDecision({ latencyMs, backend }) {
  const ms = Number(latencyMs);
  if (Number.isFinite(ms) && ms >= 0) {
    statsState.totalAiDecisions += 1;
    statsState.totalAiLatencyMs += ms;
    statsState.lastAiLatencyMs = ms;
  }
  if (backend) {
    statsState.lastBackend = String(backend);
  }
  saveStats(statsState);
}

export function getStatsSummary() {
  const s = statsState;
  const winRate =
    s.gamesPlayed > 0 ? ((s.humanWins / s.gamesPlayed) * 100).toFixed(1) : "0.0";
  const avgTurns =
    s.gamesPlayed > 0 ? (s.totalTurns / s.gamesPlayed).toFixed(1) : "0.0";
  const avgLatencyMs =
    s.totalAiDecisions > 0
      ? Math.round(s.totalAiLatencyMs / s.totalAiDecisions)
      : null;

  return {
    ...s,
    winRate,
    avgTurns,
    avgLatencyMs,
  };
}

export function resetStats() {
  statsState = defaultStats();
  saveStats(statsState);
  return getStatsSummary();
}

// ---------------------------------------------------------------------------
// GoatCounter Visitor & Event Analytics
// ---------------------------------------------------------------------------

function normalizeGoatCounterEndpoint(rawCode) {
  const code = (rawCode || "").trim();
  if (!code || code.startsWith("YOUR_")) return null;
  if (code.startsWith("https://") || code.startsWith("http://")) {
    return code.endsWith("/count") ? code : `${code.replace(/\/+$/, "")}/count`;
  }
  return `https://${code}.goatcounter.com/count`;
}

export function getGoatCounterDashboardUrl() {
  const endpoint = normalizeGoatCounterEndpoint(CONFIG.GOATCOUNTER_CODE);
  if (!endpoint) return null;
  try {
    return new URL(endpoint).origin;
  } catch (_) {
    return null;
  }
}

export function initAnalytics() {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  const endpoint = normalizeGoatCounterEndpoint(CONFIG.GOATCOUNTER_CODE);
  if (!endpoint) return;

  // Avoid noisy tracking hits from localhost during local development.
  const host = window.location && window.location.hostname;
  if (LOCAL_HOSTS.has(host)) return;

  if (document.querySelector('script[data-goatcounter]')) return;

  const script = document.createElement("script");
  script.async = true;
  script.src = "https://gc.zgo.at/count.js";
  script.crossOrigin = "anonymous";
  script.setAttribute("data-goatcounter", endpoint);
  document.head.appendChild(script);
}

export function trackEvent(path, title = "") {
  try {
    if (
      typeof window !== "undefined" &&
      window.goatcounter &&
      typeof window.goatcounter.count === "function"
    ) {
      window.goatcounter.count({
        path,
        title: title || path,
        event: true,
      });
    }
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Native <dialog> Stats Modal
// ---------------------------------------------------------------------------

function formatBackendLabel(backend) {
  switch (backend) {
    case "browser_webgpu":
      return "In-Browser WebGPU";
    case "browser_wasm":
      return "In-Browser WASM";
    case "server":
      return "Python Daemon";
    default:
      return "Waiting for move";
  }
}

function formatRelativeTime(ts) {
  if (!ts) return "";
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

export function initStatsDialog({ onReset } = {}) {
  const dialog = document.getElementById("stats-dialog");
  if (!dialog) return { open: () => {}, refresh: () => {} };

  const closeBtn = document.getElementById("stats-close-btn");
  const resetBtn = document.getElementById("stats-reset-btn");
  let confirmingReset = false;
  let confirmTimer = null;

  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  function refresh() {
    const s = getStatsSummary();
    setText("stat-games", String(s.gamesPlayed));
    setText("stat-wins", String(s.humanWins));
    setText("stat-losses", String(s.botWins));
    setText("stat-winrate", `${s.winRate}%`);
    setText("stat-streak", `${s.currentStreak} (Best ${s.bestStreak})`);
    setText("stat-avg-turns", s.avgTurns);
    setText(
      "stat-fastest-win",
      s.fastestWinTurns !== null ? `${s.fastestWinTurns} turns` : "—",
    );

    setText("stat-backend", formatBackendLabel(s.lastBackend));
    setText("stat-ai-moves", String(s.totalAiDecisions));
    setText(
      "stat-avg-latency",
      s.avgLatencyMs !== null ? `${s.avgLatencyMs} ms` : "—",
    );
    setText(
      "stat-last-latency",
      s.lastAiLatencyMs !== null ? `${Math.round(s.lastAiLatencyMs)} ms` : "—",
    );

    const recentList = document.getElementById("stats-recent-list");
    if (recentList) {
      recentList.innerHTML = "";
      if (s.recentMatches.length === 0) {
        const li = document.createElement("li");
        li.className = "stats-empty";
        li.textContent = "No completed matches yet — play a round to record stats.";
        recentList.appendChild(li);
      } else {
        for (const m of s.recentMatches) {
          const li = document.createElement("li");
          li.className = "stats-recent-item";

          const badge = document.createElement("span");
          badge.className = `stats-result-badge ${
            m.winner === "human"
              ? "win"
              : m.winner === "bot"
                ? "loss"
                : "draw"
          }`;
          badge.textContent =
            m.winner === "human" ? "WIN" : m.winner === "bot" ? "LOSS" : "DRAW";

          const detail = document.createElement("span");
          detail.textContent = `${m.turns} turns`;

          const time = document.createElement("span");
          time.className = "stats-recent-time";
          time.textContent = formatRelativeTime(m.ts);

          li.appendChild(badge);
          li.appendChild(detail);
          li.appendChild(time);
          recentList.appendChild(li);
        }
      }
    }

    const analyticsInfo = document.getElementById("stats-analytics-status");
    if (analyticsInfo) {
      const dashUrl = getGoatCounterDashboardUrl();
      analyticsInfo.innerHTML = "";
      if (dashUrl) {
        const label = document.createElement("span");
        label.textContent = "Site visitor analytics: Active ";
        const link = document.createElement("a");
        link.href = dashUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "(Open GoatCounter Dashboard)";
        analyticsInfo.appendChild(label);
        analyticsInfo.appendChild(link);
      } else {
        analyticsInfo.textContent =
          'Visitor analytics: Set your GoatCounter site code in <meta name="goatcounter-code"> inside web/index.html.';
      }
    }
  }

  function clearResetConfirm() {
    confirmingReset = false;
    if (confirmTimer) {
      clearTimeout(confirmTimer);
      confirmTimer = null;
    }
    if (resetBtn) {
      resetBtn.textContent = "Reset Stats";
      resetBtn.classList.remove("confirming");
    }
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => dialog.close());
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (!confirmingReset) {
        confirmingReset = true;
        resetBtn.textContent = "Click again to confirm";
        resetBtn.classList.add("confirming");
        confirmTimer = setTimeout(clearResetConfirm, 3500);
        return;
      }
      clearResetConfirm();
      resetStats();
      refresh();
      if (onReset) onReset();
    });
  }

  dialog.addEventListener("close", clearResetConfirm);

  // Light-dismiss fallback for browsers without native <dialog closedby="any">
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e) => {
      const r = dialog.getBoundingClientRect();
      return (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      );
    };
    let pressedOutside = false;
    dialog.addEventListener("pointerdown", (e) => {
      pressedOutside = outside(e);
    });
    dialog.addEventListener("pointerup", (e) => {
      if (pressedOutside && outside(e)) dialog.close();
    });
  }

  return {
    open() {
      refresh();
      if (typeof dialog.showModal === "function") {
        if (!dialog.open) dialog.showModal();
      } else {
        dialog.setAttribute("open", "");
      }
      trackEvent("stats-open", "Opened Stats Modal");
    },
    refresh,
  };
}
