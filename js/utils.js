// Browser-side asset preloading, avatar compositing & small UI utilities.

const CARD_COLORS = ["RED", "GREEN", "BLUE", "YELLOW"];
const CARD_TYPES = [
  "ZERO",
  "ONE",
  "TWO",
  "THREE",
  "FOUR",
  "FIVE",
  "SIX",
  "SEVEN",
  "EIGHT",
  "NINE",
  "BLOCK",
  "REVERSE",
  "PLUS_TWO",
];
const WILD_TYPES = ["COLOR_CHANGE", "PLUS_FOUR"];
const SOUND_FILES = [
  "block.mp3",
  "card_pickup.mp3",
  "card_place.mp3",
  "change_color.mp3",
  "choose_color.mp3",
  "color_change.mp3",
  "color_press.mp3",
  "plus_four.mp3",
  "plus_two.mp3",
  "reverse.mp3",
];

const imageCache = [];
const audioCache = new Map();
let preloadPromise = null;

function imagePath(src) {
  return `resources/cards/${src}.png`;
}

function collectImageSources() {
  const sources = [
    "resources/cards/UNO_CARD.png",
    "resources/cards/ANY_COLOR_CHANGE.png",
    "resources/cards/ANY_PLUS_FOUR.png",
    "resources/table.png",
    "resources/desk.png",
    "resources/defaultAvatar.png",
    "resources/UNO.png",
    "resources/crown.png",
    "resources/winner.png",
    "resources/gameBackground.jpg",
    "resources/background.jpg",
  ];

  for (const color of CARD_COLORS) {
    for (const type of CARD_TYPES) {
      sources.push(imagePath(`${color}_${type}`));
    }
    for (const type of WILD_TYPES) {
      sources.push(imagePath(`${color}_${type}`));
      sources.push(`resources/cards/gifs/${color}_${type}.gif`);
    }
  }
  return sources;
}

export function preloadAssets() {
  if (preloadPromise) return preloadPromise;

  const loadImage = (src) =>
    new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        if (typeof img.decode === "function") {
          img
            .decode()
            .catch(() => {})
            .finally(resolve);
        } else {
          resolve();
        }
      };
      img.onerror = () => resolve();
      img.src = src;
      imageCache.push(img);
    });

  const loadSound = (file) =>
    new Promise((resolve) => {
      const audio = new Audio(`resources/sounds/${file}`);
      audio.preload = "auto";
      audio.addEventListener("canplaythrough", resolve, { once: true });
      audio.addEventListener("error", resolve, { once: true });
      // Some browsers never fire canplaythrough for very short clips; don't block forever.
      setTimeout(resolve, 1500);
      audio.load();
      audioCache.set(`resources/sounds/${file}`, audio);
    });

  const tasks = [
    ...collectImageSources().map((src) => () => loadImage(src)),
    ...SOUND_FILES.map((file) => () => loadSound(file)),
  ];

  // Limit concurrent asset fetches so HTTP/1.1 static servers aren't saturated.
  const CONCURRENCY = 6;
  let nextIdx = 0;
  const workers = Array.from({ length: Math.min(CONCURRENCY, tasks.length) }, async () => {
    while (nextIdx < tasks.length) {
      const idx = nextIdx++;
      await tasks[idx]();
    }
  });

  preloadPromise = Promise.all(workers).catch(() => {});
  return preloadPromise;
}

export function playSound(fileName) {
  try {
    let audio = audioCache.get(fileName);
    if (!audio) {
      audio = new Audio(fileName);
      audio.preload = "auto";
      audioCache.set(fileName, audio);
    }
    audio.currentTime = 0;
    const played = audio.play();
    if (played && typeof played.catch === "function") {
      played.catch(() => {
        // Autoplay policies can ignore sounds before first user interaction.
      });
    }
  } catch (_) {}
}

export function setOverlay(elementId, src) {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.src = src;
  el.classList.remove("popup");
  void el.offsetWidth; // Trigger reflow for CSS animation
  el.classList.add("popup");
}

export function setCover(src) {
  const el = document.getElementById("cover");
  if (!el) return;
  el.src = src;
  el.classList.remove("popupCover");
  void el.offsetWidth;
  el.classList.add("popupCover");
}

export function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}
