const PRESETS = {
  soft: {
    durationMs: 1200,
    waveStepMs: 48,
    tileFlipMs: 220,
    confettiRateRange: [16, 28],
    particleCountPerBurst: [5, 10],
    palette: "zen"
  },
  default: {
    durationMs: 1600,
    waveStepMs: 42,
    tileFlipMs: 260,
    confettiRateRange: [24, 48],
    particleCountPerBurst: [8, 18],
    palette: "default"
  },
  festival: {
    durationMs: 2100,
    waveStepMs: 36,
    tileFlipMs: 280,
    confettiRateRange: [36, 72],
    particleCountPerBurst: [12, 24],
    palette: "festival"
  }
};

const PALETTES = {
  zen: ["#b8e0d2", "#f2c6b6", "#f6d89b", "#8ecae6", "#a8dadc", "#cdb4db", "#ffd6a5", "#90be6d"],
  default: ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f78c6b", "#c77dff", "#f4a261", "#2a9d8f", "#ff7b72", "#7dd3fc"],
  festival: ["#ff006e", "#8338ec", "#3a86ff", "#ffbe0b", "#fb5607", "#06d6a0", "#ff5d8f", "#7b2cbf", "#00bbf9", "#80ed99"]
};

const controls = {
  durationMs: document.getElementById("durationMs"),
  waveStepMs: document.getElementById("waveStepMs"),
  tileFlipMs: document.getElementById("tileFlipMs"),
  rateMin: document.getElementById("rateMin"),
  rateMax: document.getElementById("rateMax"),
  burstMin: document.getElementById("burstMin"),
  burstMax: document.getElementById("burstMax"),
  palette: document.getElementById("palette"),
  reducedMotion: document.getElementById("reducedMotion")
};

const fpsStat = document.getElementById("fpsStat");
const particleStat = document.getElementById("particleStat");
const durationStat = document.getElementById("durationStat");
const boardEl = document.getElementById("fxBoard");
const previewEl = document.getElementById("fxPreview");

const PLAY_GIVENS = "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
const PLAY_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179";
const FILLED_INDICES = [3, 5, 8, 16, 28, 36, 38, 40, 58, 59, 61, 62, 66, 67, 74, 76];

let fx = null;

function buildBoard() {
  boardEl.innerHTML = "";
  const userFillSet = new Set(FILLED_INDICES);

  for (let i = 0; i < 81; i += 1) {
    const r = Math.floor(i / 9);
    const c = i % 9;
    const cell = document.createElement("div");
    cell.className = "cell";
    if (c === 2 || c === 5) cell.classList.add("box-right");
    if (r === 2 || r === 5) cell.classList.add("box-bottom");

    const valueEl = document.createElement("span");
    valueEl.className = "cell-value";

    const givenChar = PLAY_GIVENS[i];
    if (givenChar !== "0" && givenChar !== ".") {
      cell.classList.add("is-given");
      valueEl.textContent = givenChar;
    } else if (userFillSet.has(i)) {
      valueEl.textContent = PLAY_SOLUTION[i];
    } else {
      valueEl.textContent = "";
    }

    cell.appendChild(valueEl);
    boardEl.appendChild(cell);
  }
}

function intFrom(el, fallback) {
  const num = Number(el.value);
  if (Number.isFinite(num)) return Math.floor(num);
  return fallback;
}

function configFromControls() {
  const durationMs = intFrom(controls.durationMs, 1600);
  const waveStepMs = intFrom(controls.waveStepMs, 42);
  const tileFlipMs = intFrom(controls.tileFlipMs, 260);

  const rateMinRaw = intFrom(controls.rateMin, 24);
  const rateMaxRaw = intFrom(controls.rateMax, 48);
  const rateMin = Math.max(4, Math.min(rateMinRaw, rateMaxRaw));
  const rateMax = Math.max(rateMin, rateMaxRaw);

  const burstMinRaw = intFrom(controls.burstMin, 8);
  const burstMaxRaw = intFrom(controls.burstMax, 18);
  const burstMin = Math.max(1, Math.min(burstMinRaw, burstMaxRaw));
  const burstMax = Math.max(burstMin, burstMaxRaw);

  const reducedValue = controls.reducedMotion.value;
  const reducedMotion = reducedValue === "on" ? true : reducedValue === "off" ? false : "auto";

  return {
    durationMs,
    waveStepMs,
    tileFlipMs,
    confettiRateRange: [rateMin, rateMax],
    particleCountPerBurst: [burstMin, burstMax],
    palette: PALETTES[controls.palette.value] || PALETTES.default,
    reducedMotion,
    showCenterMessage: true,
    messageText: "Completed"
  };
}

function updateStats(stats) {
  const fps = stats && Number.isFinite(stats.fps) ? stats.fps : 0;
  const particles = stats && Number.isFinite(stats.particles) ? stats.particles : 0;
  const duration = stats && Number.isFinite(stats.durationMs) ? stats.durationMs : intFrom(controls.durationMs, 1600);
  fpsStat.textContent = `FPS: ${fps || "--"}`;
  particleStat.textContent = `Particles: ${particles}`;
  durationStat.textContent = `Duration: ${duration}ms`;
}

function applyPreset(name) {
  const preset = PRESETS[name] || PRESETS.default;
  controls.durationMs.value = String(preset.durationMs);
  controls.waveStepMs.value = String(preset.waveStepMs);
  controls.tileFlipMs.value = String(preset.tileFlipMs);
  controls.rateMin.value = String(preset.confettiRateRange[0]);
  controls.rateMax.value = String(preset.confettiRateRange[1]);
  controls.burstMin.value = String(preset.particleCountPerBurst[0]);
  controls.burstMax.value = String(preset.particleCountPerBurst[1]);
  controls.palette.value = preset.palette;

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.preset === name);
  });

  fx.setConfig(configFromControls());
  updateStats({ fps: 0, particles: 0, durationMs: preset.durationMs });
}

function wireEvents() {
  document.getElementById("playBtn").addEventListener("click", () => {
    fx.setConfig(configFromControls());
    fx.play();
  });

  document.getElementById("stopBtn").addEventListener("click", () => {
    fx.stop();
  });

  document.getElementById("replayBtn").addEventListener("click", () => {
    fx.stop();
    fx.setConfig(configFromControls());
    fx.play();
  });

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyPreset(btn.dataset.preset);
    });
  });

  Object.values(controls).forEach((el) => {
    el.addEventListener("change", () => {
      fx.setConfig(configFromControls());
      updateStats({ fps: 0, particles: 0, durationMs: intFrom(controls.durationMs, 1600) });
    });
  });
}

function init() {
  buildBoard();
  fx = createCelebrationFx({
    containerEl: previewEl,
    boardEl,
    showCenterMessage: true,
    messageText: "Completed",
    onFrameStats: updateStats
  });
  applyPreset("default");
  wireEvents();
}

init();
