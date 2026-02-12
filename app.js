const SAMPLE_PUZZLES = [
  {
    id: "book-001",
    givens: "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    solution: "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
    source: "book",
    difficulty: 1,
    createdAt: "2026-02-09",
    importedAt: "2026-02-09"
  },
  {
    id: "book-002",
    givens: "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    solution: "483921657967345821251876493548132976729564138136798245372689514814253769695417382",
    source: "book",
    difficulty: 2,
    createdAt: "2026-02-09",
    importedAt: "2026-02-09"
  },
  {
    id: "book-003",
    givens: "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
    solution: "245986371169273584837514269976125438513498627482736915391657842728349156654812793",
    source: "book",
    difficulty: 3,
    createdAt: "2026-02-09",
    importedAt: "2026-02-09"
  }
];

const DB_NAME = "misudoku";
const DB_VERSION = 1;

const state = {
  puzzle: null,
  givens: "",
  grid: [],
  notes: [],
  selectedIndex: null,
  noteMode: false,
  conflictCheck: true,
  paused: false,
  attempts: 0,
  puzzleStartedAt: null,
  sessionStart: null,
  elapsedMs: 0,
  history: [],
  historyIndex: -1,
  sourceMode: "book",
  genDifficulty: 1,
  relatedHighlight: true,
  sameValueHighlight: false,
  celebrationTheme: "default",
  completed: false
};

let dbPromise = null;
let cellElements = [];
let timerId = null;
let saveTimer = null;
let selectedPuzzleId = null;
let showImportExport = false;
let celebrationFx = null;
let completionPresentationToken = 0;

const appRoot = document.getElementById("appRoot");
const boardEl = document.getElementById("board");
const puzzleMetaEl = document.getElementById("puzzleMeta");
const timerEl = document.getElementById("timer");
const attemptsEl = document.getElementById("attempts");
const noteToggle = document.getElementById("noteToggle");
const undoBtn = document.getElementById("undoBtn");
const conflictToggle = document.getElementById("conflictToggle");
const pauseBtn = document.getElementById("pauseBtn");
const puzzleBtn = document.getElementById("puzzleBtn");
const settingsBtn = document.getElementById("settingsBtn");
const themeToggle = document.getElementById("themeToggle");
const sourceSeg = document.getElementById("sourceSeg");
const toggleImportExportBtn = document.getElementById("toggleImportExport");
const bookListEl = document.getElementById("bookList");
const selectPuzzleBtn = document.getElementById("selectPuzzleBtn");
const bookPanel = document.getElementById("bookPanel");
const importExportPanel = document.getElementById("importExportPanel");
const generatedPanel = document.getElementById("generatedPanel");
const genDifficultyRow = document.getElementById("generatedDifficultyRow");
const genDifficultySeg = document.getElementById("genDifficulty");
const keypad = document.querySelector(".keypad");

const puzzleDialog = document.getElementById("puzzleDialog");
const closePuzzle = document.getElementById("closePuzzle");
const settingsDialog = document.getElementById("settingsDialog");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const relatedHighlightToggle = document.getElementById("relatedHighlightToggle");
const sameDigitHighlightToggle = document.getElementById("sameDigitHighlightToggle");
const celebrationThemeSeg = document.getElementById("celebrationThemeSeg");
const importFile = document.getElementById("importFile");
const replaceMode = document.getElementById("replaceMode");
const importBtn = document.getElementById("importBtn");
const downloadTemplateBtn = document.getElementById("downloadTemplateBtn");
const importStatus = document.getElementById("importStatus");
const exportBtn = document.getElementById("exportBtn");

const winDialog = document.getElementById("winDialog");
const winSummary = document.getElementById("winSummary");
const winNextBtn = document.getElementById("winNextBtn");
const winCloseBtn = document.getElementById("winCloseBtn");

const resetPuzzleBtn = document.getElementById("resetPuzzleBtn");
const confirmResetDialog = document.getElementById("confirmResetDialog");
const confirmResetBtn = document.getElementById("confirmResetBtn");
const cancelResetBtn = document.getElementById("cancelResetBtn");

const CELEBRATION_THEMES = {
  zen: {
    durationMs: 1200,
    waveStepMs: 48,
    tileFlipMs: 220,
    confettiRateRange: [16, 28],
    particleCountPerBurst: [5, 10],
    palette: ["#b8e0d2", "#f2c6b6", "#f6d89b", "#8ecae6", "#a8dadc", "#cdb4db", "#ffd6a5", "#90be6d"]
  },
  default: {
    durationMs: 1600,
    waveStepMs: 42,
    tileFlipMs: 260,
    confettiRateRange: [24, 48],
    particleCountPerBurst: [8, 18],
    palette: ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f78c6b", "#c77dff", "#f4a261", "#2a9d8f", "#ff7b72", "#7dd3fc"]
  },
  festival: {
    durationMs: 2100,
    waveStepMs: 36,
    tileFlipMs: 280,
    confettiRateRange: [36, 72],
    particleCountPerBurst: [12, 24],
    palette: ["#ff006e", "#8338ec", "#3a86ff", "#ffbe0b", "#fb5607", "#06d6a0", "#ff5d8f", "#7b2cbf", "#00bbf9", "#80ed99"]
  }
};

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("puzzles")) {
        db.createObjectStore("puzzles", { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains("progress")) {
        db.createObjectStore("progress", { keyPath: "puzzleId" });
      }
      if (!db.objectStoreNames.contains("meta")) {
        db.createObjectStore("meta", { keyPath: "key" });
      }
    };
  });
  return dbPromise;
}

function withStore(storeName, mode, action) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(storeName, mode);
        const store = tx.objectStore(storeName);
        let request;
        try {
          request = action(store);
        } catch (error) {
          reject(error);
          return;
        }
        if (request && typeof request.onsuccess !== "undefined") {
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        } else {
          tx.oncomplete = () => resolve(request);
          tx.onerror = () => reject(tx.error);
        }
      })
  );
}

function dbGet(storeName, key) {
  return withStore(storeName, "readonly", (store) => store.get(key));
}

function dbGetAll(storeName) {
  return withStore(storeName, "readonly", (store) => store.getAll());
}

function dbPut(storeName, value) {
  return withStore(storeName, "readwrite", (store) => store.put(value));
}

function dbClear(storeName) {
  return withStore(storeName, "readwrite", (store) => store.clear());
}

function dbDelete(storeName, key) {
  return withStore(storeName, "readwrite", (store) => store.delete(key));
}

function normalizeGridString(str) {
  return str.replace(/[^0-9.]/g, "");
}

function stringToGrid(str) {
  const clean = normalizeGridString(str);
  return clean.split("").map((char) => (char === "0" ? "." : char));
}

function normalizeGivens(str) {
  return gridToString(stringToGrid(str));
}

function gridToString(grid) {
  return grid.map((char) => (char === "0" ? "." : char)).join("");
}

function emptyNotes() {
  return Array.from({ length: 81 }, () => 0);
}

function buildBoard() {
  boardEl.innerHTML = "";
  cellElements = [];
  for (let i = 0; i < 81; i += 1) {
    const row = Math.floor(i / 9);
    const col = i % 9;
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cell";
    cell.dataset.index = String(i);
    if (col === 2 || col === 5) cell.classList.add("box-right");
    if (row === 2 || row === 5) cell.classList.add("box-bottom");

    const valueEl = document.createElement("span");
    valueEl.className = "cell-value";
    cell.appendChild(valueEl);

    const notesEl = document.createElement("div");
    notesEl.className = "cell-notes";
    for (let n = 1; n <= 9; n += 1) {
      const note = document.createElement("span");
      note.dataset.note = String(n);
      note.textContent = String(n);
      notesEl.appendChild(note);
    }
    cell.appendChild(notesEl);

    cell.addEventListener("click", () => selectCell(i));
    boardEl.appendChild(cell);
    cellElements.push(cell);
  }
}

function difficultyLabel(level) {
  switch (Number(level)) {
    case 1:
      return "简单";
    case 2:
      return "中等";
    case 3:
      return "困难";
    case 4:
      return "专家";
    default:
      return "未标注";
  }
}

function formatTime(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function getElapsedMs() {
  if (state.paused || !state.sessionStart) return state.elapsedMs;
  return state.elapsedMs + (Date.now() - state.sessionStart);
}

function updateTimer() {
  timerEl.textContent = formatTime(getElapsedMs());
}

function startTimer() {
  if (timerId) clearInterval(timerId);
  if (state.paused) return;
  state.sessionStart = Date.now();
  timerId = setInterval(updateTimer, 1000);
  updateTimer();
}

function pauseTimer() {
  if (state.sessionStart) {
    state.elapsedMs = getElapsedMs();
  }
  state.sessionStart = null;
  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }
  updateTimer();
}

function setPaused(paused) {
  state.paused = paused;
  appRoot.classList.toggle("is-paused", paused);
  pauseBtn.textContent = paused ? "继续" : "暂停";
  if (paused) {
    pauseTimer();
  } else {
    startTimer();
  }
  scheduleSave();
}

function scheduleSave() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(saveProgress, 400);
}

async function saveProgress() {
  if (!state.puzzle) return;
  const progress = {
    puzzleId: state.puzzle.id,
    startedAt: state.puzzleStartedAt,
    elapsedMs: getElapsedMs(),
    attempts: state.attempts,
    grid: gridToString(state.grid),
    notes: state.notes,
    status: state.completed ? "completed" : "in_progress",
    lastUpdated: new Date().toISOString()
  };
  await dbPut("progress", progress);
  await dbPut("meta", { key: "currentPuzzleId", value: state.puzzle.id });
}

function pushHistory() {
  const snapshot = {
    grid: gridToString(state.grid),
    notes: [...state.notes],
    attempts: state.attempts
  };
  const prev = state.history[state.historyIndex];
  if (prev && prev.grid === snapshot.grid && JSON.stringify(prev.notes) === JSON.stringify(snapshot.notes)) {
    return;
  }
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(snapshot);
  if (state.history.length > 200) {
    state.history.shift();
  }
  state.historyIndex = state.history.length - 1;
  updateUndoRedo();
}

function applySnapshot(snapshot) {
  state.grid = stringToGrid(snapshot.grid);
  state.notes = [...snapshot.notes];
  state.attempts = snapshot.attempts;
  renderBoardState();
  updateAttempts();
  scheduleSave();
}

function undo() {
  if (state.historyIndex <= 0) return;
  state.historyIndex -= 1;
  const snapshot = state.history[state.historyIndex];
  applySnapshot(snapshot);
  updateUndoRedo();
}

function redo() {
  if (state.historyIndex >= state.history.length - 1) return;
  state.historyIndex += 1;
  const snapshot = state.history[state.historyIndex];
  applySnapshot(snapshot);
  updateUndoRedo();
}

function updateUndoRedo() {
  if (undoBtn) undoBtn.disabled = state.historyIndex <= 0;
  const redoBtn = document.getElementById("redoBtn");
  if (redoBtn) redoBtn.disabled = state.historyIndex >= state.history.length - 1;
}

function selectCell(index) {
  state.selectedIndex = index;
  renderBoardState();
}

function isGiven(index) {
  return state.givens[index] !== ".";
}

function clearCell() {
  if (!state.puzzle) return;
  if (state.completed) return;
  if (state.selectedIndex === null || state.paused) return;
  if (isGiven(state.selectedIndex)) return;
  if (state.grid[state.selectedIndex] === "." && state.notes[state.selectedIndex] === 0) return;
  state.grid[state.selectedIndex] = ".";
  state.notes[state.selectedIndex] = 0;
  pushHistory();
  renderBoardState();
  scheduleSave();
}

function clearNotes() {
  if (!state.puzzle) return;
  if (state.completed) return;
  if (state.selectedIndex === null || state.paused) return;
  if (state.notes[state.selectedIndex] === 0) return;
  state.notes[state.selectedIndex] = 0;
  pushHistory();
  renderBoardState();
  scheduleSave();
}

function setCellValue(index, value) {
  if (!state.puzzle) return;
  if (state.completed) return;
  if (state.paused) return;
  if (isGiven(index)) return;
  if (state.noteMode) {
    const mask = 1 << (value - 1);
    state.notes[index] ^= mask;
    pushHistory();
  } else {
    const prev = state.grid[index];
    if (prev !== String(value)) {
      state.grid[index] = String(value);
      state.notes[index] = 0;
      state.attempts += 1;
      pushHistory();
      checkCompletion();
    }
  }
  renderBoardState();
  updateAttempts();
  scheduleSave();
}

function handleKeyInput(key) {
  if (!state.puzzle) return;
  if (state.completed) return;
  if (state.selectedIndex === null) return;
  if (key === "clear") {
    clearCell();
    return;
  }
  const value = Number(key);
  if (!Number.isNaN(value) && value >= 1 && value <= 9) {
    setCellValue(state.selectedIndex, value);
  }
}

function renderBoardState() {
  const conflicts = state.conflictCheck ? computeConflicts() : new Set();
  const selectedValue = state.selectedIndex !== null ? state.grid[state.selectedIndex] : ".";
  cellElements.forEach((cell, index) => {
    const valueEl = cell.querySelector(".cell-value");
    const notesEl = cell.querySelector(".cell-notes");
    const value = state.grid[index];
    valueEl.textContent = value === "." ? "" : value;
    cell.classList.toggle("is-given", isGiven(index));
    cell.classList.toggle("is-selected", index === state.selectedIndex);
    cell.classList.toggle("is-related", state.relatedHighlight && isRelated(index, state.selectedIndex));
    cell.classList.toggle(
      "is-same-value",
      state.sameValueHighlight &&
        selectedValue !== "." &&
        value === selectedValue &&
        index !== state.selectedIndex
    );
    cell.classList.toggle("is-conflict", conflicts.has(index));

    if (value !== ".") {
      notesEl.style.opacity = "0";
    } else {
      notesEl.style.opacity = "1";
      const mask = state.notes[index];
      notesEl.querySelectorAll("span").forEach((note) => {
        const digit = Number(note.dataset.note);
        note.classList.toggle("active", (mask & (1 << (digit - 1))) !== 0);
      });
    }
  });
}

function isRelated(index, selectedIndex) {
  if (selectedIndex === null) return false;
  if (index === selectedIndex) return false;
  const row = Math.floor(index / 9);
  const col = index % 9;
  const selRow = Math.floor(selectedIndex / 9);
  const selCol = selectedIndex % 9;
  const box = Math.floor(row / 3) * 3 + Math.floor(col / 3);
  const selBox = Math.floor(selRow / 3) * 3 + Math.floor(selCol / 3);
  return row === selRow || col === selCol || box === selBox;
}

function computeConflicts() {
  const conflicts = new Set();

  function checkUnit(indices) {
    const map = {};
    indices.forEach((idx) => {
      const value = state.grid[idx];
      if (value === ".") return;
      if (!map[value]) map[value] = [];
      map[value].push(idx);
    });
    Object.values(map).forEach((list) => {
      if (list.length > 1) {
        list.forEach((idx) => conflicts.add(idx));
      }
    });
  }

  for (let r = 0; r < 9; r += 1) {
    checkUnit(Array.from({ length: 9 }, (_, c) => r * 9 + c));
  }
  for (let c = 0; c < 9; c += 1) {
    checkUnit(Array.from({ length: 9 }, (_, r) => r * 9 + c));
  }
  for (let br = 0; br < 3; br += 1) {
    for (let bc = 0; bc < 3; bc += 1) {
      const indices = [];
      for (let r = 0; r < 3; r += 1) {
        for (let c = 0; c < 3; c += 1) {
          indices.push((br * 3 + r) * 9 + (bc * 3 + c));
        }
      }
      checkUnit(indices);
    }
  }

  return conflicts;
}

function updateAttempts() {
  attemptsEl.textContent = String(state.attempts);
}

function updatePuzzleMeta() {
  if (!state.puzzle) {
    puzzleMetaEl.textContent = "未开始";
    return;
  }
  const level = difficultyLabel(state.puzzle.difficulty);
  puzzleMetaEl.textContent = `${state.puzzle.id} · ${level}`;
}

function setNoteMode(enabled) {
  state.noteMode = enabled;
  noteToggle.classList.toggle("is-active", enabled);
  noteToggle.textContent = enabled ? "笔记中..." : "填数中...";
}

function setConflictCheck(enabled) {
  state.conflictCheck = enabled;
  conflictToggle.classList.toggle("is-active", enabled);
  renderBoardState();
  persistSettings();
}

function getTheme() {
  return document.documentElement.dataset.theme || "light";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeToggle.classList.toggle("is-active", theme === "dark");
  persistSettings();
}

function persistSettings() {
  dbPut("meta", {
    key: "settings",
    value: {
      conflictCheck: state.conflictCheck,
      theme: getTheme(),
      relatedHighlight: state.relatedHighlight,
      sameValueHighlight: state.sameValueHighlight,
      celebrationTheme: state.celebrationTheme
    }
  });
}

function setCelebrationTheme(themeName) {
  if (!CELEBRATION_THEMES[themeName]) return;
  state.celebrationTheme = themeName;
}

function syncSettingsFormFromState() {
  if (!relatedHighlightToggle || !sameDigitHighlightToggle || !celebrationThemeSeg) return;
  relatedHighlightToggle.checked = !!state.relatedHighlight;
  sameDigitHighlightToggle.checked = !!state.sameValueHighlight;
  celebrationThemeSeg.querySelectorAll(".segmented-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.celebration === state.celebrationTheme);
  });
}

function applySettingsFromForm() {
  if (!relatedHighlightToggle || !sameDigitHighlightToggle || !celebrationThemeSeg) return;
  state.relatedHighlight = !!relatedHighlightToggle.checked;
  state.sameValueHighlight = !!sameDigitHighlightToggle.checked;
  const activeThemeBtn = celebrationThemeSeg.querySelector(".segmented-btn.is-active");
  if (activeThemeBtn) {
    setCelebrationTheme(activeThemeBtn.dataset.celebration);
  }
  renderBoardState();
  persistSettings();
}

function showCompletionDialog() {
  winSummary.textContent = `用时 ${formatTime(getElapsedMs())}，尝试 ${state.attempts} 次。`;
  winDialog.showModal();
}

function cancelCompletionPresentation() {
  completionPresentationToken += 1;
  if (celebrationFx) {
    celebrationFx.stop();
  }
}

function getCelebrationThemeConfig() {
  return CELEBRATION_THEMES[state.celebrationTheme] || CELEBRATION_THEMES.default;
}

function ensureCelebrationFx() {
  if (celebrationFx) return celebrationFx;
  if (typeof window.createCelebrationFx !== "function") return null;
  const preset = getCelebrationThemeConfig();
  celebrationFx = window.createCelebrationFx({
    containerEl: appRoot,
    boardEl,
    durationMs: preset.durationMs,
    waveStepMs: preset.waveStepMs,
    tileFlipMs: preset.tileFlipMs,
    emitters: [
      { x: 0.25, y: 0.42 },
      { x: 0.5, y: 0.38 },
      { x: 0.75, y: 0.42 }
    ],
    confettiRateRange: preset.confettiRateRange,
    particleCountPerBurst: preset.particleCountPerBurst,
    palette: preset.palette,
    reducedMotion: "auto",
    showCenterMessage: true,
    messageText: "Complete",
    loopUntilAction: true
  });
  return celebrationFx;
}

function playCompletionPresentation() {
  const fx = ensureCelebrationFx();
  if (!fx) {
    showCompletionDialog();
    return;
  }

  const token = completionPresentationToken + 1;
  completionPresentationToken = token;
  const preset = getCelebrationThemeConfig();

  try {
    fx.setConfig({
      durationMs: preset.durationMs,
      waveStepMs: preset.waveStepMs,
      tileFlipMs: preset.tileFlipMs,
      confettiRateRange: preset.confettiRateRange,
      particleCountPerBurst: preset.particleCountPerBurst,
      palette: preset.palette,
      reducedMotion: "auto",
      showCenterMessage: true,
      messageText: "Complete",
      loopUntilAction: true,
      onAction: () => {
        if (token !== completionPresentationToken) return;
        showCompletionDialog();
      }
    });
    fx.play();
  } catch (error) {
    showCompletionDialog();
  }
}

function setSourceMode(mode) {
  state.sourceMode = mode;
  sourceSeg.querySelectorAll(".segmented-btn").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.source === mode);
  });
  genDifficultyRow.hidden = mode !== "generated";
  if (toggleImportExportBtn) {
    toggleImportExportBtn.hidden = mode !== "book";
  }
  if (bookPanel && generatedPanel) {
    bookPanel.hidden = mode !== "book" || showImportExport;
    generatedPanel.hidden = mode !== "generated";
  }
  if (importExportPanel) {
    importExportPanel.hidden = mode !== "book" || !showImportExport;
  }
  if (selectPuzzleBtn) {
    selectPuzzleBtn.hidden = mode !== "book" || showImportExport;
  }
}

function setGenDifficulty(level) {
  state.genDifficulty = level;
  genDifficultySeg.querySelectorAll(".segmented-btn").forEach((btn) => {
    btn.classList.toggle("is-active", Number(btn.dataset.gen) === level);
  });
}

function formatPuzzleTime(progress) {
  if (!progress || !progress.elapsedMs) return "--";
  return formatTime(progress.elapsedMs);
}

function renderBookList(puzzles, progressMap) {
  if (!bookListEl) return;
  bookListEl.innerHTML = "";
  if (!puzzles.length) {
    const empty = document.createElement("div");
    empty.className = "status";
    empty.textContent = "暂无书籍题库，请导入。";
    bookListEl.appendChild(empty);
    if (selectPuzzleBtn) selectPuzzleBtn.disabled = true;
    return;
  }
  let hasSelection = false;
  puzzles.forEach((puzzle) => {
    const progress = progressMap.get(puzzle.id);
    const row = document.createElement("button");
    row.type = "button";
    row.className = "book-row";
    if (progress?.status === "completed") {
      row.classList.add("is-completed");
    }
    if (puzzle.id === selectedPuzzleId) {
      row.classList.add("is-selected");
      hasSelection = true;
    }
    row.dataset.puzzleId = puzzle.id;
    row.innerHTML = `
      <span class="status-dot" aria-hidden="true"></span>
      <span class="book-id">${puzzle.id}</span>
      <span class="book-diff">${difficultyLabel(puzzle.difficulty)}</span>
      <span class="book-time">${formatPuzzleTime(progress)}</span>
    `;
    row.addEventListener("click", () => {
      selectedPuzzleId = puzzle.id;
      renderBookList(puzzles, progressMap);
      if (selectPuzzleBtn) selectPuzzleBtn.disabled = false;
    });
    bookListEl.appendChild(row);
  });
  if (selectPuzzleBtn) {
    if (!hasSelection) selectedPuzzleId = null;
    selectPuzzleBtn.disabled = !hasSelection;
  }
}

async function refreshBookPanel() {
  const puzzles = await dbGetAll("puzzles");
  const bookPuzzles = puzzles.filter((p) => p.source === "book");
  const progressList = await dbGetAll("progress");
  const progressMap = new Map(progressList.map((item) => [item.puzzleId, item]));
  if (!selectedPuzzleId && state.puzzle?.source === "book") {
    selectedPuzzleId = state.puzzle.id;
  }
  renderBookList(bookPuzzles, progressMap);
}

async function selectPuzzleById(puzzleId) {
  if (!puzzleId) return;
  const puzzle = await dbGet("puzzles", puzzleId);
  if (!puzzle) return;
  const progress = await dbGet("progress", puzzle.id);
  setPuzzle(puzzle, progress);
  applyPuzzleSource(puzzle);
  puzzleDialog.close();
}

function toggleImportExport() {
  showImportExport = !showImportExport;
  if (toggleImportExportBtn) {
    toggleImportExportBtn.textContent = showImportExport ? "返回题库" : "导入/导出";
  }
  setSourceMode(state.sourceMode);
}

async function resetCurrentPuzzle() {
  if (!state.puzzle) return;
  state.grid = stringToGrid(state.givens);
  state.notes = emptyNotes();
  state.attempts = 0;
  state.elapsedMs = 0;
  state.puzzleStartedAt = new Date().toISOString();
  state.completed = false;
  state.history = [];
  state.historyIndex = -1;
  pushHistory();
  setPaused(false);
  renderBoardState();
  updateAttempts();
  updateTimer();
  updateUndoRedo();
  await dbDelete("progress", state.puzzle.id);
  scheduleSave();
}

function setupEvents() {
  noteToggle.addEventListener("click", () => setNoteMode(!state.noteMode));
  undoBtn.addEventListener("click", undo);
  conflictToggle.addEventListener("click", () => setConflictCheck(!state.conflictCheck));
  pauseBtn.addEventListener("click", () => setPaused(!state.paused));

  puzzleBtn.addEventListener("click", () => {
    if (state.sourceMode === "book") {
      refreshBookPanel();
    }
    setSourceMode(state.sourceMode);
    puzzleDialog.showModal();
  });

  closePuzzle.addEventListener("click", () => puzzleDialog.close());

  settingsBtn.addEventListener("click", () => {
    syncSettingsFormFromState();
    settingsDialog.showModal();
  });

  saveSettingsBtn.addEventListener("click", () => {
    applySettingsFromForm();
    settingsDialog.close();
  });

  celebrationThemeSeg.addEventListener("click", (event) => {
    const button = event.target.closest(".segmented-btn");
    if (!button) return;
    celebrationThemeSeg.querySelectorAll(".segmented-btn").forEach((btn) => {
      btn.classList.toggle("is-active", btn === button);
    });
  });

  if (resetPuzzleBtn) {
    resetPuzzleBtn.addEventListener("click", () => confirmResetDialog.showModal());
  }
  if (confirmResetBtn) {
    confirmResetBtn.addEventListener("click", async () => {
      confirmResetDialog.close();
      await resetCurrentPuzzle();
    });
  }
  if (cancelResetBtn) {
    cancelResetBtn.addEventListener("click", () => confirmResetDialog.close());
  }
  if (toggleImportExportBtn) {
    toggleImportExportBtn.addEventListener("click", toggleImportExport);
  }
  if (selectPuzzleBtn) {
    selectPuzzleBtn.addEventListener("click", () => selectPuzzleById(selectedPuzzleId));
  }
  importBtn.addEventListener("click", handleImport);
  exportBtn.addEventListener("click", exportBackup);
  downloadTemplateBtn.addEventListener("click", downloadTemplate);
  winNextBtn.addEventListener("click", () => {
    winDialog.close();
    startNewPuzzle(state.sourceMode);
  });
  winCloseBtn.addEventListener("click", () => winDialog.close());

  themeToggle.addEventListener("click", () => {
    const nextTheme = getTheme() === "dark" ? "light" : "dark";
    setTheme(nextTheme);
  });

  puzzleDialog.addEventListener("close", () => {
    showImportExport = false;
    if (toggleImportExportBtn) toggleImportExportBtn.textContent = "导入/导出";
    setSourceMode(state.sourceMode);
  });


  keypad.addEventListener("click", (event) => {
    const button = event.target.closest(".key");
    if (!button) return;
    handleKeyInput(button.dataset.key);
  });

  sourceSeg.addEventListener("click", (event) => {
    const button = event.target.closest(".segmented-btn");
    if (!button) return;
    const mode = button.dataset.source;
    showImportExport = false;
    if (toggleImportExportBtn) toggleImportExportBtn.textContent = "导入/导出";
    setSourceMode(mode);
    if (mode === "book") {
      refreshBookPanel();
    }
  });

  genDifficultySeg.addEventListener("click", (event) => {
    const button = event.target.closest(".segmented-btn");
    if (!button) return;
    const level = Number(button.dataset.gen);
    if (Number.isNaN(level)) return;
    setGenDifficulty(level);
  });

  document.addEventListener("keydown", (event) => {
    if (puzzleDialog.open || settingsDialog.open || winDialog.open || confirmResetDialog.open) return;
    if (event.key >= "1" && event.key <= "9") {
      handleKeyInput(event.key);
    }
    if (event.key === "Backspace" || event.key === "Delete" || event.key === "0") {
      handleKeyInput("clear");
    }
    if (event.key === "n" || event.key === "N") {
      setNoteMode(!state.noteMode);
    }
    if (event.key === "z" && (event.metaKey || event.ctrlKey)) {
      if (event.shiftKey) {
        redo();
      } else {
        undo();
      }
    }
  });

  window.addEventListener("beforeunload", () => {
    saveProgress();
    if (celebrationFx) {
      celebrationFx.destroy();
      celebrationFx = null;
    }
  });
}

function checkCompletion() {
  if (!state.puzzle || !state.puzzle.solution) return;
  if (state.completed) return;
  const gridStr = gridToString(state.grid);
  if (gridStr === state.puzzle.solution) {
    state.completed = true;
    setPaused(true);
    saveProgress();
    playCompletionPresentation();
  }
}

function renderBoardInitial() {
  state.selectedIndex = null;
  renderBoardState();
  updateAttempts();
  updatePuzzleMeta();
  updateTimer();
  updateUndoRedo();
}

function setPuzzle(puzzle, progress) {
  cancelCompletionPresentation();
  state.puzzle = puzzle;
  state.givens = normalizeGivens(puzzle.givens);
  state.grid = stringToGrid(progress?.grid || puzzle.givens);
  state.notes = progress?.notes ? [...progress.notes] : emptyNotes();
  state.attempts = Number(progress?.attempts) || 0;
  state.elapsedMs = Number(progress?.elapsedMs) || 0;
  state.puzzleStartedAt = progress?.startedAt || new Date().toISOString();
  state.sessionStart = null;
  state.completed = progress?.status === "completed";
  state.history = [];
  state.historyIndex = -1;
  pushHistory();
  setPaused(state.completed);
  renderBoardInitial();
  scheduleSave();
}

async function loadSettings() {
  const settings = await dbGet("meta", "settings");
  if (settings && settings.value) {
    state.relatedHighlight = settings.value.relatedHighlight !== false;
    state.sameValueHighlight = settings.value.sameValueHighlight === true;
    setCelebrationTheme(settings.value.celebrationTheme || "default");
    state.conflictCheck = settings.value.conflictCheck !== false;
    setTheme(settings.value.theme || "light");
  } else {
    state.relatedHighlight = true;
    state.sameValueHighlight = false;
    setCelebrationTheme("default");
    setTheme("light");
  }
  setConflictCheck(state.conflictCheck);
  syncSettingsFormFromState();
}

function applyPuzzleSource(puzzle) {
  setSourceMode(puzzle.source === "generated" ? "generated" : "book");
}

async function loadCurrentPuzzle() {
  const current = await dbGet("meta", "currentPuzzleId");
  if (current && current.value) {
    const puzzle = await dbGet("puzzles", current.value);
    if (puzzle) {
      const progress = await dbGet("progress", puzzle.id);
      setPuzzle(puzzle, progress);
      applyPuzzleSource(puzzle);
      return;
    }
  }
  await startNewPuzzle("book");
}

async function startNewPuzzle(source) {
  const puzzles = await dbGetAll("puzzles");
  const progressList = await dbGetAll("progress");
  const progressMap = new Map(progressList.map((item) => [item.puzzleId, item]));

  let candidates = puzzles.filter((puzzle) => puzzle.source === source);

  if (source === "book") {
    const notCompleted = candidates.filter((puzzle) => progressMap.get(puzzle.id)?.status !== "completed");
    if (notCompleted.length > 0) {
      candidates = notCompleted;
    }
  }

  let puzzle = null;
  if (source === "generated") {
    puzzle = await generatePuzzle(state.genDifficulty);
    await dbPut("puzzles", puzzle);
  } else {
    puzzle = candidates[0];
    if (!puzzle) {
      showEmptyNotice();
      return;
    }
  }

  const progress = progressMap.get(puzzle.id);
  setPuzzle(puzzle, progress);
  applyPuzzleSource(puzzle);
}

function showEmptyNotice() {
  puzzleMetaEl.textContent = "暂无书籍题库，请导入";
  state.puzzle = null;
  state.givens = ".".repeat(81);
  state.grid = stringToGrid(".".repeat(81));
  state.notes = emptyNotes();
  state.attempts = 0;
  state.elapsedMs = 0;
  state.puzzleStartedAt = null;
  state.sessionStart = null;
  state.completed = false;
  renderBoardInitial();
  setPaused(true);
}

function createTemplate() {
  return {
    version: 1,
    exportedAt: new Date().toISOString(),
    puzzles: [
      {
        id: "book-001",
        givens: "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
        solution: "534678912672195348198342567859761423426853791713924856961537284287419635345286179",
        source: "book",
        difficulty: 1,
        createdAt: "2026-02-09",
        importedAt: "2026-02-09"
      }
    ]
  };
}

function downloadTemplate() {
  const blob = new Blob([JSON.stringify(createTemplate(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `misudoku-template-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function isValidPuzzle(puzzle) {
  if (!puzzle) return false;
  if (!puzzle.id || !puzzle.givens || !puzzle.solution) return false;
  const givens = normalizeGridString(puzzle.givens);
  const solution = normalizeGridString(puzzle.solution);
  return givens.length === 81 && solution.length === 81;
}

async function handleImport() {
  const file = importFile.files[0];
  if (!file) {
    importStatus.textContent = "请选择文件。";
    return;
  }
  importStatus.textContent = "导入中...";

  const text = await file.text();
  try {
    if (file.name.endsWith(".csv") || file.type === "text/csv") {
      const puzzles = parseCsv(text);
      const result = await applyImport({ puzzles }, replaceMode.checked);
      importStatus.textContent = `导入完成：${result.puzzles} 题。`;
    } else {
      const data = JSON.parse(text);
      const result = await applyImport(data, replaceMode.checked);
      const pieces = [];
      if (result.puzzles) pieces.push(`${result.puzzles} 题`);
      if (result.progress) pieces.push(`${result.progress} 进度`);
      importStatus.textContent = `导入完成：${pieces.join("，") || "无有效数据"}`;
    }
    importFile.value = "";
    await loadCurrentPuzzle();
    if (state.sourceMode === "book") {
      await refreshBookPanel();
    }
  } catch (error) {
    importStatus.textContent = `导入失败：${error.message}`;
  }
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) throw new Error("CSV 为空");
  const headers = lines[0].split(",").map((item) => item.trim());
  const required = ["id", "givens", "solution"];
  required.forEach((key) => {
    if (!headers.includes(key)) {
      throw new Error(`CSV 缺少字段 ${key}`);
    }
  });
  const puzzles = [];
  for (let i = 1; i < lines.length; i += 1) {
    const values = lines[i].split(",").map((item) => item.trim());
    const record = {};
    headers.forEach((key, idx) => {
      record[key] = values[idx];
    });
    puzzles.push({
      id: record.id,
      givens: record.givens,
      solution: record.solution,
      source: record.source || "book",
      difficulty: Number(record.difficulty) || 0,
      createdAt: record.createdAt || new Date().toISOString().split("T")[0],
      importedAt: new Date().toISOString().split("T")[0]
    });
  }
  return puzzles;
}

async function applyImport(data, replace) {
  if (!data || (!data.puzzles && !data.progress)) {
    throw new Error("文件格式不符合模板");
  }

  if (replace) {
    await dbClear("puzzles");
    await dbClear("progress");
    await dbClear("meta");
  }

  let puzzlesImported = 0;
  let progressImported = 0;

  if (Array.isArray(data.puzzles)) {
    for (const puzzle of data.puzzles) {
      if (!isValidPuzzle(puzzle)) continue;
      const normalized = {
        id: puzzle.id,
        givens: normalizeGivens(puzzle.givens),
        solution: normalizeGridString(puzzle.solution),
        source: puzzle.source || "book",
        difficulty: Number(puzzle.difficulty) || 0,
        createdAt: puzzle.createdAt || new Date().toISOString().split("T")[0],
        importedAt: new Date().toISOString().split("T")[0]
      };
      await dbPut("puzzles", normalized);
      puzzlesImported += 1;
    }
  }

  if (Array.isArray(data.progress)) {
    for (const progress of data.progress) {
      if (!progress.puzzleId || !progress.grid) continue;
      await dbPut("progress", progress);
      progressImported += 1;
    }
  }

  if (Array.isArray(data.meta)) {
    for (const meta of data.meta) {
      if (!meta.key) continue;
      await dbPut("meta", meta);
    }
  }

  return { puzzles: puzzlesImported, progress: progressImported };
}

async function exportBackup() {
  const puzzles = await dbGetAll("puzzles");
  const progress = await dbGetAll("progress");
  const meta = await dbGetAll("meta");
  const payload = {
    version: 1,
    exportedAt: new Date().toISOString(),
    puzzles,
    progress,
    meta
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `misudoku-backup-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function seedSamplePuzzles() {
  const puzzles = await dbGetAll("puzzles");
  if (puzzles.length > 0) return;
  for (const puzzle of SAMPLE_PUZZLES) {
    await dbPut("puzzles", puzzle);
  }
}

async function init() {
  buildBoard();
  setupEvents();
  setNoteMode(false);
  await seedSamplePuzzles();
  await loadSettings();
  await loadCurrentPuzzle();
  registerServiceWorker();
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("sw.js");
}

function createEmptyGrid() {
  return Array.from({ length: 81 }, () => ".");
}

function isValidPlacement(grid, index, value) {
  const row = Math.floor(index / 9);
  const col = index % 9;
  for (let c = 0; c < 9; c += 1) {
    if (grid[row * 9 + c] === value) return false;
  }
  for (let r = 0; r < 9; r += 1) {
    if (grid[r * 9 + col] === value) return false;
  }
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      if (grid[(boxRow + r) * 9 + (boxCol + c)] === value) return false;
    }
  }
  return true;
}

function solveGrid(grid) {
  const emptyIndex = grid.indexOf(".");
  if (emptyIndex === -1) return true;
  const candidates = shuffle(["1", "2", "3", "4", "5", "6", "7", "8", "9"]);
  for (const value of candidates) {
    if (isValidPlacement(grid, emptyIndex, value)) {
      grid[emptyIndex] = value;
      if (solveGrid(grid)) return true;
      grid[emptyIndex] = ".";
    }
  }
  return false;
}

function countSolutions(grid, limit = 2) {
  const emptyIndex = grid.indexOf(".");
  if (emptyIndex === -1) return 1;
  let count = 0;
  for (let value = 1; value <= 9; value += 1) {
    const val = String(value);
    if (isValidPlacement(grid, emptyIndex, val)) {
      grid[emptyIndex] = val;
      count += countSolutions(grid, limit - count);
      grid[emptyIndex] = ".";
      if (count >= limit) return count;
    }
  }
  return count;
}

function shuffle(array) {
  const list = [...array];
  for (let i = list.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

async function generatePuzzle(difficulty) {
  const grid = createEmptyGrid();
  solveGrid(grid);
  const solution = grid.join("");
  const givensTarget = difficulty === 1 ? 38 : difficulty === 2 ? 33 : difficulty === 3 ? 28 : 24;
  const indices = shuffle(Array.from({ length: 81 }, (_, i) => i));
  let givensCount = 81;

  for (const index of indices) {
    if (givensCount <= givensTarget) break;
    const backup = grid[index];
    grid[index] = ".";
    const gridCopy = [...grid];
    const solutions = countSolutions(gridCopy, 2);
    if (solutions !== 1) {
      grid[index] = backup;
    } else {
      givensCount -= 1;
    }
  }

  const givens = grid.join("");
  return {
    id: `generated-${Date.now()}`,
    givens,
    solution,
    source: "generated",
    difficulty,
    createdAt: new Date().toISOString().split("T")[0],
    importedAt: new Date().toISOString().split("T")[0]
  };
}

init();
