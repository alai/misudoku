(function (global) {
  const DEFAULTS = {
    containerEl: null,
    boardEl: null,
    durationMs: 1600,
    waveStepMs: 42,
    tileFlipMs: 260,
    emitters: [
      { x: 0.25, y: 0.42 },
      { x: 0.5, y: 0.38 },
      { x: 0.75, y: 0.42 }
    ],
    confettiRateRange: [24, 48],
    particleCountPerBurst: [8, 18],
    reducedMotion: "auto",
    loopUntilAction: false,
    palette: ["#ffd166", "#ef476f", "#06d6a0", "#118ab2", "#f78c6b", "#c77dff", "#f4a261", "#2a9d8f", "#ff7b72", "#7dd3fc"],
    showCenterMessage: false,
    messageText: "Completed",
    onStart: null,
    onComplete: null,
    onAction: null,
    onFrameStats: null
  };

  function clamp(v, min, max) {
    return Math.max(min, Math.min(max, v));
  }

  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  function randInt(min, max) {
    return Math.floor(rand(min, max + 1));
  }

  function choose(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function mergeConfig(base, next) {
    const merged = Object.assign({}, base, next || {});
    merged.confettiRateRange = next && next.confettiRateRange ? next.confettiRateRange.slice() : base.confettiRateRange.slice();
    merged.particleCountPerBurst = next && next.particleCountPerBurst ? next.particleCountPerBurst.slice() : base.particleCountPerBurst.slice();
    merged.emitters = next && next.emitters ? next.emitters.map((x) => ({ x: x.x, y: x.y })) : base.emitters.map((x) => ({ x: x.x, y: x.y }));
    merged.palette = next && next.palette ? next.palette.slice() : base.palette.slice();
    return merged;
  }

  function createCelebrationFx(options) {
    const cfgBase = mergeConfig(DEFAULTS, options || {});
    if (!cfgBase.containerEl || !cfgBase.boardEl) {
      throw new Error("createCelebrationFx requires containerEl and boardEl");
    }

    let config = cfgBase;
    let running = false;
    let rafId = 0;
    let completeTimer = 0;
    let waveLoopTimer = 0;
    let actionTriggered = false;
    let fpsLastTs = 0;
    let fpsFrames = 0;
    let fpsValue = 0;
    let fpsWindowStart = 0;
    let startTs = 0;

    const wrapper = document.createElement("div");
    wrapper.className = "celebration-layer";

    const waveLayer = document.createElement("div");
    waveLayer.className = "celebration-wave-layer";

    const canvas = document.createElement("canvas");
    canvas.className = "celebration-confetti-canvas";
    const ctx = canvas.getContext("2d", { alpha: true });

    const messageEl = document.createElement("div");
    messageEl.className = "celebration-message";
    messageEl.textContent = config.messageText;
    messageEl.tabIndex = 0;
    messageEl.setAttribute("role", "button");
    messageEl.setAttribute("aria-label", "Complete");

    wrapper.appendChild(waveLayer);
    wrapper.appendChild(canvas);
    wrapper.appendChild(messageEl);
    document.body.appendChild(wrapper);

    const particlePool = [];
    const activeParticles = [];

    function allocParticle() {
      return particlePool.pop() || {
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        w: 0,
        h: 0,
        life: 0,
        ttl: 0,
        rot: 0,
        spin: 0,
        swayAmp: 0,
        swayFreq: 0,
        swayPhase: 0,
        color: "#ffffff",
        kind: "rect"
      };
    }

    function freeParticle(p) {
      particlePool.push(p);
    }

    function resizeCanvas() {
      const dpr = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function resolveReducedMotion(reducedMotionOpt) {
      if (reducedMotionOpt === true) return true;
      if (reducedMotionOpt === false) return false;
      return !!(global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }

    function effectiveConfig() {
      const isReduced = resolveReducedMotion(config.reducedMotion);
      const out = mergeConfig(config, {});
      if (isReduced) {
        out.durationMs = Math.min(900, out.durationMs);
        out.waveStepMs = Math.max(28, Math.floor(out.waveStepMs * 0.9));
        out.tileFlipMs = Math.max(160, Math.floor(out.tileFlipMs * 0.75));
        out.confettiRateRange = [Math.max(8, Math.floor(out.confettiRateRange[0] * 0.25)), Math.max(12, Math.floor(out.confettiRateRange[1] * 0.3))];
        out.particleCountPerBurst = [Math.max(2, Math.floor(out.particleCountPerBurst[0] * 0.3)), Math.max(4, Math.floor(out.particleCountPerBurst[1] * 0.35))];
      }
      out._reduced = isReduced;
      return out;
    }

    function boardMetrics(boardEl) {
      const rect = boardEl.getBoundingClientRect();
      return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    }

    function buildWaveClone(cfg) {
      waveLayer.innerHTML = "";
      const metrics = boardMetrics(config.boardEl);
      const clone = document.createElement("div");
      clone.className = "celebration-board-clone";
      clone.style.left = metrics.left + "px";
      clone.style.top = metrics.top + "px";
      clone.style.width = metrics.width + "px";
      clone.style.height = metrics.height + "px";
      clone.style.setProperty("--fx-wave-step", cfg.waveStepMs + "ms");
      clone.style.setProperty("--fx-tile-flip", cfg.tileFlipMs + "ms");
      clone.classList.toggle("is-reduced", cfg._reduced);

      const sourceCells = config.boardEl.querySelectorAll(".cell");
      sourceCells.forEach((src, idx) => {
        const r = Math.floor(idx / 9);
        const c = idx % 9;
        const cell = document.createElement("div");
        cell.className = "celebration-wave-cell";
        if (src.classList.contains("box-right")) cell.classList.add("box-right");
        if (src.classList.contains("box-bottom")) cell.classList.add("box-bottom");
        if (src.classList.contains("is-given")) cell.classList.add("is-given");

        const valueEl = document.createElement("div");
        valueEl.className = "celebration-wave-value";
        const v = src.querySelector(".cell-value");
        valueEl.textContent = v ? v.textContent : "";

        const waveIndex = r + c;
        cell.style.animationDelay = (waveIndex * cfg.waveStepMs) + "ms";
        cell.style.animationDuration = cfg.tileFlipMs + "ms";
        cell.appendChild(valueEl);
        clone.appendChild(cell);
      });
      waveLayer.appendChild(clone);
      requestAnimationFrame(() => clone.classList.add("is-active"));

      const waveEndMs = 16 * cfg.waveStepMs + cfg.tileFlipMs + 120;
      return waveEndMs;
    }

    function clearWaveLoop() {
      if (waveLoopTimer) {
        global.clearInterval(waveLoopTimer);
        waveLoopTimer = 0;
      }
    }

    function emitterSchedule(cfg) {
      return cfg.emitters.map(function () {
        const rate = rand(cfg.confettiRateRange[0], cfg.confettiRateRange[1]);
        const intervalMs = 1000 / Math.max(1, rate);
        return rand(intervalMs * 0.6, intervalMs * 1.35);
      });
    }

    function emitBurst(cfg, x, y) {
      const count = randInt(cfg.particleCountPerBurst[0], cfg.particleCountPerBurst[1]);
      for (let i = 0; i < count; i += 1) {
        const p = allocParticle();
        const speed = rand(180, 520);
        const theta = rand(-Math.PI * 0.9, -Math.PI * 0.1);
        p.x = x;
        p.y = y;
        p.vx = Math.cos(theta) * speed;
        p.vy = Math.sin(theta) * speed - rand(40, 220);
        p.w = rand(3, 8);
        p.h = rand(6, 18);
        p.life = 0;
        p.ttl = rand(0.95, 2.2);
        p.rot = rand(0, Math.PI * 2);
        p.spin = rand(-8, 8);
        p.swayAmp = rand(0.5, 4.5);
        p.swayFreq = rand(6, 12);
        p.swayPhase = rand(0, Math.PI * 2);
        p.color = choose(cfg.palette);
        p.kind = Math.random() < 0.3 ? "strip" : "rect";
        activeParticles.push(p);
      }
    }

    function renderParticle(p, dt) {
      p.life += dt;
      p.vx *= 0.985;
      p.vy = p.vy + 1200 * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.rot += p.spin * dt;

      const alpha = clamp(1 - p.life / p.ttl, 0, 1);
      if (alpha <= 0) return false;

      const sway = Math.sin(p.life * p.swayFreq + p.swayPhase) * p.swayAmp;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(p.x + sway, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      if (p.kind === "strip") {
        ctx.fillRect(-p.w * 0.25, -p.h * 0.5, p.w * 0.5, p.h);
      } else {
        ctx.fillRect(-p.w * 0.5, -p.h * 0.5, p.w, p.h);
      }
      ctx.restore();

      return p.life < p.ttl && p.y < (window.innerHeight + 32);
    }

    function animate(ts, cfg, emitterState) {
      if (!running) return;

      if (!fpsLastTs) {
        fpsLastTs = ts;
        startTs = ts;
        fpsWindowStart = ts;
      }

      const elapsedMs = ts - startTs;
      const dt = Math.min(0.04, (ts - fpsLastTs) / 1000);
      fpsLastTs = ts;

      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      const containerRect = config.containerEl.getBoundingClientRect();

      for (let i = 0; i < emitterState.length; i += 1) {
        const e = emitterState[i];
        if (elapsedMs >= e.nextEmitMs && (cfg.loopUntilAction || elapsedMs <= cfg.durationMs)) {
          const x = containerRect.left + e.x * containerRect.width;
          const y = containerRect.top + e.y * containerRect.height;
          emitBurst(cfg, x, y);
          const nextRate = rand(cfg.confettiRateRange[0], cfg.confettiRateRange[1]);
          const nextInterval = 1000 / Math.max(1, nextRate);
          e.nextEmitMs += rand(nextInterval * 0.6, nextInterval * 1.35);
        }
      }

      for (let i = activeParticles.length - 1; i >= 0; i -= 1) {
        const p = activeParticles[i];
        if (!renderParticle(p, dt)) {
          activeParticles.splice(i, 1);
          freeParticle(p);
        }
      }

      fpsFrames += 1;
      if (ts - fpsWindowStart >= 500) {
        const dur = Math.max(16, ts - fpsWindowStart);
        fpsValue = Math.round((fpsFrames * 1000) / dur);
        fpsFrames = 0;
        fpsWindowStart = ts;
      }
      if (typeof cfg.onFrameStats === "function") {
        cfg.onFrameStats({ fps: fpsValue, particles: activeParticles.length, durationMs: cfg.durationMs });
      }

      if (!cfg.loopUntilAction && elapsedMs > cfg.durationMs + 1200 && activeParticles.length === 0) {
        stop();
        return;
      }
      rafId = requestAnimationFrame((nextTs) => animate(nextTs, cfg, emitterState));
    }

    function startMessage(cfg) {
      messageEl.textContent = cfg.messageText;
      messageEl.classList.toggle("is-visible", !!cfg.showCenterMessage);
    }

    function play() {
      stop(true);

      const cfg = effectiveConfig();
      resizeCanvas();
      wrapper.classList.add("is-active");
      startMessage(cfg);
      actionTriggered = false;

      const waveEndMs = buildWaveClone(cfg);
      clearWaveLoop();
      if (cfg.loopUntilAction) {
        waveLoopTimer = global.setInterval(() => {
          if (!running) return;
          buildWaveClone(cfg);
        }, Math.max(220, waveEndMs));
      }
      const baseSchedule = emitterSchedule(cfg);
      const emitterState = cfg.emitters.map(function (e, i) {
        return {
          x: e.x,
          y: e.y,
          nextEmitMs: rand(60, 180) + baseSchedule[i]
        };
      });

      running = true;
      fpsLastTs = 0;
      fpsFrames = 0;
      fpsValue = 0;
      fpsWindowStart = 0;

      if (typeof cfg.onStart === "function") cfg.onStart();
      rafId = requestAnimationFrame((ts) => animate(ts, cfg, emitterState));

      if (!cfg.loopUntilAction) {
        const totalMs = Math.max(cfg.durationMs, waveEndMs) + 80;
        completeTimer = global.setTimeout(() => {
          if (!running) return;
          if (typeof cfg.onComplete === "function") cfg.onComplete();
          stop();
        }, totalMs);
      }

      return Promise.resolve();
    }

    function stop(skipCallback) {
      if (completeTimer) {
        global.clearTimeout(completeTimer);
        completeTimer = 0;
      }
      clearWaveLoop();
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      running = false;
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      waveLayer.innerHTML = "";
      messageEl.classList.remove("is-visible");
      wrapper.classList.remove("is-active");

      while (activeParticles.length > 0) {
        freeParticle(activeParticles.pop());
      }

      if (!skipCallback && typeof config.onFrameStats === "function") {
        config.onFrameStats({ fps: 0, particles: 0, durationMs: effectiveConfig().durationMs });
      }
    }

    function triggerAction() {
      if (!running || actionTriggered) return;
      actionTriggered = true;
      const cb = config.onAction;
      stop(true);
      if (typeof cb === "function") cb();
    }

    function destroy() {
      stop(true);
      global.removeEventListener("resize", resizeCanvas);
      if (wrapper.parentNode) {
        wrapper.parentNode.removeChild(wrapper);
      }
    }

    function setConfig(partial) {
      config = mergeConfig(config, partial || {});
      messageEl.textContent = config.messageText;
    }

    messageEl.addEventListener("click", triggerAction);
    messageEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        triggerAction();
      }
    });

    global.addEventListener("resize", resizeCanvas);

    return {
      play,
      stop,
      destroy,
      setConfig
    };
  }

  global.createCelebrationFx = createCelebrationFx;
})(window);
