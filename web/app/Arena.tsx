"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type Language, setLanguage, useLanguage } from "./language";
import BrowserArenaWorker from "../workers/browser-arena.worker.js?worker";
import SearchHelperWorker from "../workers/search-helper.worker.js?worker";
import arenaWasmUrl from "../lib/wasm/tank_bg.wasm?url";

type Mode = "watch" | "play" | "selfplay";
type Controls = {
  forward: boolean;
  backup: boolean;
  turn_left: boolean;
  turn_right: boolean;
  fire: boolean;
};
type Tank = {
  index: number;
  x: number;
  y: number;
  rotation: number;
  display_scale: number;
  alive: boolean;
  bullets_fired: number;
};
type ArenaState = {
  connected: boolean;
  mode: Mode;
  paused: boolean;
  seed: number;
  frame: number;
  round: number;
  fps: number;
  effective_fps: number;
  frozen: boolean;
  world_width: number;
  world_height: number;
  wall_width: number;
  bullet_radius: number;
  walls: number[][];
  scores: number[];
  streak: number;
  best_streak: number;
  left_policy: string;
  right_policy: string;
  left_label: string;
  right_label: string;
  tanks: Tank[];
  bullets: { id: string; x: number; y: number; owner: number; lifetime: number }[];
  runtime: {
    step_ms: number;
    search_workers?: number;
    left_decision_ms: number;
    right_decision_ms: number;
  };
  telemetry: {
    frames: number;
    searches: number;
    search_rate: number;
    search_changes: number;
    change_rate: number;
    deadline_hits: number;
    candidates_evaluated?: number;
    last_decision_ms: number;
    last_search_ms?: number;
    p50_ms: number;
    p95_ms: number;
    last_reason: string;
  };
  available_policies: { value: string; label: string }[];
  vision?: Vision | null;
};
type Vision = {
  dodge: boolean[];
  shot: { points: number[][]; hit: "opponent" | "self" | "none" };
  threats: string[];
  actions?: [number, number, number, number][];
  shieldVetoed?: boolean;
};
const EMPTY: Controls = {
  forward: false,
  backup: false,
  turn_left: false,
  turn_right: false,
  fire: false,
};


function drawTank(
  ctx: CanvasRenderingContext2D,
  tank: Tank,
  unit: number,
  left: number,
  top: number,
  body: string,
  turret: string,
) {
  if (!tank.alive) return;
  const x = left + tank.x * unit;
  const y = top + tank.y * unit;
  // Match the SWF/Python renderer exactly. display_scale converts the
  // original 61×81 local tank artwork into world pixels; unit then converts
  // world pixels into CSS pixels.  The previous code treated display_scale
  // itself as a size and made the tank roughly 3–4× too small.
  const scale = tank.display_scale * unit;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate((tank.rotation * Math.PI) / 180);
  ctx.scale(scale, scale);
  ctx.lineWidth = 1.5 / Math.max(scale, 0.01);
  ctx.strokeStyle = "#15181d";
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.rect(-30.5, -40.5, 61, 81);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#181b20";
  ctx.beginPath();
  ctx.moveTo(-30.5, -40.5);
  ctx.lineTo(-18.91, -40.5);
  ctx.lineTo(-18.91, 40.5);
  ctx.lineTo(-30.5, 40.5);
  ctx.closePath();
  ctx.moveTo(30.5, -40.5);
  ctx.lineTo(18.91, -40.5);
  ctx.lineTo(18.91, 40.5);
  ctx.lineTo(30.5, 40.5);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = turret;
  ctx.fillRect(-8.5, -55, 17, 55);
  ctx.strokeRect(-8.5, -55, 17, 55);
  ctx.beginPath();
  ctx.arc(0, 0, 23.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

const RED_PALETTE = { body: "#c53f36", turret: "#e05a50" };
const BLACK_PALETTE = { body: "#25292f", turret: "#444a53" };

function paletteFor(state: ArenaState, index: number) {
  const policies = [state.left_policy, state.right_policy];
  const policy = policies[index];
  const other = policies[1 - index];
  if (policy === "laika-js" && other !== "laika-js") return BLACK_PALETTE;
  return index === 0 ? RED_PALETTE : BLACK_PALETTE;
}

/** The last two physics frames from the worker, and when the newer arrived. */
type Frames = { prev: ArenaState | null; curr: ArenaState | null; at: number };

const PHYSICS_FRAME_MS = 1000 / 25;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/** Interpolate headings the short way round. */
function lerpAngle(a: number, b: number, t: number) {
  const delta = ((((b - a) % 360) + 540) % 360) - 180;
  return a + delta * t;
}

/** Same maze and round: safe to interpolate between the two frames. */
function continuous(prev: ArenaState, curr: ArenaState) {
  return prev.seed === curr.seed && prev.round === curr.round
    && prev.walls.length === curr.walls.length;
}

const THROTTLE_ZH = ["后退", "", "前进"];
const THROTTLE_EN = ["back", "", "forward"];
const TURN_ZH = ["左转", "", "右转"];
const TURN_EN = ["left", "", "right"];

function actionName([throttle, turn, fire]: number[], zh: boolean) {
  const parts = [(zh ? THROTTLE_ZH : THROTTLE_EN)[throttle], (zh ? TURN_ZH : TURN_EN)[turn]]
    .filter(Boolean);
  if (fire) parts.push(zh ? "开火" : "fire");
  return parts.length ? parts.join(zh ? " · " : " + ") : (zh ? "不动" : "hold");
}

/** The AI view: shot path, threats, nine-move safety grid and the policy's choice. */
function drawVision(
  ctx: CanvasRenderingContext2D, vision: Vision, state: ArenaState, tankX: number, tankY: number,
  unit: number, left: number, top: number, zh: boolean, height: number,
) {
  const sx = (x: number) => left + x * unit;
  const sy = (y: number) => top + y * unit;

  const { points, hit } = vision.shot;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = hit === "opponent" ? "#2e8f75" : hit === "self" ? "#d0453a" : "rgba(80, 86, 96, .55)";
  if (hit === "none") ctx.setLineDash([6, 5]);
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
  ctx.stroke();
  ctx.restore();

  const threatened = new Set(vision.threats);
  ctx.save();
  ctx.strokeStyle = "#d0453a";
  ctx.lineWidth = 2;
  for (const bullet of state.bullets) {
    if (!threatened.has(bullet.id)) continue;
    ctx.beginPath();
    ctx.arc(sx(bullet.x), sy(bullet.y), Math.max(7, state.bullet_radius * unit * 3), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();

  // 3 x 3 grid above the tank: rows forward / none / back, columns left / none / right.
  const cell = 9;
  const gx = sx(tankX) - cell * 1.5;
  const gy = sy(tankY) - state.tanks[0].display_scale * unit * 62 - cell * 3;
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, .85)";
  ctx.fillRect(gx - 2, gy - 2, cell * 3 + 4, cell * 3 + 4);
  for (let throttle = 0; throttle < 3; throttle++) {
    for (let turn = 0; turn < 3; turn++) {
      ctx.fillStyle = vision.dodge[3 * throttle + turn] ? "#2e8f75" : "#d0453a";
      ctx.fillRect(gx + turn * cell + 1, gy + (2 - throttle) * cell + 1, cell - 2, cell - 2);
    }
  }
  ctx.restore();

  if (!vision.actions) return;
  const lines = vision.actions;
  const w = 210;
  const h = 30 + lines.length * 22 + (vision.shieldVetoed ? 22 : 0);
  const px = 14;
  const py = height - h - 14;
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, .92)";
  ctx.strokeStyle = "#e1e4e8";
  ctx.fillRect(px, py, w, h);
  ctx.strokeRect(px, py, w, h);
  ctx.fillStyle = "#15181d";
  ctx.font = "600 12px system-ui, sans-serif";
  ctx.fillText(zh ? "网络的选择" : "Network's choice", px + 10, py + 19);
  ctx.font = "12px system-ui, sans-serif";
  lines.forEach((action, i) => {
    const y = py + 38 + i * 22;
    ctx.fillStyle = "#e9ecef";
    ctx.fillRect(px + 10, y - 9, 80, 10);
    ctx.fillStyle = i === 0 ? "#3f7f86" : "#9aa7ab";
    ctx.fillRect(px + 10, y - 9, 80 * action[3], 10);
    ctx.fillStyle = "#15181d";
    ctx.fillText(`${Math.round(action[3] * 100)}%  ${actionName(action, zh)}`, px + 98, y);
  });
  if (vision.shieldVetoed) {
    ctx.fillStyle = "#d0453a";
    ctx.fillText(zh ? "护盾否决了一个危险动作" : "Shield vetoed a deadly action", px + 10, py + h - 10);
  }
  ctx.restore();
}

/**
 * Physics runs at the original 25 Hz; drawing runs at the display's refresh
 * rate, one physics frame behind, interpolating between the last two frames.
 */
function MazeCanvas({ frames, language }: { frames: { current: Frames }; language: Language }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let raf = 0;
    let size = "";
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const { prev, curr, at } = frames.current;
      if (!canvas || !curr) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const key = `${rect.width}x${rect.height}@${ratio}`;
      if (key !== size) {
        size = key;
        canvas.width = Math.max(1, Math.round(rect.width * ratio));
        canvas.height = Math.max(1, Math.round(rect.height * ratio));
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const t = Math.min(1, Math.max(0, (performance.now() - at) / PHYSICS_FRAME_MS));
      const from = prev && continuous(prev, curr) ? prev : curr;

      const pad = Math.max(22, Math.min(rect.width, rect.height) * 0.055);
      const unit = Math.min(
        (rect.width - pad * 2) / curr.world_width,
        (rect.height - pad * 2) / curr.world_height,
      );
      const worldW = curr.world_width * unit;
      const worldH = curr.world_height * unit;
      const left = (rect.width - worldW) / 2;
      const top = (rect.height - worldH) / 2;

      ctx.fillStyle = "#f0efeb";
      ctx.fillRect(left, top, worldW, worldH);
      ctx.strokeStyle = "#3f444d";
      ctx.lineWidth = Math.max(4, curr.wall_width * unit);
      ctx.lineCap = "square";
      for (const [x1, y1, x2, y2] of curr.walls) {
        ctx.beginPath();
        ctx.moveTo(left + x1 * unit, top + y1 * unit);
        ctx.lineTo(left + x2 * unit, top + y2 * unit);
        ctx.stroke();
      }

      const earlier = new Map(from.bullets.map((bullet) => [bullet.id, bullet]));
      const radius = Math.max(2, curr.bullet_radius * unit);
      for (const bullet of curr.bullets) {
        const before = earlier.get(bullet.id) ?? bullet;
        ctx.beginPath();
        ctx.fillStyle = paletteFor(curr, bullet.owner).body;
        ctx.arc(
          left + lerp(before.x, bullet.x, t) * unit,
          top + lerp(before.y, bullet.y, t) * unit,
          radius, 0, Math.PI * 2,
        );
        ctx.fill();
      }
      let tank0: Tank | null = null;
      for (let i = 0; i < curr.tanks.length; i++) {
        const tank = curr.tanks[i];
        const before = from.tanks[i];
        const shown = before && before.alive && tank.alive
          ? {
            ...tank,
            x: lerp(before.x, tank.x, t),
            y: lerp(before.y, tank.y, t),
            rotation: lerpAngle(before.rotation, tank.rotation, t),
          }
          : tank;
        const palette = paletteFor(curr, i);
        drawTank(ctx, shown, unit, left, top, palette.body, palette.turret);
        if (i === 0) tank0 = shown;
      }
      if (curr.vision && tank0) {
        drawVision(ctx, curr.vision, curr, tank0.x, tank0.y, unit, left, top, language === "zh", rect.height);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [frames, language]);

  return (
    <canvas
      ref={canvasRef}
      className="maze-canvas"
      aria-label={language === "zh" ? "Tank Trouble 实时迷宫对战" : "Live Tank Trouble maze battle"}
    />
  );
}

/** A short, falling square-wave pop: a shot leaving the barrel. */
function playShot(ctx: AudioContext) {
  const t = ctx.currentTime;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "square";
  osc.frequency.setValueAtTime(880, t);
  osc.frequency.exponentialRampToValueAtTime(220, t + 0.08);
  gain.gain.setValueAtTime(0.08, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.09);
  osc.connect(gain).connect(ctx.destination);
  osc.start(t);
  osc.stop(t + 0.1);
}

/** Low-passed noise with a long decay: a tank destroyed. */
function playExplosion(ctx: AudioContext) {
  const t = ctx.currentTime;
  const length = Math.floor(ctx.sampleRate * 0.6);
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / length);
  const noise = ctx.createBufferSource();
  noise.buffer = buffer;
  const filter = ctx.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1400, t);
  filter.frequency.exponentialRampToValueAtTime(120, t + 0.55);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.5, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
  noise.connect(filter).connect(gain).connect(ctx.destination);
  noise.start(t);
}

const ENGLISH_POLICY_LABELS: Record<string, string> = {
  "ppo-league": "Our AI (network + real-time search)",
  "laika-js": "Laika (original AI)",
  "random-js": "Random",
};

export function Arena() {
  const language = useLanguage();
  const [state, setState] = useState<ArenaState | null>(null);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<Mode>("watch");
  const [leftPolicy, setLeftPolicy] = useState("ppo-league");
  const [rightPolicy, setRightPolicy] = useState("laika-js");
  const [seed, setSeed] = useState("970000");
  const arenaRef = useRef<HTMLDivElement>(null);
  // Every physics frame goes to the canvas; React re-renders only when
  // something the controls show has changed.
  const framesRef = useRef<Frames>({ prev: null, curr: null, at: 0 });
  const uiKeyRef = useRef("");
  // Sound starts muted: browsers only allow audio after a click.
  const [muted, setMuted] = useState(true);
  const mutedRef = useRef(true);
  const [vision, setVision] = useState(false);
  const audioRef = useRef<AudioContext | null>(null);
  const browserWorkerRef = useRef<Worker | null>(null);
  const workerRequestIdRef = useRef(0);
  const seedRef = useRef(seed);
  const workerRequestsRef = useRef(new Map<number, {
    resolve: (value: ArenaState) => void;
    reject: (error: Error) => void;
  }>());
  const keysRef = useRef(new Set<string>());
  const controlsRef = useRef<Controls>({ ...EMPTY });
  const zh = language === "zh";

  useEffect(() => {
    document.documentElement.lang = zh ? "zh-CN" : "en";
    document.title = zh ? "Tank Trouble AI 对战场" : "Tank Trouble AI Arena";
  }, [zh]);

  const changeLanguage = setLanguage;

  useEffect(() => {
    seedRef.current = seed;
  }, [seed]);

  const receive = useCallback((next: ArenaState, force: boolean) => {
    const frames = framesRef.current;
    const previous = frames.curr;
    framesRef.current = { prev: previous, curr: next, at: performance.now() };
    const ctx = audioRef.current;
    if (!mutedRef.current && ctx && previous && previous.round === next.round) {
      if (next.bullets.length > previous.bullets.length) playShot(ctx);
      if (next.tanks.some((tank, i) => previous.tanks[i]?.alive && !tank.alive)) playExplosion(ctx);
    }
    const key = [
      next.scores.join(":"), next.left_policy, next.right_policy, next.left_label, next.right_label,
      next.mode, next.paused, next.available_policies.length,
    ].join("|");
    if (force || key !== uiKeyRef.current) {
      uiKeyRef.current = key;
      setState(next);
    }
    setConnected(true);
  }, []);

  useEffect(() => {
    const workerRequests = workerRequestsRef.current;
    const browserWorker = new BrowserArenaWorker({ name: "tank-trouble-arena" });
    browserWorkerRef.current = browserWorker;
    browserWorker.onmessage = (event: MessageEvent) => {
      const message = event.data as {
        type: "state" | "response" | "error";
        id?: number;
        state?: ArenaState;
        message?: string;
      };
      if (message.type === "state" && message.state) {
        receive(message.state, false);
        return;
      }
      if (message.id === undefined) {
        if (message.type === "error") setConnected(false);
        return;
      }
      const pending = workerRequests.get(message.id);
      if (!pending) return;
      workerRequests.delete(message.id);
      if (message.type === "response" && message.state) {
        receive(message.state, true);
        pending.resolve(message.state);
      } else {
        pending.reject(new Error(message.message ?? "browser worker command failed"));
      }
    };
    browserWorker.onerror = () => setConnected(false);
    const wasm = new URL(arenaWasmUrl, location.href).href;
    // Helpers hold copies of the arena and share the search's rollouts; leave
    // a core for the page and one for the arena worker.
    const helperCount = Math.min(7, Math.max(0, (navigator.hardwareConcurrency || 4) - 2));
    const helpers = Array.from({ length: helperCount }, (_, i) => new SearchHelperWorker({ name: `tank-trouble-search-${i + 1}` }));
    const ports = helpers.map((helper) => {
      const channel = new MessageChannel();
      helper.postMessage({ type: "connect", port: channel.port2 }, [channel.port2]);
      return channel.port1;
    });
    browserWorker.postMessage({ type: "init", seed: Number(seedRef.current) || 970000, wasm, helpers: ports }, ports);
    return () => {
      browserWorker.postMessage({ type: "dispose" });
      browserWorker.terminate();
      helpers.forEach((helper) => helper.terminate());
      browserWorkerRef.current = null;
      for (const pending of workerRequests.values()) {
        pending.reject(new Error("browser arena worker stopped"));
      }
      workerRequests.clear();
    };
  }, [receive]);

  const dispatch = useCallback(async (payload: Record<string, unknown>) => {
    const browserWorker = browserWorkerRef.current;
    if (!browserWorker) throw new Error("browser arena worker is not ready");
    const id = workerRequestIdRef.current + 1;
    workerRequestIdRef.current = id;
    return new Promise<ArenaState>((resolve, reject) => {
      workerRequestsRef.current.set(id, { resolve, reject });
      browserWorker.postMessage({ type: "command", id, payload });
    });
  }, []);

  const sendControls = useCallback(() => {
    if (mode !== "play") return;
    const keys = keysRef.current;
    const next = {
      forward: keys.has("w") || keys.has("arrowup") || keys.has("e"),
      backup: keys.has("s") || keys.has("arrowdown"),
      turn_left: keys.has("a") || keys.has("arrowleft"),
      turn_right: keys.has("d") || keys.has("arrowright") || keys.has("f"),
      fire: keys.has(" ") || keys.has("q") || keys.has("m"),
    };
    if (JSON.stringify(next) === JSON.stringify(controlsRef.current)) return;
    controlsRef.current = next;
    dispatch({ action: "input", controls: next }).catch(() => setConnected(false));
  }, [dispatch, mode]);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (["w", "a", "s", "d", "e", "f", "q", "m", " ", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) {
        event.preventDefault();
        keysRef.current.add(key);
        sendControls();
      }
      if (key === "r") {
        event.preventDefault();
        dispatch({ action: "new_maze", seed: "random" }).catch(() => setConnected(false));
      }
    };
    const up = (event: KeyboardEvent) => {
      keysRef.current.delete(event.key.toLowerCase());
      sendControls();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [dispatch, sendControls]);

  const toggleSound = () => {
    if (!audioRef.current) audioRef.current = new AudioContext();
    void audioRef.current.resume();
    mutedRef.current = !muted;
    setMuted(!muted);
  };

  const policies = state?.available_policies ?? [
    { value: "ppo-league", label: "我们的 AI（网络 + 实时搜索）" },
  ];
  const policyLabel = (policy: { value: string; label: string }) => (
    zh ? policy.label : (ENGLISH_POLICY_LABELS[policy.value] ?? policy.label)
  );
  const leftDisplayLabel = !zh && (state?.left_policy === "human" || mode === "play")
    ? "You"
    : state?.left_label ?? "Our AI";
  const leftPalette = state ? paletteFor(state, 0) : RED_PALETTE;
  const rightPalette = state ? paletteFor(state, 1) : BLACK_PALETTE;

  // Watching an AI against Laika runs the game's own Laika, exactly as the
  // evaluation does; any other pairing is two controllers playing each other.
  const spectate = (left: string, right: string): Mode => (
    right === "laika-js" && left !== "laika-js" ? "watch" : "selfplay"
  );

  const changeMode = async (next: "watch" | "play") => {
    const left = next === "play" ? "human" : "ppo-league";
    const right = next === "play" ? "ppo-league" : "laika-js";
    const engineMode = next === "play" ? "play" : spectate(left, right);
    setMode(engineMode);
    setLeftPolicy(left);
    setRightPolicy(right);
    await dispatch({ action: "mode", mode: engineMode, left_policy: left, right_policy: right });
    // A score belongs to a pairing; a new pairing starts from 0 : 0.
    await dispatch({ action: "reset_score" });
    if (next !== "play") {
      keysRef.current.clear();
      controlsRef.current = { ...EMPTY };
      await dispatch({ action: "input", controls: EMPTY });
    }
  };

  const updateMatch = async (left: string, right: string) => {
    const engineMode = mode === "play" ? "play" : spectate(left, right);
    setMode(engineMode);
    setLeftPolicy(left);
    setRightPolicy(right);
    await dispatch({ action: "mode", mode: engineMode, left_policy: left, right_policy: right });
    await dispatch({ action: "reset_score" });
  };

  return (
    <main className="site-shell">
      <section className="control-deck top-bar" aria-label={zh ? "对战设置" : "Match setup"}>
        <div className="mode-tabs">
          <button className={mode !== "play" ? "active" : ""} onClick={() => changeMode("watch")}>{zh ? "观战" : "Watch"}</button>
          <button className={mode === "play" ? "active" : ""} onClick={() => changeMode("play")}>{zh ? "和它打" : "Play"}</button>
        </div>
        {mode === "play" ? (
          <label>
            <span>{zh ? "对手" : "Opponent"}</span>
            <select value={rightPolicy} onChange={(event) => updateMatch("human", event.target.value)}>
              {policies.map((policy) => <option key={policy.value} value={policy.value}>{policyLabel(policy)}</option>)}
            </select>
          </label>
        ) : (
          <>
            <label>
              <span>{zh ? "红方" : "Red"}</span>
              <select value={leftPolicy} onChange={(event) => updateMatch(event.target.value, rightPolicy)}>
                {policies.map((policy) => <option key={policy.value} value={policy.value}>{policyLabel(policy)}</option>)}
              </select>
            </label>
            <label>
              <span>{zh ? "黑方" : "Black"}</span>
              <select value={rightPolicy} onChange={(event) => updateMatch(leftPolicy, event.target.value)}>
                {policies.map((policy) => <option key={policy.value} value={policy.value}>{policyLabel(policy)}</option>)}
              </select>
            </label>
          </>
        )}
        <div className="score-row" aria-label={zh ? "比分" : "Score"}>
          <div className="combatant">
            <span className="color-chip" style={{ background: leftPalette.body }} />
            <strong>{leftDisplayLabel}</strong>
            <span className="score">{state?.scores?.[0] ?? 0}</span>
          </div>
          <span className="versus">:</span>
          <div className="combatant right">
            <span className="score">{state?.scores?.[1] ?? 0}</span>
            <strong>{state?.right_label ?? "Laika"}</strong>
            <span className="color-chip" style={{ background: rightPalette.body }} />
          </div>
        </div>
      </section>

      <section className="arena-card" ref={arenaRef}>
        {mode === "play" && (
          <p className="play-hint">{zh ? "WASD / 方向键移动 · Q / 空格开火" : "WASD / arrow keys to move · Q / Space to fire"}</p>
        )}
        <div className="stage-wrap">
          <MazeCanvas frames={framesRef} language={language} />
          {!connected && <div className="stage-loading">{zh ? "加载中…" : "Loading…"}</div>}
          <div className="stage-actions">
            {mode !== "play" && (
              <button
                type="button"
                className={vision ? "active" : ""}
                aria-label={zh ? "AI 视角" : "AI view"}
                title={zh ? "AI 视角" : "AI view"}
                onClick={() => {
                  setVision(!vision);
                  void dispatch({ action: "vision", enabled: !vision });
                }}
              >
                👁
              </button>
            )}
            <button type="button" aria-label={muted ? (zh ? "开启声音" : "Unmute") : (zh ? "静音" : "Mute")} onClick={toggleSound}>
              {muted ? "🔇" : "🔊"}
            </button>
            <button
              type="button"
              aria-label={state?.paused ? (zh ? "继续" : "Resume") : (zh ? "暂停" : "Pause")}
              onClick={() => dispatch({ action: "pause", paused: !state?.paused })}
            >
              {state?.paused ? "▶" : "Ⅱ"}
            </button>
            <button type="button" aria-label={zh ? "全屏" : "Fullscreen"} onClick={() => arenaRef.current?.requestFullscreen()}>
              ⛶
            </button>
          </div>
        </div>
      </section>

      <section className="control-deck" aria-label={zh ? "对战控制" : "Battle controls"}>
        <button className="secondary" onClick={() => dispatch({ action: "new_maze", seed: seed || "random" })}>{zh ? "换迷宫" : "New maze"} <kbd>R</kbd></button>
        <button className="secondary" onClick={() => dispatch({ action: "reset_score" })}>{zh ? "清零比分" : "Reset score"}</button>
        <label>
          <span>{zh ? "种子" : "Seed"}</span>
          <input value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="random" />
        </label>
        {/* A plain link: vinext's <Link> prefetch throws on Cloudflare Pages. */}
        <a className="secondary report-link" href="/report">{zh ? "方法与结果" : "Method & results"}</a>
        <div className="language-switch" role="group" aria-label={zh ? "语言切换" : "Language switcher"}>
          <button type="button" className={zh ? "active" : ""} aria-pressed={zh} onClick={() => changeLanguage("zh")}>中</button>
          <button type="button" className={!zh ? "active" : ""} aria-pressed={!zh} onClick={() => changeLanguage("en")}>EN</button>
        </div>
      </section>
    </main>
  );
}
