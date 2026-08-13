"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import BrowserArenaWorker from "../workers/browser-arena.worker.js?worker";
import { CHAMPION_BASELINE } from "../lib/champion-baseline.js";

type Mode = "watch" | "play" | "selfplay";
type RuntimeKind = "browser" | "python";
type Language = "zh" | "en";
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
  bullets: { x: number; y: number; owner: number; lifetime: number }[];
  runtime: {
    step_ms: number;
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
};
type LeagueSummary = {
  games: number;
  win: number;
  loss: number;
  double_death: number;
  draw: number;
  winRate: number;
  colorGap: number;
  p95DecisionMs: number;
};
type LeagueReport = {
  generatedAt: string;
  candidate: string;
  roundsPerSide: number;
  overall: LeagueSummary;
  opponents: Record<string, LeagueSummary>;
};

const API = "http://127.0.0.1:8766/api";
const LANGUAGE_STORAGE_KEY = "tank-trouble-language";
const LANGUAGE_EVENT = "tank-trouble-language-change";
const PYTHON_RESEARCH_ENABLED =
  process.env.NEXT_PUBLIC_ENABLE_PYTHON_RESEARCH === "1";
const EMPTY: Controls = {
  forward: false,
  backup: false,
  turn_left: false,
  turn_right: false,
  fire: false,
};

function languageSnapshot(): Language {
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  if (saved === "zh" || saved === "en") return saved;
  return window.navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

function subscribeLanguage(callback: () => void) {
  window.addEventListener(LANGUAGE_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(LANGUAGE_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

async function command(payload: Record<string, unknown>) {
  const response = await fetch(`${API}/command`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`command failed: ${response.status}`);
  return response.json();
}

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

function isTacticalPolicy(policy: string | undefined) {
  return Boolean(policy?.startsWith("p27-js-tactical"));
}

function paletteFor(state: ArenaState, index: number) {
  const policies = [state.left_policy, state.right_policy];
  const policy = policies[index];
  const other = policies[1 - index];
  if (isTacticalPolicy(policy) && !isTacticalPolicy(other)) return RED_PALETTE;
  if (policy === "laika-js" && other !== "laika-js") return BLACK_PALETTE;
  return index === 0 ? RED_PALETTE : BLACK_PALETTE;
}

function MazeCanvas({ state, language }: { state: ArenaState | null; language: Language }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state) return;
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const pad = Math.max(22, Math.min(rect.width, rect.height) * 0.055);
    const unit = Math.min(
      (rect.width - pad * 2) / state.world_width,
      (rect.height - pad * 2) / state.world_height,
    );
    const worldW = state.world_width * unit;
    const worldH = state.world_height * unit;
    const left = (rect.width - worldW) / 2;
    const top = (rect.height - worldH) / 2;

    ctx.fillStyle = "#f0efeb";
    ctx.fillRect(left, top, worldW, worldH);
    ctx.strokeStyle = "#3f444d";
    ctx.lineWidth = Math.max(4, state.wall_width * unit);
    ctx.lineCap = "square";
    for (const [x1, y1, x2, y2] of state.walls) {
      ctx.beginPath();
      ctx.moveTo(left + x1 * unit, top + y1 * unit);
      ctx.lineTo(left + x2 * unit, top + y2 * unit);
      ctx.stroke();
    }

    for (const bullet of state.bullets) {
      ctx.beginPath();
      ctx.fillStyle = paletteFor(state, bullet.owner).body;
      ctx.arc(
        left + bullet.x * unit,
        top + bullet.y * unit,
        Math.max(2, state.bullet_radius * unit),
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    if (state.tanks[0]) {
      const palette = paletteFor(state, 0);
      drawTank(ctx, state.tanks[0], unit, left, top, palette.body, palette.turret);
    }
    if (state.tanks[1]) {
      const palette = paletteFor(state, 1);
      drawTank(ctx, state.tanks[1], unit, left, top, palette.body, palette.turret);
    }
  }, [state]);

  return (
    <canvas
      ref={canvasRef}
      className="maze-canvas"
      aria-label={language === "zh" ? "Tank Trouble 实时迷宫对战" : "Live Tank Trouble maze battle"}
    />
  );
}

function percent(value = 0) {
  return `${(value * 100).toFixed(1)}%`;
}

function millis(value = 0) {
  return `${value.toFixed(value >= 10 ? 1 : 2)} ms`;
}

const OPPONENT_LABELS: Record<string, string> = {
  "laika-js": "Laika",
  "hunter-js": "Hunter",
  "dodger-js": "Dodger",
  "random-js": "Random",
};

const ENGLISH_POLICY_LABELS: Record<string, string> = {
  "p27-js-tactical-v2": "Tactical (current champion)",
  "p27-js-tactical": "Tactical Legacy (frozen baseline)",
  "killfield-js": "KillField JS (third-party speed baseline)",
  "laika-js": "Laika (official script)",
  "hunter-js": "Hunter JS (strong pursuit)",
  "dodger-js": "Dodger JS (strong evasion)",
  "random-js": "Random JS",
  "idle-js": "Idle JS",
  "p27-exact-shield": "P27b + Exact Shield (strongest, slow)",
  "p27-hybrid": "P27b Hybrid (fair experiment)",
  p27b: "P27b (smooth)",
  laika: "Laika",
  hunter: "Hunter",
  random: "Random",
};

function LeagueMonitor({
  tactical,
  killfield,
  language,
}: {
  tactical: LeagueReport | null;
  killfield: LeagueReport | null;
  language: Language;
}) {
  const zh = language === "zh";
  const opponentNames = tactical ? Object.keys(tactical.opponents) : [];
  const delta = tactical && killfield
    ? tactical.overall.winRate - killfield.overall.winRate
    : 0;
  return (
    <section className="league-panel" aria-label={zh ? "自博弈晋级监控" : "Self-play promotion monitor"}>
      <div className="league-heading">
        <div>
          <span className="eyebrow">CHAMPION GATE · SELF-PLAY LEAGUE</span>
          <h2>{zh ? "自博弈晋级监控" : "Self-play promotion monitor"}</h2>
          <p>{zh
            ? "同种子、红黑换边，只使用当前可见状态。网页负责演示与监控；离线飞轮收集困难局并训练候选，全部门槛通过后才更新冠军。"
            : "Matched seeds and swapped colors using visible state only. The site demonstrates and monitors; the offline flywheel collects hard games and evaluates candidates before any champion update."}</p>
        </div>
        <div className="champion-lock">
          <span>{zh ? "冻结冠军" : "Frozen champion"}</span>
          <strong>{CHAMPION_BASELINE.name}</strong>
          <small>{zh ? "Laika 盲测" : "Laika blind test"} {CHAMPION_BASELINE.laikaBenchmark.wins}/{CHAMPION_BASELINE.laikaBenchmark.rounds} · {percent(CHAMPION_BASELINE.laikaBenchmark.winRate)}</small>
        </div>
      </div>

      <div className="league-scoreboard">
        <div>
          <span>{zh ? "正式四对手联赛" : "Four-opponent league"}</span>
          <strong>{tactical ? `${tactical.overall.win}/${tactical.overall.games}` : (zh ? "载入中" : "Loading")}</strong>
          <small>Tactical · {tactical ? percent(tactical.overall.winRate) : "—"}</small>
        </div>
        <div>
          <span>{zh ? "第三方基线" : "Third-party baseline"}</span>
          <strong>{killfield ? `${killfield.overall.win}/${killfield.overall.games}` : (zh ? "载入中" : "Loading")}</strong>
          <small>KillField · {killfield ? percent(killfield.overall.winRate) : "—"}</small>
        </div>
        <div className={delta > 0 ? "positive" : ""}>
          <span>{zh ? "总体胜率差" : "Overall win-rate gap"}</span>
          <strong>{tactical && killfield ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp` : "—"}</strong>
          <small>{zh ? "208 局，全新种子换边评测" : "208 games, fresh seeds, both colors"}</small>
        </div>
        <div>
          <span>{zh ? "晋级门槛" : "Promotion gates"}</span>
          <strong>{zh ? "5 项" : "5 gates"}</strong>
          <small>{zh ? "Laika · 对手池 · 双死 · 换边 · p95" : "Laika · pool · double death · colors · p95"}</small>
        </div>
      </div>

      <div className="league-table-wrap">
        <table className="league-table">
          <thead>
            <tr>
              <th>{zh ? "公平对手" : "Fair opponent"}</th>
              <th>Tactical</th>
              <th>KillField</th>
              <th>{zh ? "差值" : "Delta"}</th>
              <th>{zh ? "诊断" : "Diagnosis"}</th>
            </tr>
          </thead>
          <tbody>
            {opponentNames.map((opponent) => {
              const current = tactical?.opponents[opponent];
              const baseline = killfield?.opponents[opponent];
              const difference = current && baseline ? current.winRate - baseline.winRate : 0;
              const diagnosis = !current
                ? (zh ? "等待数据" : "Waiting for data")
                : current.draw > 0
                  ? (zh ? `${current.draw} 超时 · 优先改进终局能力` : `${current.draw} timeouts · improve endgame`)
                  : current.colorGap > CHAMPION_BASELINE.promotionGate.maximumColorGap
                    ? (zh ? `换边差 ${percent(current.colorGap)} · 扩大样本` : `Color gap ${percent(current.colorGap)} · expand sample`)
                    : (zh ? "稳定" : "Stable");
              return (
                <tr key={opponent}>
                  <th>{OPPONENT_LABELS[opponent] ?? opponent}</th>
                  <td>{current ? `${current.win}/${current.games} · ${percent(current.winRate)}` : "—"}</td>
                  <td>{baseline ? `${baseline.win}/${baseline.games} · ${percent(baseline.winRate)}` : "—"}</td>
                  <td className={difference > 0 ? "positive-text" : difference < 0 ? "negative-text" : ""}>
                    {current && baseline ? `${difference >= 0 ? "+" : ""}${(difference * 100).toFixed(1)} pp` : "—"}
                  </td>
                  <td>{diagnosis}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="route-strip" aria-label={zh ? "后续技术路线" : "Technical roadmap"}>
        <span className="done">{zh ? "1 · Tactical Legacy 基线" : "1 · Tactical Legacy baseline"}</span>
        <span className="done">{zh ? "2 · 104 局换边联赛" : "2 · 104-game color-swapped league"}</span>
        <span className="done">{zh ? "3 · Dodger 反规避" : "3 · Anti-evasion vs Dodger"}</span>
        <span className="done">{zh ? "4 · 300 局盲测晋级" : "4 · 300-game blind promotion"}</span>
        <span className="done">{zh ? "5 · 双亡安全审计" : "5 · Double-death safety audit"}</span>
        <span className="active">{zh ? "6 · 夜间困难局飞轮" : "6 · Nightly hard-game flywheel"}</span>
      </div>
    </section>
  );
}

export function Arena() {
  const language = useSyncExternalStore(subscribeLanguage, languageSnapshot, () => "zh");
  const [state, setState] = useState<ArenaState | null>(null);
  const [runtime, setRuntime] = useState<RuntimeKind>("browser");
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<Mode>("watch");
  const [leftPolicy, setLeftPolicy] = useState("p27-js-tactical-v2");
  const [rightPolicy, setRightPolicy] = useState("laika-js");
  const [seed, setSeed] = useState("970000");
  const [muted, setMuted] = useState(true);
  const [tacticalLeague, setTacticalLeague] = useState<LeagueReport | null>(null);
  const [killfieldLeague, setKillfieldLeague] = useState<LeagueReport | null>(null);
  const arenaRef = useRef<HTMLDivElement>(null);
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

  const changeLanguage = (next: Language) => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
    window.dispatchEvent(new Event(LANGUAGE_EVENT));
  };

  useEffect(() => {
    seedRef.current = seed;
  }, [seed]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch("/league-tactical-latest.json", { cache: "no-store" }).then((response) => response.json()),
      fetch("/league-killfield-latest.json", { cache: "no-store" }).then((response) => response.json()),
    ]).then(([tactical, killfield]) => {
      if (cancelled) return;
      setTacticalLeague(tactical as LeagueReport);
      setKillfieldLeague(killfield as LeagueReport);
    }).catch(() => {
      if (!cancelled) {
        setTacticalLeague(null);
        setKillfieldLeague(null);
      }
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (runtime === "browser") {
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
          setState(message.state);
          setConnected(true);
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
          setState(message.state);
          setConnected(true);
          pending.resolve(message.state);
        } else {
          pending.reject(new Error(message.message ?? "browser worker command failed"));
        }
      };
      browserWorker.onerror = () => setConnected(false);
      browserWorker.postMessage({ type: "init", seed: Number(seedRef.current) || 970000 });
      return () => {
        browserWorker.postMessage({ type: "dispose" });
        browserWorker.terminate();
        browserWorkerRef.current = null;
        for (const pending of workerRequests.values()) {
          pending.reject(new Error("browser arena worker stopped"));
        }
        workerRequests.clear();
      };
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const response = await fetch(`${API}/state`, { cache: "no-store" });
        if (!response.ok) throw new Error("offline");
        const next = (await response.json()) as ArenaState;
        if (!cancelled) {
          setState(next);
          setConnected(true);
          setMode(next.mode);
        }
      } catch {
        if (!cancelled) setConnected(false);
      } finally {
        if (!cancelled) timer = setTimeout(poll, 40);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runtime]);

  const dispatch = useCallback(async (payload: Record<string, unknown>) => {
    if (runtime === "browser") {
      const browserWorker = browserWorkerRef.current;
      if (!browserWorker) throw new Error("browser arena worker is not ready");
      const id = workerRequestIdRef.current + 1;
      workerRequestIdRef.current = id;
      return new Promise<ArenaState>((resolve, reject) => {
        workerRequestsRef.current.set(id, { resolve, reject });
        browserWorker.postMessage({ type: "command", id, payload });
      });
    }
    try {
      const next = await command(payload) as ArenaState;
      setConnected(true);
      return next;
    } catch (error) {
      setConnected(false);
      throw error;
    }
  }, [runtime]);

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

  const policies = state?.available_policies ?? [
    { value: "p27-exact-shield", label: "P27b + Exact Shield（最强·慢）" },
    { value: "p27-hybrid", label: "P27b Hybrid（公平实验）" },
    { value: "p27b", label: "P27b（流畅）" },
    { value: "laika", label: "Laika" },
    { value: "hunter", label: "Hunter" },
    { value: "random", label: "Random" },
  ];
  const policyLabel = (policy: { value: string; label: string }) => (
    zh ? policy.label : (ENGLISH_POLICY_LABELS[policy.value] ?? policy.label)
  );
  const leftDisplayLabel = !zh && (state?.left_policy === "human" || mode === "play")
    ? "You"
    : state?.left_label ?? (runtime === "browser" ? "Tank Trouble Tactical" : (zh ? "P27b（流畅）" : "P27b (smooth)"));
  const leftPalette = state ? paletteFor(state, 0) : RED_PALETTE;
  const rightPalette = state ? paletteFor(state, 1) : BLACK_PALETTE;

  const changeMode = async (next: Mode) => {
    setMode(next);
    const browser = runtime === "browser";
    const left = next === "watch"
      ? (browser ? "p27-js-tactical-v2" : "p27b")
      : next === "play"
        ? "human"
        : (leftPolicy === "human" || leftPolicy.endsWith("laika") || leftPolicy.endsWith("laika-js"))
          ? (browser ? "p27-js-tactical-v2" : "p27b")
          : leftPolicy;
    const right = next === "watch"
      ? (browser ? "laika-js" : "laika")
      : next === "play"
        ? (browser ? "p27-js-tactical-v2" : "laika")
        : rightPolicy;
    setLeftPolicy(left);
    setRightPolicy(right);
    await dispatch({ action: "mode", mode: next, left_policy: left, right_policy: right });
    if (next !== "play") {
      keysRef.current.clear();
      controlsRef.current = { ...EMPTY };
      await dispatch({ action: "input", controls: EMPTY });
    }
  };

  const updateMatch = async (left: string, right: string) => {
    setLeftPolicy(left);
    setRightPolicy(right);
    await dispatch({ action: "mode", mode, left_policy: left, right_policy: right });
  };

  const changeRuntime = (next: RuntimeKind) => {
    if (next === runtime) return;
    keysRef.current.clear();
    controlsRef.current = { ...EMPTY };
    setState(null);
    setConnected(false);
    setRuntime(next);
    setMode("watch");
    setLeftPolicy(next === "browser" ? "p27-js-tactical-v2" : "p27b");
    setRightPolicy(next === "browser" ? "laika-js" : "laika");
  };

  const statusCopy = useMemo(() => {
    if (!connected) {
      return runtime === "browser"
        ? (zh ? "正在启动浏览器对战 Worker" : "Starting browser battle worker")
        : (zh ? "等待本地对战服务" : "Waiting for the local battle service");
    }
    if (state?.paused) return zh ? "已暂停" : "Paused";
    if (state?.frozen) return zh ? "本回合结算中" : "Settling this round";
    if (mode === "play") return zh ? "WASD / 方向键移动 · Q / 空格开火" : "WASD / arrow keys to move · Q / Space to fire";
    if (mode === "selfplay") return zh ? "双方 AI 正在实时自博弈" : "Both AIs are playing live";
    if (runtime === "browser") {
      return zh
        ? "纯浏览器模式：H36 搜索 + 按需两段式安全验证，无 Python 往返"
        : "Browser-native H36 search with on-demand two-stage safety checks; no Python round trips";
    }
    if (state?.left_policy === "p27-exact-shield") {
      return zh ? "最强慢速模式：逐帧精确状态安全审计" : "Strongest slow mode: frame-exact safety audit";
    }
    if (state?.left_policy === "p27-hybrid") {
      return zh ? "实验模式：P27b 网络先验 + 公平采样搜索" : "Experimental: P27b neural prior with fair sampled search";
    }
    return zh ? "P27b Frozen：每个 25 Hz 物理帧都重新决策" : "P27b Frozen: a new decision on every 25 Hz physics frame";
  }, [connected, mode, runtime, state?.frozen, state?.left_policy, state?.paused, zh]);

  return (
    <main className="site-shell">
      <header className="masthead">
        <div>
          <h1>tank trouble ai</h1>
          <p className="intro">{zh
            ? "浏览器里的迷宫坦克对战。观察搜索型 AI 挑战 Laika，亲自上场，或让两个控制器持续自博弈。"
            : "Maze tank battles in your browser. Watch a search-based AI challenge Laika, play it yourself, or run two controllers head to head."}</p>
        </div>
        <div className="header-actions">
          <span className="browser-badge">JS · 25 FPS</span>
          <div className="language-switch" role="group" aria-label={zh ? "语言切换" : "Language switcher"}>
            <button type="button" className={zh ? "active" : ""} aria-pressed={zh} onClick={() => changeLanguage("zh")}>中</button>
            <button type="button" className={!zh ? "active" : ""} aria-pressed={!zh} onClick={() => changeLanguage("en")}>EN</button>
          </div>
          <a className="repo-link" href="https://github.com/Mingjie-Mao/tank-trouble" target="_blank" rel="noreferrer">
            GitHub <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <section className="arena-card" ref={arenaRef}>
        <div className="score-row">
          <div className="combatant">
            <span className="color-chip" style={{ background: leftPalette.body }} />
            <strong>{leftDisplayLabel}</strong>
            <span className="score">{state?.scores?.[0] ?? 0}</span>
          </div>
          <span className="versus">VS</span>
          <div className="combatant right">
            <span className="score">{state?.scores?.[1] ?? 0}</span>
            <strong>{state?.right_label ?? "Laika"}</strong>
            <span className="color-chip" style={{ background: rightPalette.body }} />
          </div>
        </div>

        <div className="round-row">
          <div>
            <strong>{zh ? `第 ${state?.round ?? 1} 回合` : `Round ${state?.round ?? 1}`}</strong>
            <span>{statusCopy}</span>
          </div>
          <div className={`live-pill ${connected ? "online" : "offline"}`}>
            <span />{connected
              ? `${zh ? "实时" : "LIVE"} · ${(state?.effective_fps ?? 0).toFixed(1)} FPS`
              : runtime === "browser" ? "JS WORKER STARTING" : "PYTHON BACKEND OFFLINE"}
          </div>
        </div>

        <div className="stage-wrap">
          <MazeCanvas state={state} language={language} />
          {!connected && runtime === "python" && (
            <div className="connection-card" role="status">
              <strong>{zh ? "对战服务还没有连接" : "Battle service is not connected"}</strong>
              <span>{zh ? "在项目终端运行 ./run_web_arena.sh" : "Run ./run_web_arena.sh in the project terminal"}</span>
            </div>
          )}
          <div className="stage-actions">
            <button type="button" aria-label={muted ? (zh ? "开启声音" : "Unmute") : (zh ? "静音" : "Mute")} onClick={() => setMuted(!muted)}>
              {muted ? (zh ? "音×" : "S×") : (zh ? "音" : "S")}
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
        <div className="runtime-switch" role="group" aria-label={zh ? "运行时选择" : "Runtime selection"}>
          <span>{zh ? "运行时" : "Runtime"}</span>
          <button className={runtime === "browser" ? "active" : ""} onClick={() => changeRuntime("browser")}>Browser JS</button>
          {PYTHON_RESEARCH_ENABLED && (
            <button className={runtime === "python" ? "active" : ""} onClick={() => changeRuntime("python")}>Python Research</button>
          )}
          <small>{runtime === "browser" ? (zh ? "物理、Laika、搜索在独立 Worker" : "Physics, Laika and search in a dedicated Worker") : "P27b / Hybrid / Exact Shield"}</small>
        </div>
        <div className="mode-tabs">
          <button className={mode === "watch" ? "active" : ""} onClick={() => changeMode("watch")}>{zh ? "看它打" : "Watch"}</button>
          <button className={mode === "play" ? "active" : ""} onClick={() => changeMode("play")}>{zh ? "和它打" : "Play"}</button>
          <button className={mode === "selfplay" ? "active" : ""} onClick={() => changeMode("selfplay")}>AI vs AI</button>
        </div>

        <div className="setting-row">
          <button className="secondary" onClick={() => dispatch({ action: "new_maze", seed: seed || "random" })}>{zh ? "换一张迷宫" : "New maze"} <kbd>R</kbd></button>
          <button className="secondary" onClick={() => dispatch({ action: "reset_score" })}>{zh ? "清零比分" : "Reset score"}</button>
          <label>
            <span>{zh ? "种子" : "Seed"}</span>
            <input value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="random" />
          </label>
          {mode === "selfplay" && (
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
          {mode === "watch" && (
            <label className={runtime === "browser" ? "" : ""}>
              <span>{zh ? "观战 AI" : "Watch AI"}</span>
              <select
                value={leftPolicy}
                onChange={(event) => updateMatch(event.target.value, "laika")}
              >
                {policies
                  .filter((policy) => policy.value !== "laika" && policy.value !== "laika-js")
                  .map((policy) => <option key={policy.value} value={policy.value}>{policyLabel(policy)}</option>)}
              </select>
            </label>
          )}
        </div>
      </section>

      <section className="telemetry" aria-label={zh ? "实时性能" : "Live performance"}>
        <div className="metric lead">
          <span>{state?.left_policy === "p27b" ? (zh ? "在线搜索" : "Online search") : (zh ? "搜索率" : "Search rate")}</span>
          <strong>{percent(state?.telemetry.search_rate)}</strong>
          <small>{state?.left_policy === "p27b" ? (zh ? "纯网络，不做 rollout" : "Pure network, no rollout") : (zh ? "搜索帧占比" : "Share of searched frames")}</small>
        </div>
        <div className="metric">
          <span>{zh ? "决策 p95" : "Decision p95"}</span>
          <strong>{millis(state?.telemetry.p95_ms)}</strong>
          <small>{runtime === "browser" ? (zh ? "JS 搜索控制器" : "JS search controller") : state?.left_policy === "p27b" ? (zh ? "P27b 网络决策" : "P27b network decision") : (zh ? "网络 + 安全审计" : "Network + safety audit")}</small>
        </div>
        <div className="metric">
          <span>{runtime === "browser" || state?.left_policy === "p27b" ? (zh ? "物理帧率" : "Physics rate") : (zh ? "危险接管率" : "Safety takeover rate")}</span>
          <strong>{runtime === "browser" || state?.left_policy === "p27b"
            ? `${(state?.effective_fps ?? 0).toFixed(1)} FPS`
            : percent(state?.telemetry.change_rate)}</strong>
          <small>{runtime === "browser" || state?.left_policy === "p27b"
            ? (zh ? "目标 25 FPS" : "25 FPS target")
            : (zh ? `${state?.telemetry.search_changes ?? 0} 次不安全审计` : `${state?.telemetry.search_changes ?? 0} unsafe audits`)}</small>
        </div>
        <div className="metric">
          <span>{zh ? "当前决策" : "Current decision"}</span>
          <strong>{millis(state?.runtime.left_decision_ms)}</strong>
          <small>{state?.telemetry.last_reason ?? "network"}</small>
        </div>
        <div className="metric">
          <span>{zh ? "连续获胜" : "Win streak"}</span>
          <strong>{state?.streak ?? 0}</strong>
          <small>{zh ? "最佳" : "Best"} {state?.best_streak ?? 0}</small>
        </div>
      </section>

      <LeagueMonitor tactical={tacticalLeague} killfield={killfieldLeague} language={language} />

      <footer>
        <span>{runtime === "browser"
          ? (zh ? "浏览器 JavaScript 物理、Laika 与搜索 · KillField MIT 对照基线" : "Browser JavaScript physics, Laika and search · KillField MIT comparison baseline")
          : (zh ? "JavaScript/Canvas 渲染 · Python 1:1 物理 · P27b Frozen" : "JavaScript/Canvas rendering · Python 1:1 physics · P27b Frozen")}</span>
        <span>seed {state?.seed ?? seed} · <a href="https://github.com/Mingjie-Mao/tank-trouble" target="_blank" rel="noreferrer">Mingjie-Mao/tank-trouble</a></span>
      </footer>
    </main>
  );
}
