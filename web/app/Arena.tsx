"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import BrowserArenaWorker from "../workers/browser-arena.worker.js?worker";
import { CHAMPION_BASELINE } from "../lib/champion-baseline.js";

type Mode = "watch" | "play" | "selfplay";
type RuntimeKind = "browser" | "python";
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
const PYTHON_RESEARCH_ENABLED =
  process.env.NEXT_PUBLIC_ENABLE_PYTHON_RESEARCH === "1";
const EMPTY: Controls = {
  forward: false,
  backup: false,
  turn_left: false,
  turn_right: false,
  fire: false,
};

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

function MazeCanvas({ state }: { state: ArenaState | null }) {
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
      ctx.fillStyle = bullet.owner === 0 ? "#be3a31" : "#242931";
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
      drawTank(ctx, state.tanks[0], unit, left, top, "#278a91", "#3fa8ae");
    }
    if (state.tanks[1]) {
      drawTank(ctx, state.tanks[1], unit, left, top, "#cf7b36", "#e29955");
    }
  }, [state]);

  return <canvas ref={canvasRef} className="maze-canvas" aria-label="Tank Trouble 实时迷宫对战" />;
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

function LeagueMonitor({
  tactical,
  killfield,
}: {
  tactical: LeagueReport | null;
  killfield: LeagueReport | null;
}) {
  const opponentNames = tactical ? Object.keys(tactical.opponents) : [];
  const delta = tactical && killfield
    ? tactical.overall.winRate - killfield.overall.winRate
    : 0;
  return (
    <section className="league-panel" aria-label="自博弈晋级监控">
      <div className="league-heading">
        <div>
          <span className="eyebrow">CHAMPION GATE · SELF-PLAY LEAGUE</span>
          <h2>自博弈晋级监控</h2>
          <p>同种子、红黑换边，只使用当前可见状态。网页负责演示与监控；离线飞轮收集困难局并训练候选，全部门槛通过后才更新冠军。</p>
        </div>
        <div className="champion-lock">
          <span>冻结冠军</span>
          <strong>{CHAMPION_BASELINE.name}</strong>
          <small>Laika 盲测 {CHAMPION_BASELINE.laikaBenchmark.wins}/{CHAMPION_BASELINE.laikaBenchmark.rounds} · {percent(CHAMPION_BASELINE.laikaBenchmark.winRate)}</small>
        </div>
      </div>

      <div className="league-scoreboard">
        <div>
          <span>正式四对手联赛</span>
          <strong>{tactical ? `${tactical.overall.win}/${tactical.overall.games}` : "载入中"}</strong>
          <small>Tactical · {tactical ? percent(tactical.overall.winRate) : "—"}</small>
        </div>
        <div>
          <span>第三方基线</span>
          <strong>{killfield ? `${killfield.overall.win}/${killfield.overall.games}` : "载入中"}</strong>
          <small>KillField · {killfield ? percent(killfield.overall.winRate) : "—"}</small>
        </div>
        <div className={delta > 0 ? "positive" : ""}>
          <span>总体胜率差</span>
          <strong>{tactical && killfield ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp` : "—"}</strong>
          <small>208 局，全新种子换边评测</small>
        </div>
        <div>
          <span>晋级门槛</span>
          <strong>5 项</strong>
          <small>Laika · 对手池 · 双死 · 换边 · p95</small>
        </div>
      </div>

      <div className="league-table-wrap">
        <table className="league-table">
          <thead>
            <tr>
              <th>公平对手</th>
              <th>Tactical</th>
              <th>KillField</th>
              <th>差值</th>
              <th>诊断</th>
            </tr>
          </thead>
          <tbody>
            {opponentNames.map((opponent) => {
              const current = tactical?.opponents[opponent];
              const baseline = killfield?.opponents[opponent];
              const difference = current && baseline ? current.winRate - baseline.winRate : 0;
              const diagnosis = !current
                ? "等待数据"
                : current.draw > 0
                  ? `${current.draw} 超时 · 优先改进终局能力`
                  : current.colorGap > CHAMPION_BASELINE.promotionGate.maximumColorGap
                    ? `换边差 ${percent(current.colorGap)} · 扩大样本`
                    : "稳定";
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

      <div className="route-strip" aria-label="后续技术路线">
        <span className="done">1 · Tactical Legacy 基线</span>
        <span className="done">2 · 104 局换边联赛</span>
        <span className="done">3 · Dodger 反规避</span>
        <span className="done">4 · 300 局盲测晋级</span>
        <span className="done">5 · 双亡安全审计</span>
        <span className="active">6 · 夜间困难局飞轮</span>
      </div>
    </section>
  );
}

export function Arena() {
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
      return runtime === "browser" ? "正在启动浏览器对战 Worker" : "等待本地对战服务";
    }
    if (state?.paused) return "已暂停";
    if (state?.frozen) return "本回合结算中";
    if (mode === "play") return "WASD / 方向键移动 · Q / 空格开火";
    if (mode === "selfplay") return "双方 AI 正在实时自博弈";
    if (runtime === "browser") {
      return "纯浏览器模式：H36 搜索 + 按需两段式安全验证，无 Python 往返";
    }
    if (state?.left_policy === "p27-exact-shield") {
      return "最强慢速模式：逐帧精确状态安全审计";
    }
    if (state?.left_policy === "p27-hybrid") {
      return "实验模式：P27b 网络先验 + 公平采样搜索";
    }
    return "P27b Frozen：每个 25 Hz 物理帧都重新决策";
  }, [connected, mode, runtime, state?.frozen, state?.left_policy, state?.paused]);

  return (
    <main className="site-shell">
      <header className="masthead">
        <div>
          <h1>tank trouble ai</h1>
          <p className="intro">浏览器里的迷宫坦克对战。观察搜索型 AI 挑战 Laika，亲自上场，或让两个控制器持续自博弈。</p>
        </div>
        <div className="header-actions">
          <span className="browser-badge">JS · 25 FPS</span>
          <a className="repo-link" href="https://github.com/Mingjie-Mao/tank-trouble" target="_blank" rel="noreferrer">
            GitHub <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      <section className="arena-card" ref={arenaRef}>
        <div className="score-row">
          <div className="combatant">
            <span className="color-chip red" />
            <strong>{state?.left_label ?? (runtime === "browser" ? "Tank Trouble Tactical" : "P27b（流畅）")}</strong>
            <span className="score">{state?.scores?.[0] ?? 0}</span>
          </div>
          <span className="versus">VS</span>
          <div className="combatant right">
            <span className="score">{state?.scores?.[1] ?? 0}</span>
            <strong>{state?.right_label ?? "Laika"}</strong>
            <span className="color-chip graphite" />
          </div>
        </div>

        <div className="round-row">
          <div>
            <strong>第 {state?.round ?? 1} 回合</strong>
            <span>{statusCopy}</span>
          </div>
          <div className={`live-pill ${connected ? "online" : "offline"}`}>
            <span />{connected
              ? `实时 · ${(state?.effective_fps ?? 0).toFixed(1)} FPS`
              : runtime === "browser" ? "JS WORKER STARTING" : "PYTHON BACKEND OFFLINE"}
          </div>
        </div>

        <div className="stage-wrap">
          <MazeCanvas state={state} />
          {!connected && runtime === "python" && (
            <div className="connection-card" role="status">
              <strong>对战服务还没有连接</strong>
              <span>在项目终端运行 ./run_web_arena.sh</span>
            </div>
          )}
          <div className="stage-actions">
            <button type="button" aria-label={muted ? "开启声音" : "静音"} onClick={() => setMuted(!muted)}>
              {muted ? "音×" : "音"}
            </button>
            <button
              type="button"
              aria-label={state?.paused ? "继续" : "暂停"}
              onClick={() => dispatch({ action: "pause", paused: !state?.paused })}
            >
              {state?.paused ? "▶" : "Ⅱ"}
            </button>
            <button type="button" aria-label="全屏" onClick={() => arenaRef.current?.requestFullscreen()}>
              ⛶
            </button>
          </div>
        </div>
      </section>

      <section className="control-deck" aria-label="对战控制">
        <div className="runtime-switch" role="group" aria-label="运行时选择">
          <span>运行时</span>
          <button className={runtime === "browser" ? "active" : ""} onClick={() => changeRuntime("browser")}>Browser JS</button>
          {PYTHON_RESEARCH_ENABLED && (
            <button className={runtime === "python" ? "active" : ""} onClick={() => changeRuntime("python")}>Python Research</button>
          )}
          <small>{runtime === "browser" ? "物理、Laika、搜索在独立 Worker" : "P27b / Hybrid / Exact Shield"}</small>
        </div>
        <div className="mode-tabs">
          <button className={mode === "watch" ? "active" : ""} onClick={() => changeMode("watch")}>看它打</button>
          <button className={mode === "play" ? "active" : ""} onClick={() => changeMode("play")}>和它打</button>
          <button className={mode === "selfplay" ? "active" : ""} onClick={() => changeMode("selfplay")}>AI 对 AI</button>
        </div>

        <div className="setting-row">
          <button className="secondary" onClick={() => dispatch({ action: "new_maze", seed: seed || "random" })}>换一张迷宫 <kbd>R</kbd></button>
          <button className="secondary" onClick={() => dispatch({ action: "reset_score" })}>清零比分</button>
          <label>
            <span>种子</span>
            <input value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="random" />
          </label>
          {mode === "selfplay" && (
            <>
              <label>
                <span>红方</span>
                <select value={leftPolicy} onChange={(event) => updateMatch(event.target.value, rightPolicy)}>
                  {policies.map((policy) => <option key={policy.value} value={policy.value}>{policy.label}</option>)}
                </select>
              </label>
              <label>
                <span>黑方</span>
                <select value={rightPolicy} onChange={(event) => updateMatch(leftPolicy, event.target.value)}>
                  {policies.map((policy) => <option key={policy.value} value={policy.value}>{policy.label}</option>)}
                </select>
              </label>
            </>
          )}
          {mode === "watch" && (
            <label className={runtime === "browser" ? "" : ""}>
              <span>观战 AI</span>
              <select
                value={leftPolicy}
                onChange={(event) => updateMatch(event.target.value, "laika")}
              >
                {policies
                  .filter((policy) => policy.value !== "laika" && policy.value !== "laika-js")
                  .map((policy) => <option key={policy.value} value={policy.value}>{policy.label}</option>)}
              </select>
            </label>
          )}
        </div>
      </section>

      <section className="telemetry" aria-label="实时性能">
        <div className="metric lead">
          <span>{state?.left_policy === "p27b" ? "在线搜索" : "搜索率"}</span>
          <strong>{percent(state?.telemetry.search_rate)}</strong>
          <small>{state?.left_policy === "p27b" ? "纯网络，不做 rollout" : "搜索帧占比"}</small>
        </div>
        <div className="metric">
          <span>决策 p95</span>
          <strong>{millis(state?.telemetry.p95_ms)}</strong>
          <small>{runtime === "browser" ? "JS 搜索控制器" : state?.left_policy === "p27b" ? "P27b 网络决策" : "网络 + 安全审计"}</small>
        </div>
        <div className="metric">
          <span>{runtime === "browser" || state?.left_policy === "p27b" ? "物理帧率" : "危险接管率"}</span>
          <strong>{runtime === "browser" || state?.left_policy === "p27b"
            ? `${(state?.effective_fps ?? 0).toFixed(1)} FPS`
            : percent(state?.telemetry.change_rate)}</strong>
          <small>{runtime === "browser" || state?.left_policy === "p27b"
            ? "目标 25 FPS"
            : `${state?.telemetry.search_changes ?? 0} 次不安全审计`}</small>
        </div>
        <div className="metric">
          <span>当前决策</span>
          <strong>{millis(state?.runtime.left_decision_ms)}</strong>
          <small>{state?.telemetry.last_reason ?? "network"}</small>
        </div>
        <div className="metric">
          <span>连续获胜</span>
          <strong>{state?.streak ?? 0}</strong>
          <small>最佳 {state?.best_streak ?? 0}</small>
        </div>
      </section>

      <LeagueMonitor tactical={tacticalLeague} killfield={killfieldLeague} />

      <footer>
        <span>{runtime === "browser"
          ? "浏览器 JavaScript 物理、Laika 与搜索 · KillField MIT 对照基线"
          : "JavaScript/Canvas 渲染 · Python 1:1 物理 · P27b Frozen"}</span>
        <span>seed {state?.seed ?? seed} · <a href="https://github.com/Mingjie-Mao/tank-trouble" target="_blank" rel="noreferrer">Mingjie-Mao/tank-trouble</a></span>
      </footer>
    </main>
  );
}
