import { Game } from "./killfield-runtime/src/game.js";
import { LaikaAI } from "./killfield-runtime/src/laika.js";
import { Rng } from "./killfield-runtime/src/rng.js";
import { BULLET_VISUAL_RADIUS, FPS } from "./killfield-runtime/src/constants.js";
import { KillFieldAgent } from "./killfield-runtime/src/killfield/teacher.js";
import { mirrorView } from "./killfield-runtime/src/killfield/mirror.js";
import { SaferSearchAgent } from "./safer-search-agent.js";
import { ConsensusSafetyAgent } from "./consensus-safety-agent.js";
import { TwoStageSearchAgent } from "./two-stage-search-agent.js";
import { ShadowCorrectionAgent } from "./shadow-correction-agent.js";
import { TacticalSafetyAgent } from "./tactical-safety-agent.js";
import { TacticalV2Agent } from "./tactical-v2-agent.js";
import { DodgerJsAgent, LaikaJsAgent } from "./scripted-opponents.js";

const EMPTY = Object.freeze({
  forward: false,
  backup: false,
  turn_left: false,
  turn_right: false,
  fire: false,
});

const POLICIES = [
  { value: "p27-js-tactical-v2", label: "Tactical（当前冠军）" },
  { value: "p27-js-tactical", label: "Tactical Legacy（冻结基线）" },
  { value: "killfield-js", label: "KillField JS（第三方速度基线）" },
  { value: "laika-js", label: "Laika（官方脚本）" },
  { value: "hunter-js", label: "Hunter JS（强追击）" },
  { value: "dodger-js", label: "Dodger JS（强规避）" },
  { value: "random-js", label: "Random JS" },
  { value: "idle-js", label: "Idle JS" },
];

function controls(value = EMPTY) {
  return {
    forward: Boolean(value.forward),
    backup: Boolean(value.backup),
    turn_left: Boolean(value.turn_left),
    turn_right: Boolean(value.turn_right),
    fire: Boolean(value.fire),
  };
}

function applyControls(tank, value) {
  const next = controls(value);
  tank.forward = next.forward;
  tank.backup = next.backup;
  tank.turnLeft = next.turn_left;
  tank.turnRight = next.turn_right;
  tank.fire = next.fire;
}

function controlsFromTank(tank) {
  return {
    forward: Boolean(tank.forward),
    backup: Boolean(tank.backup),
    turn_left: Boolean(tank.turnLeft),
    turn_right: Boolean(tank.turnRight),
    fire: Boolean(tank.fire),
  };
}

class RandomJsAgent {
  constructor(seed = 0) {
    this.rng = new Rng(seed);
    this.remaining = 0;
    this.lastDecisionKind = "random";
    this.planMs = [];
  }

  reset() {
    this.remaining = 0;
  }

  drive(game) {
    const tank = game.tanks[0];
    if (this.remaining <= 0) {
      const throttle = this.rng.randrange(3);
      const turn = this.rng.randrange(3);
      this.action = [throttle, turn, this.rng.random() < 0.08 ? 1 : 0];
      this.remaining = 3 + this.rng.randrange(8);
    }
    this.remaining -= 1;
    const [throttle, turn, fire] = this.action ?? [1, 1, 0];
    tank.forward = throttle === 2;
    tank.backup = throttle === 0;
    tank.turnLeft = turn === 0;
    tank.turnRight = turn === 2;
    tank.fire = fire === 1;
  }

  telemetry() {
    return { decision: "random", planMedianMs: 0, planP95Ms: 0 };
  }
}

class IdleJsAgent {
  constructor() {
    this.lastDecisionKind = "idle";
  }

  reset() {}

  drive(game) {
    applyControls(game.tanks[0], EMPTY);
  }

  telemetry() {
    return { decision: "idle", planMedianMs: 0, planP95Ms: 0 };
  }
}

function makeAgent(name, seed, opponentModel) {
  if (name === "p27-js-tactical-v2") {
    return new TacticalV2Agent({ seed, oppModel: opponentModel });
  }
  if (name === "p27-js-tactical") {
    return new TacticalSafetyAgent({ seed, oppModel: opponentModel });
  }
  if (name === "p27-js-shadow") {
    return new ShadowCorrectionAgent({ seed, oppModel: opponentModel });
  }
  const twoStageMatch = /^p27-js-two-stage(?:-m(\d+))?$/.exec(name);
  if (twoStageMatch) {
    return new TwoStageSearchAgent({
      seed,
      oppModel: opponentModel,
      marginThreshold: Number(twoStageMatch[1] ?? 30),
    });
  }
  if (name === "p27-js-consensus") {
    return new ConsensusSafetyAgent({ seed, oppModel: opponentModel });
  }
  if (name === "p27-js-shield") {
    return new SaferSearchAgent({ seed, oppModel: opponentModel });
  }
  const horizonMatch = /^killfield-h(\d+)-js$/.exec(name);
  if (horizonMatch) {
    return new KillFieldAgent({
      seed,
      oppModel: opponentModel,
      horizon: Number(horizonMatch[1]),
    });
  }
  if (name === "killfield-js") {
    return new KillFieldAgent({ seed, oppModel: opponentModel });
  }
  if (name === "laika-js") return new LaikaJsAgent();
  if (name === "hunter-js") return new LaikaJsAgent({ profile: "hunter" });
  if (name === "dodger-js") return new DodgerJsAgent();
  if (name === "random-js") return new RandomJsAgent(seed);
  if (name === "idle-js") return new IdleJsAgent();
  throw new Error(`unknown browser policy: ${name}`);
}

function label(name) {
  if (name === "human") return "你";
  if (name === "laika-js") return "Laika";
  if (name === "hunter-js") return "Hunter JS";
  if (name === "dodger-js") return "Dodger JS";
  if (name === "p27-js-shield") return "JS Safety prototype（未晋级）";
  if (name === "p27-js-tactical") return "Tactical Legacy";
  if (name === "p27-js-tactical-v2") return "Tactical";
  if (name === "p27-js-consensus") return "JS Consensus Safety（候选）";
  if (name.startsWith("p27-js-two-stage")) return "JS Two-stage Search（候选）";
  const horizonMatch = /^killfield-h(\d+)-js$/.exec(name);
  if (horizonMatch) return `Deep Search H${horizonMatch[1]} JS（实验）`;
  return POLICIES.find((policy) => policy.value === name)?.label ?? name;
}

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
}

/**
 * Browser-native fixed-step duel. In the web product a dedicated Worker owns
 * this arena; Node evaluation scripts instantiate it directly.
 */
export class BrowserArena {
  constructor({ seed = 970000 } = {}) {
    this.mode = "watch";
    this.leftPolicy = "p27-js-tactical-v2";
    this.rightPolicy = "laika-js";
    this.seed = Number(seed) >>> 0;
    this.paused = false;
    this.humanControls = controls();
    this.streak = 0;
    this.bestStreak = 0;
    this.lastEvents = [];
    this.lastStepMs = 0;
    this.lastDecisionMs = [0, 0];
    this.stepTimestamps = [];
    this.policyFrames = 0;
    this.searches = 0;
    this.searchChanges = 0;
    this.decisionSamples = [];
    this.lastReason = "initialising";
    this._newGame(this.seed, false);
  }

  _newGame(seed, keepScores = true) {
    const oldScores = keepScores && this.game ? this.game.scores.slice() : [0, 0];
    this.seed = Number(seed) >>> 0;
    const nativeLaika = this.mode === "watch";
    this.game = new Game({
      seed: this.seed,
      aiFactory: nativeLaika ? (game, tank) => new LaikaAI(game, tank) : null,
    });
    this.game.scores = oldScores;
    this._configureControllers();
  }

  _configureControllers() {
    const nativeLaika = this.mode === "watch";
    this.game.aiFactory = nativeLaika
      ? (game, tank) => new LaikaAI(game, tank)
      : null;
    this.game.tanks[1].ai = nativeLaika
      ? new LaikaAI(this.game, this.game.tanks[1])
      : null;

    this.leftAgent = this.mode === "play"
      ? null
      : makeAgent(this.leftPolicy, this.seed + 101, nativeLaika ? "L2" : "L1");
    this.rightAgent = this.mode === "selfplay"
      ? makeAgent(this.rightPolicy, this.seed + 7919, "L1")
      : this.mode === "play"
        ? makeAgent(this.rightPolicy, this.seed + 7919, "L1")
        : null;
    this.rightView = this.rightAgent ? mirrorView(this.game) : null;
    this.lastDecisionMs = [0, 0];
  }

  _drive(agent, game, side) {
    if (!agent) return;
    const beforeGuard = agent.ownBulletGuardEvents ?? 0;
    const started = performance.now();
    agent.drive(game);
    const elapsed = performance.now() - started;
    this.lastDecisionMs[side] = elapsed;
    if (side !== 0) return;
    this.policyFrames += 1;
    this.decisionSamples.push(elapsed);
    if (this.decisionSamples.length > 600) this.decisionSamples.shift();
    const kind = agent.lastDecisionKind ?? "controller";
    this.lastReason = kind;
    if (kind === "plan" || kind === "post_kill_plan") this.searches += 1;
    this.searchChanges += Math.max(0, (agent.ownBulletGuardEvents ?? 0) - beforeGuard);
  }

  step(now = performance.now()) {
    if (this.paused) return this.state();
    const started = performance.now();
    this.lastDecisionMs = [0, 0];

    if (!this.game.frozen) {
      if (this.mode === "play") {
        applyControls(this.game.tanks[0], this.humanControls);
      } else {
        this._drive(this.leftAgent, this.game, 0);
      }
      if (this.rightAgent && this.rightView) {
        this._drive(this.rightAgent, this.rightView, 1);
      }
    } else {
      applyControls(this.game.tanks[0], EMPTY);
      if (!this.game.tanks[1].ai) applyControls(this.game.tanks[1], EMPTY);
    }

    this.lastEvents = this.game.step().map((event) => Array.from(event));
    for (const event of this.lastEvents) {
      if (event[0] === "round_end") {
        if (event[1] === 0) {
          this.streak += 1;
          this.bestStreak = Math.max(this.bestStreak, this.streak);
        } else {
          this.streak = 0;
        }
      }
      if (event[0] === "new_round") {
        this._configureControllers();
      }
    }
    this.lastStepMs = performance.now() - started;
    this.stepTimestamps.push(now);
    if (this.stepTimestamps.length > 120) this.stepTimestamps.shift();
    return this.state();
  }

  command(payload) {
    const action = payload?.action;
    if (action === "input") {
      this.humanControls = controls(payload.controls);
    } else if (action === "mode") {
      const nextMode = payload.mode ?? this.mode;
      if (!["watch", "play", "selfplay"].includes(nextMode)) {
        throw new Error("invalid mode");
      }
      this.mode = nextMode;
      if (nextMode === "watch") {
        this.leftPolicy = payload.left_policy ?? "p27-js-tactical-v2";
        this.rightPolicy = "laika-js";
      } else if (nextMode === "play") {
        this.leftPolicy = "human";
        this.rightPolicy = payload.right_policy ?? "p27-js-tactical-v2";
      } else {
        this.leftPolicy = payload.left_policy ?? "p27-js-tactical-v2";
        this.rightPolicy = payload.right_policy ?? "p27-js-tactical-v2";
      }
      this._configureControllers();
    } else if (action === "new_maze") {
      const value = payload.seed;
      const nextSeed = value === undefined || value === null || value === "" || value === "random"
        ? Math.floor(Math.random() * 0x7fffffff)
        : Number(value);
      if (!Number.isFinite(nextSeed)) throw new Error("invalid seed");
      this._newGame(nextSeed, true);
    } else if (action === "reset_score") {
      this.game.scores = [0, 0];
      this.streak = 0;
      this.bestStreak = 0;
    } else if (action === "pause") {
      this.paused = Boolean(payload.paused ?? !this.paused);
    } else {
      throw new Error(`unknown action: ${action}`);
    }
    return this.state();
  }

  state() {
    const timestamps = this.stepTimestamps;
    const effectiveFps = timestamps.length > 1
      ? ((timestamps.length - 1) * 1000) / (timestamps.at(-1) - timestamps[0])
      : 0;
    const searches = this.searches;
    const frames = this.policyFrames;
    const mazeWidth = this.game.maze.length;
    const mazeHeight = this.game.maze[0].length;
    return {
      connected: true,
      mode: this.mode,
      paused: this.paused,
      seed: this.seed,
      frame: this.game.frame,
      round: this.game.roundNumber,
      fps: FPS,
      effective_fps: effectiveFps,
      frozen: this.game.frozen,
      world_width: mazeWidth * this.game.scale,
      world_height: mazeHeight * this.game.scale,
      wall_width: this.game.wallHalfT * 2,
      bullet_radius: BULLET_VISUAL_RADIUS,
      walls: this.game.walls,
      scores: this.game.scores.slice(),
      streak: this.streak,
      best_streak: this.bestStreak,
      left_policy: this.mode === "play" ? "human" : this.leftPolicy,
      right_policy: this.rightPolicy,
      left_label: label(this.mode === "play" ? "human" : this.leftPolicy),
      right_label: label(this.rightPolicy),
      tanks: this.game.tanks.map((tank, index) => ({
        index,
        x: tank.x,
        y: tank.y,
        rotation: tank.rotation,
        display_scale: tank.displayScale,
        alive: tank.alive,
        bullets_fired: tank.bulletsFired,
      })),
      bullets: this.game.bullets.map((bullet) => ({
        x: bullet.x,
        y: bullet.y,
        owner: bullet.owner.number,
        lifetime: bullet.lifetime,
      })),
      runtime: {
        step_ms: this.lastStepMs,
        left_decision_ms: this.lastDecisionMs[0],
        right_decision_ms: this.lastDecisionMs[1],
      },
      telemetry: {
        frames,
        searches,
        search_rate: searches / Math.max(1, frames),
        search_changes: this.searchChanges,
        change_rate: this.searchChanges / Math.max(1, searches),
        deadline_hits: 0,
        candidates_evaluated: searches * 10,
        last_decision_ms: this.lastDecisionMs[0],
        last_search_ms: this.lastDecisionMs[0],
        p50_ms: percentile(this.decisionSamples, 0.5),
        p95_ms: percentile(this.decisionSamples, 0.95),
        last_reason: this.mode === "play" ? "human input" : this.lastReason,
      },
      available_policies: POLICIES,
    };
  }
}

export { EMPTY as EMPTY_BROWSER_CONTROLS, controlsFromTank };
