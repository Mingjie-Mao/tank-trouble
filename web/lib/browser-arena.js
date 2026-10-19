import { Game } from "./engine/game.js";
import { LaikaAI } from "./engine/laika.js";
import { Rng } from "./engine/rng.js";
import { BULLET_VISUAL_RADIUS, FPS } from "./engine/constants.js";
import { mirrorView } from "./killfield-runtime/src/killfield/mirror.js";
import { TacticalCandidateAgent } from "./tactical-candidate-agent.js";
import { DodgerJsAgent, LaikaJsAgent } from "./scripted-opponents.js";
import { PpoLeagueAgent } from "../rl/ppo-agent.js";
import { PpoMpcAgent } from "../rl/ppo-mpc.js";
import { visionFor } from "../rl/vision.js";

const EMPTY = Object.freeze({
  forward: false,
  backup: false,
  turn_left: false,
  turn_right: false,
  fire: false,
});

// Policies that run the K4 wall-contact physics. Physics is a property of the
// world rather than of an agent, so it cannot live in makeAgent; deriving it
// from the policies in play keeps every entry point correct without each one
// having to remember a flag. Two omissions of exactly that kind have already
// cost real measurements: mirrorView dropped the flag from its passthrough
// list, and playWatchGame did not accept it at all.

export function policyUsesWallSliding() {
  // The engine (lib/engine/) implements the original collision model only.
  // Policies once promoted under wall-contact physics now run under it too.
  return false;
}

const POLICIES = [
  { value: "ppo-league", label: "League PPO（强化学习）" },
  { value: "p27-js-tactical-v2", label: "Tactical（MPC 搜索）" },
  { value: "laika-js", label: "Laika（原版 AI）" },
  { value: "hunter-js", label: "Hunter（强追击）" },
  { value: "dodger-js", label: "Dodger（强规避）" },
  { value: "random-js", label: "Random（随机）" },
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

export function makeAgent(name, seed, opponentModel) {
  // Tactical: real-time MPC search over the engine.
  if (name === "p27-js-tactical-v2") {
    return new TacticalCandidateAgent({
      seed,
      oppModel: opponentModel,
      enableShotSettlementAudit: true,
    });
  }
  // Tactical with a wall-clock budget on its evasion search, as the league
  // PPO trainer runs it (12 ms). Non-deterministic by design; not in the UI.
  if (/^p27-js-tactical-v2-b\d+$/.test(name)) {
    return new TacticalCandidateAgent({
      seed,
      oppModel: opponentModel,
      enableShotSettlementAudit: true,
      evasionBudgetMs: Number(name.slice(name.lastIndexOf("-b") + 2)),
    });
  }
  // League PPO (training/js_league_ppo.py): one network forward pass per
  // frame with the move/fire safety shield, no search.
  if (name === "ppo-league") return new PpoLeagueAgent({ seed });
  // PPO-MPC (rl/ppo-mpc.js): offline teacher for distillation, too slow for
  // real time and not in the UI list; graded with rl/eval-agent.mjs.
  const mpcMatch = /^ppo-mpc(?:-k(\d+))?(?:-m(\d+))?(?:-h(\d+))?$/.exec(name);
  if (mpcMatch) {
    return new PpoMpcAgent({
      seed,
      candidates: Number(mpcMatch[1] ?? 4),
      rollouts: Number(mpcMatch[2] ?? 4),
      horizon: Number(mpcMatch[3] ?? 32),
    });
  }
  if (name === "laika-js") return new LaikaJsAgent();
  if (name === "hunter-js") return new LaikaJsAgent({ profile: "hunter" });
  if (name === "dodger-js") return new DodgerJsAgent();
  if (name === "random-js") return new RandomJsAgent(seed);
  throw new Error(`unknown browser policy: ${name}`);
}

function label(name) {
  if (name === "human") return "你";
  if (name === "laika-js") return "Laika";
  if (name === "hunter-js") return "Hunter";
  if (name === "dodger-js") return "Dodger";
  if (name === "ppo-league") return "League PPO";
  if (name === "p27-js-tactical-v2") return "Tactical";
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
  constructor({ seed = 970000, wallSliding = null } = {}) {
    this.mode = "watch";
    this.leftPolicy = "ppo-league";
    this.rightPolicy = "laika-js";
    this.seed = Number(seed) >>> 0;
    // K4 wall-contact physics, applied to the whole world (both tanks and every
    // planner rollout). `null` derives it from the policies in play, so a
    // frozen policy keeps the physics its published numbers describe and the
    // promoted champion gets its own. Pass a boolean to force either way.
    this.wallSlidingOverride = wallSliding === null || wallSliding === undefined
      ? null : Boolean(wallSliding);
    this.wallSliding = this.wallSlidingOverride ?? false;
    this.paused = false;
    this.visionEnabled = false;
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
    this.wallSliding = this.wallSlidingOverride ?? (
      policyUsesWallSliding(this.leftPolicy) || policyUsesWallSliding(this.rightPolicy)
    );
    this.game = null;
    this.game = new Game({
      seed: this.seed,
      aiFactory: nativeLaika ? (game, tank) => new LaikaAI(game, tank) : null,
      wallSliding: this.wallSliding,
    });
    this.game.scores = oldScores;
    this._configureControllers();
  }

  /** Physics follows whichever policies are currently in play. */
  _resolveWallSliding() {
    const derived = this.wallSlidingOverride ?? (
      policyUsesWallSliding(this.leftPolicy) || policyUsesWallSliding(this.rightPolicy)
    );
    this.wallSliding = derived;
    // Applied to the live game too: switching policies must switch the world,
    // otherwise a frozen policy keeps running under the champion's physics.
    if (this.game) this.game.wallSliding = derived;
    return derived;
  }

  _configureControllers() {
    this._resolveWallSliding();
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
        this.leftPolicy = payload.left_policy ?? "ppo-league";
        this.rightPolicy = "laika-js";
      } else if (nextMode === "play") {
        this.leftPolicy = "human";
        this.rightPolicy = payload.right_policy ?? "ppo-league";
      } else {
        this.leftPolicy = payload.left_policy ?? "ppo-league";
        this.rightPolicy = payload.right_policy ?? "ppo-league";
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
    } else if (action === "vision") {
      this.visionEnabled = Boolean(payload.enabled);
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
        id: bullet.name,
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
        candidates_evaluated: this.leftAgent?.priorCandidates ?? searches * 10,
        last_decision_ms: this.lastDecisionMs[0],
        last_search_ms: this.lastDecisionMs[0],
        p50_ms: percentile(this.decisionSamples, 0.5),
        p95_ms: percentile(this.decisionSamples, 0.95),
        last_reason: this.mode === "play" ? "human input" : this.lastReason,
      },
      available_policies: POLICIES,
      // Tank 0's view, computed only while the arena's AI view is on.
      vision: this.visionEnabled && this.mode !== "play" ? visionFor(this.game, this.leftAgent) : null,
    };
  }
}

export { EMPTY as EMPTY_BROWSER_CONTROLS, controlsFromTank };
