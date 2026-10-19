/**
 * Duel environment for league PPO, on this repository's own engine and Laika.
 *
 * One episode is one round on a fresh maze, paid once at the end by the
 * round's true result: a kill followed by death to a bullet already in flight
 * is a double death, never a win. The observation layout builds on
 * Cichlider/killfield's duel observation (MIT).
 *
 * The observation is "facts about the world, never answers about the
 * decision": maze, both poses and their frame-differenced motion, every live
 * bullet with its wall-only forecast, the current barrel's ballistics, and the
 * nine-move `dodgeSafety` lookahead, and how many frames each side must turn
 * before it holds a hit. Never the seed, the RNG, the opponent's buttons or its
 * goal stack.
 */

import * as C from "../lib/engine/constants.js";
import { Game, Tank, normRot } from "../lib/engine/game.js";
import { LaikaAI } from "../lib/engine/laika.js";
import { hOpen, vOpen } from "../lib/engine/maze.js";
import { mirrorView } from "../lib/killfield-runtime/src/killfield/mirror.js";
import { incomingRisk, reflectiveClosest } from "../lib/killfield-runtime/src/killfield/risk.js";
import { applyAction, makeSandbox } from "../lib/killfield-runtime/src/killfield/sandbox.js";
import { CANDIDATES } from "../lib/killfield-runtime/src/killfield/score.js";
import { TacticalCandidateAgent } from "../lib/tactical-candidate-agent.js";

// ------------------------------------------------------------------ layout

export const ACTIONS = CANDIDATES.length; // 18 = 3 throttle x 3 turn x 2 fire
export const MAP_W = 12;
export const MAP_H = 10;
export const MAP_CHANNELS = 7;
export const MAP_DIM = MAP_W * MAP_H * MAP_CHANNELS;
const RAY_COUNT = 16;
const SELF_DIM = 12;
const OPPONENT_DIM = 12;
const NAV_DIM = 10;
const AIM_DIM = 5;
export const BULLET_SLOTS = 10;
export const BULLET_DIM = 10;
const THREAT_DIM = 3;
const PHASE_DIM = 4;
const LAST_ACTION_DIM = 3;
const SELF_THREAT_COUNT_DIM = 1;
const ACTION_HISTORY_DEPTH = 3;
const OLDER_ACTIONS_DIM = (ACTION_HISTORY_DEPTH - 1) * LAST_ACTION_DIM;
const CHANGE_RATE_DIM = 1;
export const DODGE_DIM = 9;
const IDLE_STREAK_DIM = 1;
const IDLE_STREAK_CAP_FRAMES = 25;
const FIRST_STRIKE_DIM = 2;
// Turning in place covers 180 degrees in 18 frames either way.
const FIRST_STRIKE_MAX_TURNS = 18;

const RAY_OFFSET = MAP_DIM;
const SELF_OFFSET = RAY_OFFSET + RAY_COUNT;
const OPPONENT_OFFSET = SELF_OFFSET + SELF_DIM;
const NAV_OFFSET = OPPONENT_OFFSET + OPPONENT_DIM;
export const AIM_SELF_OFFSET = NAV_OFFSET + NAV_DIM;
export const AIM_OPPONENT_OFFSET = AIM_SELF_OFFSET + AIM_DIM;
export const BULLET_OFFSET = AIM_OPPONENT_OFFSET + AIM_DIM;
export const THREAT_OFFSET = BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM;
export const PHASE_OFFSET = THREAT_OFFSET + THREAT_DIM;
export const LAST_ACTION_OFFSET = PHASE_OFFSET + PHASE_DIM;
export const SELF_THREAT_COUNT_OFFSET = LAST_ACTION_OFFSET + LAST_ACTION_DIM;
const OLDER_ACTIONS_OFFSET = SELF_THREAT_COUNT_OFFSET + SELF_THREAT_COUNT_DIM;
const CHANGE_RATE_OFFSET = OLDER_ACTIONS_OFFSET + OLDER_ACTIONS_DIM;
export const DODGE_OFFSET = CHANGE_RATE_OFFSET + CHANGE_RATE_DIM;
const IDLE_STREAK_OFFSET = DODGE_OFFSET + DODGE_DIM;
export const FIRST_STRIKE_OFFSET = IDLE_STREAK_OFFSET + IDLE_STREAK_DIM;
export const OBS_DIM = FIRST_STRIKE_OFFSET + FIRST_STRIKE_DIM; // 1030

// Bumped on any layout or semantic change; checkpoints carry it. Schema 25
// only appended the first-strike pair, so a schema-24 network reads a 25
// observation unchanged by ignoring its last two values.
export const OBS_SCHEMA = "tt-duel-25";
export const PREVIOUS_OBS_SCHEMA = "tt-duel-24";

// ------------------------------------------------------------- normalisers

const RAY_CELLS = 4.0;
const RAY_STEPS_PER_CELL = 8;
const MAX_PATH_CELLS = 60.0;
const MAX_BULLET_SPEED_CELLS = 0.5;
const FORECAST_FRAMES = 75.0;
const FORECAST_BOUNCES = 2;
const HIT_RADIUS_CELLS = 0.25;
export const DODGE_HORIZON = 24;

// ---------------------------------------------------------------- terminal

export const DUEL_FRAMES = 750; // 30 s at 25 FPS; both alive past this is a draw
const DUEL_GRACE_FRAMES = C.NUMBEROFFRAMESBEFOREEND;
const WIN_FULL_FRAMES = 10 * C.FPS;
const REWARD_WIN = 1.0;
const WIN_FLOOR = 0.5;
const REWARD_LOSS = -1.0;
const REWARD_DOUBLE_DEATH = -0.1;
const REWARD_DRAW = -1.0;
const STYLE_MAX = 0.25;
const STYLE_ANCHOR_RATE = 0.13;

export const Outcome = Object.freeze({
  RUNNING: 0, WIN: 1, LOSS: 2, DOUBLE_DEATH: 3, DRAW: 4,
});

/** Full value inside ten seconds, decaying logarithmically to 0.5 at the clock. */
export function winReward(frames) {
  if (frames <= WIN_FULL_FRAMES) return REWARD_WIN;
  const elapsed = Math.min(frames, DUEL_FRAMES) / WIN_FULL_FRAMES;
  const span = Math.log(DUEL_FRAMES / WIN_FULL_FRAMES);
  const value = REWARD_WIN - (REWARD_WIN - WIN_FLOOR) * Math.log(elapsed) / span;
  return Math.min(REWARD_WIN, Math.max(WIN_FLOOR, value));
}

/**
 * Paid on every terminal: full marks for changing action no more often than
 * Laika and the planner do (~13% of frames). A rate, so dying early buys nothing.
 */
export function styleBonus(changes, frames) {
  if (frames <= 1) return STYLE_MAX;
  const rate = changes / (frames - 1);
  if (rate <= STYLE_ANCHOR_RATE) return STYLE_MAX;
  const span = Math.log(1.0 / STYLE_ANCHOR_RATE);
  return Math.min(STYLE_MAX, Math.max(0.0,
    STYLE_MAX * (1.0 - Math.log(rate / STYLE_ANCHOR_RATE) / span)));
}

export function outcomeReward(outcome, frames, doubleDeath = REWARD_DOUBLE_DEATH) {
  switch (outcome) {
    case Outcome.WIN: return winReward(frames);
    case Outcome.LOSS: return REWARD_LOSS;
    case Outcome.DOUBLE_DEATH: return doubleDeath;
    case Outcome.DRAW: return REWARD_DRAW;
    default: return 0.0;
  }
}

// ----------------------------------------------------------------- history

/** What the seat reading an observation has been doing lately. */
export class SeatHistory {
  constructor() {
    this.actions = new Array(ACTION_HISTORY_DEPTH).fill(null);
    this.changes = 0;
    this.idleStreak = 0;
  }

  record(action) {
    if (this.actions[0] !== null && this.actions[0] !== action) this.changes += 1;
    for (let i = ACTION_HISTORY_DEPTH - 1; i > 0; i--) this.actions[i] = this.actions[i - 1];
    this.actions[0] = action;
    const [throttle, turn] = CANDIDATES[action];
    this.idleStreak = throttle === 1 && turn === 1 ? this.idleStreak + 1 : 0;
  }

  changeRate(frames) {
    if (frames < 2) return 0.0;
    return Math.min(1.0, Math.max(0.0, this.changes / (frames - 1)));
  }
}

// ----------------------------------------------------------------- helpers

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Rotate a world vector into a tank's frame: +x ahead, +y to its left. */
function toOwnFrame(rotation, dx, dy) {
  const facing = (rotation - 90.0) * C.DEG;
  const sin = Math.sin(facing);
  const cos = Math.cos(facing);
  return [dx * cos + dy * sin, -dx * sin + dy * cos];
}

function wallRay(game, x, y, angle) {
  const sin = Math.sin(angle);
  const cos = Math.cos(angle);
  const steps = RAY_CELLS * RAY_STEPS_PER_CELL;
  const step = game.scale / RAY_STEPS_PER_CELL;
  for (let i = 1; i <= steps; i++) {
    const travelled = step * i;
    if (game.wallHit(x + cos * travelled, y + sin * travelled)) return travelled / game.scale;
  }
  return RAY_CELLS;
}

function cellOf(game, tank) {
  return [
    Math.max(0, Math.floor(tank.x / game.scale)),
    Math.max(0, Math.floor(tank.y / game.scale)),
  ];
}

function pathCells(game, from, to) {
  const w = game.maze.length;
  const h = game.maze[0].length;
  if (to[0] < 0 || to[1] < 0 || to[0] >= w || to[1] >= h) return null;
  const dm = game.distMap(from[0], from[1]);
  if (!dm) return null;
  const v = dm[to[0]][to[1]];
  return Number.isFinite(v) ? v : null;
}

function deadEndAt(game, cell) {
  const w = game.maze.length;
  const h = game.maze[0].length;
  if (cell[0] >= w || cell[1] >= h) return 0.0;
  const v = game.deadEnds[cell[0]][cell[1]];
  return Number.isFinite(v) ? v : C.MAXDEADENDPENALTY;
}

function tankFieldOf(game, tank) {
  return game.tankFields[tank.number];
}

/**
 * The current barrel's ballistics only: hit / self-hit / nothing, time to hit,
 * closest approach. A port of `checkBulletPath` that works for either seat
 * without building a throwaway LaikaAI.
 */
function aimAssist(game, me, out, offset) {
  if (!me.alive) {
    out[offset + 2] = 1.0;
    return;
  }
  const scale = game.scale;
  const maxClosestDistance = 2.0 * scale;
  const rad = ((me.rotation - 90) * Math.PI) / 180;
  let x = me.x + Math.cos(rad) * scale * 4.5 / 16;
  let y = me.y + Math.sin(rad) * scale * 4.5 / 16;
  let xs = Math.cos(rad) * C.BULLETSPEED * (scale / 50);
  let ys = Math.sin(rad) * C.BULLETSPEED * (scale / 50);
  const fullLife = C.BULLETLIFETIME / 3;
  let life = fullLife;
  let closest = C.MOVIEWIDTH + C.MOVIEHEIGHT;
  while (life > 0) {
    const prevX = x;
    const prevY = y;
    x += xs;
    y += ys;
    if (game.wallHit(x, y)) {
      const hitXInv = game.wallHit(prevX - xs, prevY + ys);
      const hitYInv = game.wallHit(prevX + xs, prevY - ys);
      if (hitXInv && !hitYInv) ys = -ys;
      else if (hitYInv && !hitXInv) xs = -xs;
      else { xs = -xs; ys = -ys; }
      x = prevX + xs;
      y = prevY + ys;
    }
    for (let i = 0; i < game.tanksCount; i++) {
      const tank = game.tanks[i];
      if (tank.alive && tank.pointInBbox(x, y)) {
        if (tank.pointInShape(x, y)) {
          out[offset + (tank === me ? 1 : 0)] = 1.0;
          out[offset + 3] = clamp((fullLife - life) / C.BULLETLIFETIME, 0, 1);
          out[offset + 4] = clamp(closest / (C.MOVIEWIDTH + C.MOVIEHEIGHT), 0, 1);
          return;
        }
      } else if (tank.alive && tank !== me) {
        const d = Math.abs(tank.x - x) + Math.abs(tank.y - y);
        if (d < maxClosestDistance) {
          const field = tankFieldOf(game, tank);
          const dm = game.distMap(field.x, field.y);
          const cx = Math.floor(x / scale);
          const cy = Math.floor(y / scale);
          const cd = dm && cx >= 0 && cx < dm.length && cy >= 0 && cy < dm[cx].length
            ? dm[cx][cy] : NaN;
          if (cd <= 2.0 && d < closest) closest = d;
        }
      }
    }
    life -= 1;
  }
  out[offset + 2] = 1.0;
  out[offset + 3] = clamp(fullLife / C.BULLETLIFETIME, 0, 1);
  out[offset + 4] = clamp(closest / (C.MOVIEWIDTH + C.MOVIEHEIGHT), 0, 1);
}

/** A still tank's hit shape, with its rotation's sin/cos taken once. */
function shapeOf(tank) {
  const th = tank.rotation * C.DEG;
  // Farthest shape point from the centre: the barrel tip's corners.
  const reach = tank.displayScale * Math.hypot(C.TANK_BASE_WIDTH / 2, -C.TANK_SHAPE_BARREL_TIP_Y);
  return { x: tank.x, y: tank.y, c: Math.cos(th), s: Math.sin(th), k: 1 / tank.displayScale,
    r2: reach * reach };
}

/** Tank.pointInShape for a shape from shapeOf(). */
function inShape(f, px, py) {
  const dx = px - f.x;
  const dy = py - f.y;
  if (dx * dx + dy * dy > f.r2) return false;
  const lx = (dx * f.c + dy * f.s) * f.k;
  const ly = (-dx * f.s + dy * f.c) * f.k;
  if (Math.abs(lx) <= C.TANK_BASE_WIDTH / 2 && Math.abs(ly) <= C.TANK_BASE_HEIGHT / 2) return true;
  return Math.abs(lx) <= C.TANK_SHAPE_BARREL_HALF_WIDTH
    && C.TANK_SHAPE_BARREL_TIP_Y <= ly && ly <= 0;
}

/**
 * Does a shot fired at barrel angle `angle` hit `target` before `shooter`?
 * The same flight as aimAssist (and Laika's checkBulletPath): a third of a
 * bullet's life, reflecting off walls, everyone standing still. The shooter's
 * barrel turns with the angle; its hull does not move.
 */
function shotHits(game, shooter, from, target, angle) {
  const scale = game.scale;
  const rad = ((angle - 90) * Math.PI) / 180;
  let x = shooter.x + Math.cos(rad) * scale * 4.5 / 16;
  let y = shooter.y + Math.sin(rad) * scale * 4.5 / 16;
  let xs = Math.cos(rad) * C.BULLETSPEED * (scale / 50);
  let ys = Math.sin(rad) * C.BULLETSPEED * (scale / 50);
  let life = C.BULLETLIFETIME / 3;
  while (life > 0) {
    const prevX = x;
    const prevY = y;
    x += xs;
    y += ys;
    if (game.wallHit(x, y)) {
      const hitXInv = game.wallHit(prevX - xs, prevY + ys);
      const hitYInv = game.wallHit(prevX + xs, prevY - ys);
      if (hitXInv && !hitYInv) ys = -ys;
      else if (hitYInv && !hitXInv) xs = -xs;
      else { xs = -xs; ys = -ys; }
      x = prevX + xs;
      y = prevY + ys;
    }
    if (inShape(target, x, y)) return true;
    if (inShape(from, x, y)) return false;
    life -= 1;
  }
  return false;
}

/**
 * First strike: the fewest frames `shooter` must turn in place before its
 * barrel holds a shot that hits `target`, or -1 if none within half a turn.
 * Comparing the two directions of a duel says who can fire first.
 */
export function turnsToHit(game, shooter, target) {
  if (!shooter.alive || !target.alive) return -1;
  const goal = shapeOf(target);
  const at = (rotation) => {
    const from = shapeOf(shooter);
    const th = rotation * C.DEG;
    from.c = Math.cos(th);
    from.s = Math.sin(th);
    return shotHits(game, shooter, from, goal, rotation);
  };
  const r = shooter.rotation;
  if (at(r)) return 0;
  for (let k = 1; k <= FIRST_STRIKE_MAX_TURNS; k++) {
    const turn = k * C.TANK_TURN_SPEED;
    if (at(r - turn)) return k;
    if (k < FIRST_STRIKE_MAX_TURNS && at(r + turn)) return k;
  }
  return -1;
}

/** Wall segments inflated by the half thickness, as `reflectiveClosest` wants. */
export function inflatedBoxes(game) {
  const t = game.wallHalfT;
  return game.walls.map(([x1, y1, x2, y2]) => [
    Math.min(x1, x2) - t, Math.min(y1, y2) - t,
    Math.max(x1, x2) + t, Math.max(y1, y2) + t,
  ]);
}

function nothingCanReach(game, meIndex, out, offset) {
  for (const b of game.bullets) if (!b.removed) return false;
  if (game.tanks[1 - meIndex].fire) return false;
  for (let i = 0; i < DODGE_DIM; i++) out[offset + i] = 1.0;
  return true;
}

/**
 * Per-move survival outlook for the nine no-fire moves, rolled `horizon`
 * frames against bullets already in flight with the other tank holding its
 * current buttons (L1). Positive: survives, scaled by closest bullet
 * clearance. Negative: dies, less negative the longer it lasted.
 *
 * Bit-identical to `dodgeSafetyReference`, at a fraction of the cost. Tanks
 * never collide with each other and none of the nine moves fires, so until
 * the moment I am hit the rest of the world — the other tank, every bullet,
 * the round clock — evolves the same way whichever move I hold. It is rolled
 * once with my tank taken out of play, recording each frame's hit-test
 * points and surviving bullets; each move then simulates only its own hull
 * against that record. Bullets run before tanks, so each frame's hit tests
 * meet my hull where the previous frame left it, and only then does it move.
 */
export function dodgeSafety(game, meIndex, out, offset, horizon = DODGE_HORIZON) {
  if (nothingCanReach(game, meIndex, out, offset)) return;
  if (!game.tanks[meIndex].alive) {
    // Already dead during settlement: every rollout "dies" on its first frame.
    for (let i = 0; i < DODGE_DIM; i++) out[offset + i] = -1.0;
    return;
  }
  const view = meIndex === 0 ? game : mirrorView(game);

  const world = makeSandbox(view, "L1", 0);
  const ghost = world.tanks[0];
  ghost.alive = false; // no motion, no hit tests, aliveCount untouched
  const frames = [];
  for (let f = 0; f < horizon; f++) {
    // The freeze is set by the round logic at the end of a frame, so it
    // decides whether bullets and tanks run on the next one.
    const frozen = world.frozen;
    const updating = [];
    for (const b of world.bullets) if (!b.removed) updating.push(b);
    const events = world.step();
    // Each bullet's hit-test point this frame, in update order.
    const hits = [];
    if (!frozen) for (const b of updating) hits.push(b.x, b.y);
    const clear = [];
    for (const b of world.bullets) clear.push(b.x, b.y);
    let ended = world.frozen;
    for (const e of events) if (e[0] === "round_end") ended = true;
    frames.push({ frozen, hits, clear, ended });
    if (ended) break;
  }

  const real = view.tanks[0];
  const hull = {
    frozen: false,
    wallSliding: world.wallSliding,
    scale: world.scale,
    wallHit: (x, y) => world.wallGrid.hit(x, y),
    lockedControl: () => false,
  };
  const inv = 1 / Math.max(world.scale, 1e-6);
  for (let move = 0; move < DODGE_DIM; move++) {
    const [throttle, turn] = CANDIDATES[move * 2];
    const me = Object.create(Tank.prototype);
    Object.assign(me, real);
    me.game = hull;
    me.ai = null;
    me.forward = throttle === 2;
    me.backup = throttle === 0;
    me.turnLeft = turn === 0;
    me.turnRight = turn === 2;
    me.fire = false;
    let minClearance = 8.0;
    let value = null;
    for (let f = 0; f < frames.length; f++) {
      const frame = frames[f];
      const hits = frame.hits;
      for (let i = 0; i < hits.length; i += 2) {
        if (me.pointInShape(hits[i], hits[i + 1])) { value = -1.0 + 0.5 * (f / horizon); break; }
      }
      if (value !== null) break;
      if (!frame.frozen) me.update();
      const clear = frame.clear;
      for (let i = 0; i < clear.length; i += 2) {
        const d = Math.hypot(clear[i] - me.x, clear[i + 1] - me.y) * inv;
        if (d < minClearance) minClearance = d;
      }
      if (frame.ended) break;
    }
    out[offset + move] = value ?? clamp(minClearance / 8.0, 0, 1);
  }
}

/** The direct definition: nine independent full-world rollouts. Kept as the oracle. */
export function dodgeSafetyReference(game, meIndex, out, offset, horizon = DODGE_HORIZON) {
  if (nothingCanReach(game, meIndex, out, offset)) return;
  const view = meIndex === 0 ? game : mirrorView(game);
  for (let move = 0; move < DODGE_DIM; move++) {
    const [throttle, turn] = CANDIDATES[move * 2];
    const sb = makeSandbox(view, "L1", 0);
    applyAction(sb, [throttle, turn, 0]);
    const me = sb.tanks[0];
    let minClearance = 8.0;
    let survived = true;
    let elapsed = 0;
    while (elapsed < horizon) {
      const events = sb.step();
      if (!me.alive) { survived = false; break; }
      const inv = 1 / Math.max(sb.scale, 1e-6);
      for (const b of sb.bullets) {
        const d = Math.hypot(b.x - me.x, b.y - me.y) * inv;
        if (d < minClearance) minClearance = d;
      }
      let ended = sb.frozen;
      for (const e of events) if (e[0] === "round_end") ended = true;
      if (ended) break;
      elapsed += 1;
    }
    out[offset + move] = survived
      ? clamp(minClearance / 8.0, 0, 1)
      : -1.0 + 0.5 * (elapsed / horizon);
  }
}

// ------------------------------------------------------------------ encode

/**
 * Encode seat `meIndex`'s observation into `out[offset .. offset + OBS_DIM)`,
 * and its live-bullet mask into `mask[maskOffset .. + BULLET_SLOTS)`.
 */
export function encodeObservation(game, meIndex, prevPose, boxes, history, frames,
  out, offset = 0, mask = null, maskOffset = 0) {
  out.fill(0, offset, offset + OBS_DIM);
  if (mask) mask.fill(0, maskOffset, maskOffset + BULLET_SLOTS);
  const v = out;
  const o = offset;
  const otherIndex = 1 - meIndex;
  const me = game.tanks[meIndex];
  const them = game.tanks[otherIndex];
  const scale = game.scale;
  const maze = game.maze;
  const mw = maze.length;
  const mh = maze[0].length;
  const width = mw * scale;
  const height = mh * scale;
  const span = width + height;
  const facing = (me.rotation - 90.0) * C.DEG;

  // --- maze grid, padded to the generator's largest arena
  const myCell = cellOf(game, me);
  const theirCell = cellOf(game, them);
  for (let x = 0; x < Math.min(mw, MAP_W); x++) {
    for (let y = 0; y < Math.min(mh, MAP_H); y++) {
      const base = o + (x * MAP_H + y) * MAP_CHANNELS;
      v[base] = 1.0;
      v[base + 1] = hOpen(maze, x, y - 1) ? 0 : 1; // top
      v[base + 2] = vOpen(maze, x + 1, y) ? 0 : 1; // right
      // The arena boundary is always a wall, whatever the edge flag says.
      v[base + 3] = y < mh - 1 && hOpen(maze, x, y) ? 0 : 1; // bottom
      v[base + 4] = vOpen(maze, x, y) ? 0 : 1; // left
      v[base + 5] = myCell[0] === x && myCell[1] === y ? 1 : 0;
      v[base + 6] = them.alive && theirCell[0] === x && theirCell[1] === y ? 1 : 0;
    }
  }

  // --- wall rays around the hull
  for (let i = 0; i < RAY_COUNT; i++) {
    const angle = facing + (2 * Math.PI * i) / RAY_COUNT;
    v[o + RAY_OFFSET + i] = wallRay(game, me.x, me.y, angle) / RAY_CELLS;
  }

  // --- self
  const speedScale = MAX_BULLET_SPEED_CELLS * scale;
  const maxSlots = Math.max(1, game.settingsMaxBullets);
  const myPrev = prevPose[meIndex];
  const [myAhead, myLeft] = toOwnFrame(me.rotation, me.x - myPrev[0], me.y - myPrev[1]);
  let s = o + SELF_OFFSET;
  v[s] = clamp(me.x / width, 0, 1);
  v[s + 1] = clamp(me.y / height, 0, 1);
  v[s + 2] = Math.cos(facing);
  v[s + 3] = Math.sin(facing);
  v[s + 4] = clamp(myAhead / speedScale, -1, 1);
  v[s + 5] = clamp(myLeft / speedScale, -1, 1);
  v[s + 6] = clamp(normRot(me.rotation - myPrev[2]) / C.TANK_TURN_SPEED, -1, 1);
  v[s + 7] = Math.max(0, game.settingsMaxBullets - me.bulletsFired) / maxSlots;
  v[s + 8] = game.weaponReady(me) ? 1 : 0;
  v[s + 9] = me.alive ? 1 : 0;
  v[s + 10] = me.hitSomething ? 1 : 0;
  v[s + 11] = me.wallSliding ? 1 : 0;

  // --- opponent, in my frame
  const [relAhead, relLeft] = toOwnFrame(me.rotation, them.x - me.x, them.y - me.y);
  const theirPrev = prevPose[otherIndex];
  const [theirAhead, theirLeft] = toOwnFrame(
    me.rotation, them.x - theirPrev[0], them.y - theirPrev[1]);
  const relativeHeading = (them.rotation - me.rotation) * C.DEG;
  s = o + OPPONENT_OFFSET;
  v[s] = clamp(relAhead / span, -1, 1);
  v[s + 1] = clamp(relLeft / span, -1, 1);
  v[s + 2] = Math.cos(relativeHeading);
  v[s + 3] = Math.sin(relativeHeading);
  v[s + 4] = clamp(theirAhead / speedScale, -1, 1);
  v[s + 5] = clamp(theirLeft / speedScale, -1, 1);
  v[s + 6] = clamp(normRot(them.rotation - theirPrev[2]) / C.TANK_TURN_SPEED, -1, 1);
  v[s + 7] = Math.max(0, game.settingsMaxBullets - them.bulletsFired) / maxSlots;
  v[s + 8] = game.weaponReady(them) ? 1 : 0;
  v[s + 9] = them.alive ? 1 : 0;
  v[s + 10] = them.hitSomething ? 1 : 0;
  v[s + 11] = them.wallSliding ? 1 : 0;

  // --- navigation
  s = o + NAV_OFFSET;
  const here = pathCells(game, myCell, theirCell);
  v[s] = here === null ? 1.0 : clamp(here / MAX_PATH_CELLS, 0, 1);
  if (here !== null) {
    let best = null;
    const steps = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    for (const [dx, dy] of steps) {
      let open;
      if (dy === -1) open = hOpen(maze, myCell[0], myCell[1] - 1);
      else if (dx === 1) open = vOpen(maze, myCell[0] + 1, myCell[1]);
      else if (dy === 1) open = myCell[1] < mh - 1 && hOpen(maze, myCell[0], myCell[1]);
      else open = vOpen(maze, myCell[0], myCell[1]);
      if (!open) continue;
      const there = pathCells(game, [myCell[0] + dx, myCell[1] + dy], theirCell);
      if (there !== null && there < here && (best === null || there < best[0])) {
        const [ahead, left] = toOwnFrame(me.rotation, dx, dy);
        const quadrant = Math.abs(ahead) >= Math.abs(left)
          ? (ahead > 0 ? 0 : 2)
          : (left > 0 ? 1 : 3);
        best = [there, quadrant];
      }
    }
    if (best) v[s + 1 + best[1]] = 1.0;
  }
  const straight = Math.hypot(relAhead, relLeft);
  v[s + 5] = clamp(straight / span, 0, 1);
  if (straight > 1e-9) {
    v[s + 6] = relAhead / straight;
    v[s + 7] = relLeft / straight;
  }
  v[s + 8] = deadEndAt(game, myCell) / C.MAXDEADENDPENALTY;
  v[s + 9] = deadEndAt(game, theirCell) / C.MAXDEADENDPENALTY;

  // --- aim assist, mine and theirs
  aimAssist(game, me, v, o + AIM_SELF_OFFSET);
  aimAssist(game, them, v, o + AIM_OPPONENT_OFFSET);

  // --- every live bullet (both tanks hold five, so ten slots is exact)
  const hitRadius = HIT_RADIUS_CELLS * scale;
  let worstPass = 1.0;
  let slot = 0;
  let selfThreats = 0;
  for (const bullet of game.bullets) {
    if (bullet.removed) continue;
    if (slot >= BULLET_SLOTS) break;
    const base = o + BULLET_OFFSET + slot * BULLET_DIM;
    const [ahead, left] = toOwnFrame(me.rotation, bullet.x - me.x, bullet.y - me.y);
    const [vx, vy] = toOwnFrame(me.rotation, bullet.xSpeed, bullet.ySpeed);
    const mine = bullet.owner === me;
    v[base] = clamp(ahead / span, -1, 1);
    v[base + 1] = clamp(left / span, -1, 1);
    v[base + 2] = clamp(vx / speedScale, -1, 1);
    v[base + 3] = clamp(vy / speedScale, -1, 1);
    v[base + 4] = mine ? 1 : 0;
    v[base + 5] = bullet.hasBounced ? 1 : 0;
    v[base + 6] = clamp(bullet.lifetime / C.BULLETLIFETIME, 0, 1);
    const speed = Math.hypot(bullet.xSpeed, bullet.ySpeed);
    if (speed > 1e-9) {
      // An unbounced shot of mine flies straight away from me and the forecast
      // assumes I hold still, so it is not a threat to me.
      if (!(mine && !bullet.hasBounced)) {
        const approach = reflectiveClosest(
          bullet.x, bullet.y, bullet.xSpeed / speed, bullet.ySpeed / speed, speed,
          FORECAST_FRAMES, FORECAST_BOUNCES, boxes, me.x, me.y);
        if (approach.distance <= hitRadius) {
          v[base + 7] = 1.0;
          v[base + 9] = clamp(approach.frame / FORECAST_FRAMES, 0, 1);
          selfThreats += 1;
        } else {
          v[base + 9] = 1.0;
        }
        worstPass = Math.min(worstPass, Math.min(1.0, approach.distance / (2.0 * scale)));
      } else {
        v[base + 9] = 1.0;
      }
      if (!(!mine && !bullet.hasBounced) && them.alive) {
        const approach = reflectiveClosest(
          bullet.x, bullet.y, bullet.xSpeed / speed, bullet.ySpeed / speed, speed,
          FORECAST_FRAMES, FORECAST_BOUNCES, boxes, them.x, them.y);
        v[base + 8] = approach.distance <= hitRadius ? 1 : 0;
      }
    } else {
      v[base + 9] = 1.0;
    }
    if (mask) mask[maskOffset + slot] = 1;
    slot += 1;
  }

  // --- threat summary
  const view = meIndex === 0 ? game : mirrorView(game);
  v[o + THREAT_OFFSET] = clamp(incomingRisk(view, boxes), 0, 1);
  const otherView = meIndex === 0 ? mirrorView(game) : game;
  v[o + THREAT_OFFSET + 1] = clamp(incomingRisk(otherView, boxes), 0, 1);
  v[o + THREAT_OFFSET + 2] = worstPass;
  v[o + SELF_THREAT_COUNT_OFFSET] = selfThreats / BULLET_SLOTS;

  // --- round phase and clock (keyed on aliveCount: one Game per episode)
  const settling = game.aliveCount <= 1 && !game.frozen;
  v[o + PHASE_OFFSET] = !settling && !game.frozen ? 1 : 0;
  v[o + PHASE_OFFSET + 1] = settling ? 1 : 0;
  v[o + PHASE_OFFSET + 2] = game.frozen ? 1 : 0;
  v[o + PHASE_OFFSET + 3] = clamp(frames / DUEL_FRAMES, 0, 1);

  // --- what I have been doing
  for (let age = 0; age < ACTION_HISTORY_DEPTH; age++) {
    const action = history.actions[age];
    if (action === null) continue;
    const a = CANDIDATES[action];
    const base = o + (age === 0 ? LAST_ACTION_OFFSET
      : OLDER_ACTIONS_OFFSET + (age - 1) * LAST_ACTION_DIM);
    v[base] = a[0] / 2;
    v[base + 1] = a[1] / 2;
    v[base + 2] = a[2];
  }
  v[o + CHANGE_RATE_OFFSET] = history.changeRate(frames);
  dodgeSafety(game, meIndex, v, o + DODGE_OFFSET);
  v[o + IDLE_STREAK_OFFSET] = Math.min(history.idleStreak, IDLE_STREAK_CAP_FRAMES)
    / IDLE_STREAK_CAP_FRAMES;
  v[o + FIRST_STRIKE_OFFSET] = firstStrikeFeature(turnsToHit(game, me, them));
  v[o + FIRST_STRIKE_OFFSET + 1] = firstStrikeFeature(turnsToHit(game, them, me));
}

const firstStrikeFeature = (turns) => (turns < 0 ? 1 : turns / (FIRST_STRIKE_MAX_TURNS + 1));

// -------------------------------------------------------------- opponents

/** Opponent kinds a slot can hold. Frozen checkpoints are driven by the trainer. */
export const OpponentKind = Object.freeze({ LAIKA: 0, TACTICAL: 1, FROZEN: 2 });

function writeAction(tank, action) {
  const [throttle, turn, fire] = CANDIDATES[action];
  tank.forward = throttle === 2;
  tank.backup = throttle === 0;
  tank.turnLeft = turn === 0;
  tank.turnRight = turn === 2;
  tank.fire = fire === 1;
}

function poses(game) {
  return game.tanks.map((t) => [t.x, t.y, t.rotation]);
}

/**
 * One round against one opponent. `opponent` is an OpponentKind; a FROZEN
 * opponent's action is supplied to `step()` by the caller each frame.
 */
export class DuelEpisode {
  constructor(seed, opponent, { doubleDeathReward = REWARD_DOUBLE_DEATH } = {}) {
    this.seed = seed >>> 0;
    this.opponent = opponent;
    // What a mutual kill pays. The default keeps it just below zero; the true
    // win metric counts it as a non-win, so a run can price it closer to a loss.
    this.doubleDeathReward = doubleDeathReward;
    this.game = new Game({
      seed: this.seed,
      aiFactory: opponent === OpponentKind.LAIKA ? (g, t) => new LaikaAI(g, t) : null,
    });
    this.agent = null;
    if (opponent === OpponentKind.TACTICAL) {
      // L1 against a learned policy: L2 would replay the Laika script inside
      // its lookahead, a strong and wrong prior about us.
      this.agent = new TacticalCandidateAgent({
        seed: (this.seed ^ 0x5bd1e995) >>> 0,
        oppModel: "L1",
        enableShotSettlementAudit: true,
        // A training opponent need not be reproducible, but one 130 ms
        // evasion search stalls its whole worker. Wall-clock narrowing caps
        // that tail; graded matches against Tactical should not use this env.
        evasionBudgetMs: 12,
      });
      this.view = mirrorView(this.game);
    }
    this.frames = 0;
    this.outcome = Outcome.RUNNING;
    this.prevPose = poses(this.game);
    this.boxes = inflatedBoxes(this.game);
    this.history = [new SeatHistory(), new SeatHistory()];
    this.shots = 0;
    this.hits = 0;
    // Every bullet-tank hit this round as [owner, victim, frame]: enough to
    // say after the fact who killed whom, and whether anyone shot themselves.
    this.kills = [];
  }

  observe(meIndex, out, offset = 0, mask = null, maskOffset = 0) {
    encodeObservation(this.game, meIndex, this.prevPose, this.boxes,
      this.history[meIndex], this.frames, out, offset, mask, maskOffset);
  }

  /**
   * Advance one frame with tank 0's buttons already written by an external
   * controller (any browser agent). The buttons are folded back into the
   * Discrete(18) index so the round's bookkeeping is identical to step().
   */
  stepControlled() {
    const t = this.game.tanks[0];
    const throttle = t.forward ? 2 : t.backup ? 0 : 1;
    const turn = t.turnLeft ? 0 : t.turnRight ? 2 : 1;
    return this.step(6 * throttle + 2 * turn + (t.fire ? 1 : 0), null);
  }

  /** Advance one frame. Returns the reward (non-zero only at the terminal). */
  step(action, opponentAction = null) {
    const game = this.game;
    this.history[0].record(action);
    if (game.tanks[0].alive && !game.frozen) writeAction(game.tanks[0], action);
    this.prevPose = poses(game);
    if (this.agent) {
      if (game.tanks[1].alive && !game.frozen) this.agent.drive(this.view);
    } else if (this.opponent === OpponentKind.FROZEN && opponentAction !== null) {
      this.history[1].record(opponentAction);
      if (game.tanks[1].alive && !game.frozen) writeAction(game.tanks[1], opponentAction);
    }

    const events = game.step();
    this.frames += 1;
    let winner;
    let ended = false;
    for (const e of events) {
      if (e[0] === "fire" && e[1] === 0) this.shots += 1;
      if (e[0] === "hit" && e[1] === 0 && e[2] === 1) this.hits += 1;
      if (e[0] === "hit") this.kills.push([e[1], e[2], this.frames]);
      if (e[0] === "round_end") { ended = true; winner = e[1]; }
    }
    if (ended) {
      this.outcome = winner === 0 ? Outcome.WIN
        : winner === null ? Outcome.DOUBLE_DEATH : Outcome.LOSS;
    } else if (this.frames >= DUEL_FRAMES && game.endCount < 0) {
      this.outcome = Outcome.DRAW;
    } else if (this.frames >= DUEL_FRAMES + DUEL_GRACE_FRAMES) {
      this.outcome = Outcome.DRAW;
    }
    if (this.outcome === Outcome.RUNNING) return 0.0;
    return outcomeReward(this.outcome, this.frames, this.doubleDeathReward)
      + styleBonus(this.history[0].changes, this.frames);
  }
}
