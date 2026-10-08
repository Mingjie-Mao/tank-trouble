/**
 * The league-PPO policy (`training/js_league_ppo.py`) as a browser agent.
 *
 * Each frame it encodes the same 1028-value observation the policy was
 * trained on (`duel-env.js`), runs the network in plain JS and samples an
 * action. Sampling, not argmax: the greedy policy stalls into 30 s draws
 * (45.3% vs 56.5% true wins over the same 400 Laika seeds).
 *
 * The weights are ~1.7 MB, so they are fetched with a dynamic import the
 * first time the policy is chosen and cached for every round after; until
 * they arrive the tank holds still.
 */

import { Rng } from "../lib/engine/rng.js";
import { CANDIDATES } from "../lib/killfield-runtime/src/killfield/score.js";
import {
  AIM_SELF_OFFSET, BULLET_DIM, BULLET_OFFSET, BULLET_SLOTS, DODGE_OFFSET, MAP_DIM, OBS_DIM,
  OBS_SCHEMA, SeatHistory, encodeObservation, inflatedBoxes,
} from "./duel-env.js";

const BULLET_END = BULLET_OFFSET + BULLET_SLOTS * BULLET_DIM;

function decode(tensor) {
  const binary = typeof Buffer !== "undefined"
    ? Buffer.from(tensor.data, "base64")
    : Uint8Array.from(atob(tensor.data), (c) => c.charCodeAt(0));
  const bytes = new Uint8Array(binary.byteLength);
  bytes.set(binary);
  return { shape: tensor.shape, data: new Float32Array(bytes.buffer) };
}

/** y = W x + b for a PyTorch Linear (W is [out, in], row-major). */
function linear(layer, x, xOffset = 0, out = new Float32Array(layer.out)) {
  const { w, b, out: rows, in: cols } = layer;
  for (let r = 0; r < rows; r++) {
    let sum = b[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) sum += w[base + c] * x[xOffset + c];
    out[r] = sum;
  }
  return out;
}

const relu = (v) => { for (let i = 0; i < v.length; i++) if (v[i] < 0) v[i] = 0; return v; };

export class PpoNetwork {
  constructor(meta, tensors) {
    if (meta.schema !== OBS_SCHEMA) {
      throw new Error(`PPO model schema ${meta.schema} != observation schema ${OBS_SCHEMA}`);
    }
    this.meta = meta;
    const layer = (name) => {
      const w = decode(tensors[`${name}.weight`]);
      const b = decode(tensors[`${name}.bias`]);
      return { w: w.data, b: b.data, out: w.shape[0], in: w.shape[1] };
    };
    this.map0 = layer("map.0");
    this.map2 = layer("map.2");
    this.bul0 = layer("bullets.0");
    this.bul2 = layer("bullets.2");
    this.bulOut = layer("bullet_out.0");
    this.sc0 = layer("scalars.0");
    this.sc2 = layer("scalars.2");
    this.trunk = layer("trunk.0");
    this.actor = layer("actor");
    this.scalars = new Float32Array(OBS_DIM - MAP_DIM - BULLET_SLOTS * BULLET_DIM);
    this.pooled = new Float32Array(2 * this.bul2.out);
    this.joined = new Float32Array(this.map2.out + this.bulOut.out + this.sc2.out);
  }

  /** Action logits for one observation. */
  logits(obs, mask) {
    const m = relu(linear(this.map2, relu(linear(this.map0, obs, 0))));

    const width = this.bul2.out;
    const pooled = this.pooled.fill(0);
    let count = 0;
    for (let slot = 0; slot < BULLET_SLOTS; slot++) {
      if (!mask[slot]) continue;
      const h = relu(linear(this.bul2, relu(linear(this.bul0, obs, BULLET_OFFSET + slot * BULLET_DIM))));
      for (let i = 0; i < width; i++) {
        pooled[i] += h[i];
        pooled[width + i] = count === 0 ? h[i] : Math.max(pooled[width + i], h[i]);
      }
      count += 1;
    }
    if (count > 0) for (let i = 0; i < width; i++) pooled[i] /= count;
    const b = relu(linear(this.bulOut, pooled));

    const sc = this.scalars;
    sc.set(obs.subarray(MAP_DIM, BULLET_OFFSET), 0);
    sc.set(obs.subarray(BULLET_END, OBS_DIM), BULLET_OFFSET - MAP_DIM);
    const s0 = linear(this.sc0, sc);
    for (let i = 0; i < s0.length; i++) s0[i] = Math.tanh(s0[i]);
    const s = relu(linear(this.sc2, s0));

    const joined = this.joined;
    joined.set(m, 0);
    joined.set(b, m.length);
    joined.set(s, m.length + b.length);
    return linear(this.actor, relu(linear(this.trunk, joined)));
  }
}

/**
 * Safety shield over the policy's action distribution, in place.
 *
 * Reads only channels the policy already sees: if some movement survives the
 * 24-frame `dodge_safety` rollout, every action whose movement dies is
 * dropped (if none survives, only the longest-lasting movement is kept); and
 * fire is dropped while the current barrel's shot is forecast to hit the
 * shooter. Same rule as `shielded_action(shield="move+fire")` in
 * training/js_league_ppo.py, measured at +8.4 pp true wins vs Laika over 1000
 * paired seeds (77.5% -> 85.9%, McNemar p < 1e-7).
 *
 * Returns true when the policy's most likely action was removed.
 */
export function shieldProbabilities(probs, obs) {
  let best = 0;
  let anySafe = false;
  let longest = -Infinity;
  for (let move = 0; move < 9; move++) {
    const d = obs[DODGE_OFFSET + move];
    if (d >= 0) anySafe = true;
    if (d > longest) longest = d;
  }
  for (let a = 1; a < probs.length; a++) if (probs[a] > probs[best]) best = a;
  const suicide = obs[AIM_SELF_OFFSET + 1] > 0.5;
  let kept = 0;
  const keep = new Array(probs.length);
  for (let a = 0; a < probs.length; a++) {
    const d = obs[DODGE_OFFSET + (a >> 1)];
    keep[a] = (anySafe ? d >= 0 : d === longest) && !(suicide && (a & 1));
    if (keep[a]) kept += probs[a];
  }
  if (kept > 0) {
    for (let a = 0; a < probs.length; a++) probs[a] = keep[a] ? probs[a] / kept : 0;
  }
  return !keep[best];
}

let cached = null;
let loading = null;

/** Load (once) and return the exported network. */
export async function loadPpoNetwork() {
  if (cached) return cached;
  loading ??= import("./models/ppo-league.js").then(({ PPO_META, PPO_TENSORS }) => {
    cached = new PpoNetwork(PPO_META, PPO_TENSORS);
    return cached;
  });
  return loading;
}

function poses(game) {
  return game.tanks.map((t) => [t.x, t.y, t.rotation]);
}

export class PpoLeagueAgent {
  constructor({ seed = 0, shield = true } = {}) {
    this.rng = new Rng(seed >>> 0);
    this.shield = shield;
    this.overrides = 0;
    this.obs = new Float32Array(OBS_DIM);
    this.mask = new Uint8Array(BULLET_SLOTS);
    this.lastDecisionKind = "ppo";
    this.decisions = [];
    this.round = null;
    this.lastOverridden = false;
    if (!cached) loadPpoNetwork().catch(() => {});
  }

  reset() {
    this.round = null;
  }

  _startRound(game) {
    this.round = game.roundNumber;
    this.history = new SeatHistory();
    this.frames = 0;
    this.prevPose = poses(game);
    this.boxes = inflatedBoxes(game);
  }

  /**
   * The policy's (shielded) action distribution for this frame, or null while
   * the weights are still loading. Reads the history of what tank 0 actually
   * did, so a controller that overrides the policy must still `commit()`.
   */
  policy(game) {
    if (this.round !== game.roundNumber) this._startRound(game);
    if (!cached) return null;
    encodeObservation(game, 0, this.prevPose, this.boxes, this.history, this.frames,
      this.obs, 0, this.mask, 0);
    const probs = cached.logits(this.obs, this.mask);
    let max = -Infinity;
    for (const v of probs) if (v > max) max = v;
    let total = 0;
    for (let i = 0; i < probs.length; i++) { probs[i] = Math.exp(probs[i] - max); total += probs[i]; }
    for (let i = 0; i < probs.length; i++) probs[i] /= total;
    this.lastOverridden = this.shield && shieldProbabilities(probs, this.obs);
    if (this.lastOverridden) this.overrides += 1;
    return probs;
  }

  sample(probs) {
    let u = this.rng.random();
    let action = -1;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] <= 0) continue;
      action = i; // the last kept action is the rounding fallback
      u -= probs[i];
      if (u < 0) break;
    }
    return action;
  }

  /**
   * Record that tank 0 takes `action` this frame, writing its buttons unless
   * `write` is false (a controller that already wrote them). Same order as
   * DuelEpisode.step(): record, write, then remember the pre-step poses the
   * next observation differences against.
   */
  commit(game, action, write = true) {
    this.history.record(action);
    this.frames += 1;
    const me = game.tanks[0];
    if (write && me.alive) {
      const [throttle, turn, fire] = CANDIDATES[action];
      me.forward = throttle === 2;
      me.backup = throttle === 0;
      me.turnLeft = turn === 0;
      me.turnRight = turn === 2;
      me.fire = fire === 1;
    }
    this.prevPose = poses(game);
  }

  drive(game) {
    const started = performance.now();
    const probs = this.policy(game);
    if (!probs) {
      const me = game.tanks[0];
      me.forward = me.backup = me.turnLeft = me.turnRight = me.fire = false;
      this.lastDecisionKind = "ppo_loading";
      return;
    }
    this.commit(game, this.choose(game, probs));
    this.lastDecisionKind = this.lastOverridden ? "ppo_shield" : "ppo";
    this.decisions.push(performance.now() - started);
    if (this.decisions.length > 200) this.decisions.shift();
  }

  /** Pick this frame's action from the policy's distribution. Subclasses may search. */
  choose(game, probs) {
    return this.sample(probs);
  }

  telemetry() {
    const sorted = this.decisions.slice().sort((a, b) => a - b);
    const at = (q) => (sorted.length ? sorted[Math.floor(q * (sorted.length - 1))] : 0);
    return {
      decision: this.lastDecisionKind,
      planMedianMs: at(0.5),
      planP95Ms: at(0.95),
      model: cached ? `${cached.meta.run}/${cached.meta.checkpoint}` : "loading",
      shieldOverrides: this.overrides,
    };
  }
}
