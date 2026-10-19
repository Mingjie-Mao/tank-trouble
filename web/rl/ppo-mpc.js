/**
 * PPO-MPC: the policy proposes, rollouts of the policy decide.
 *
 * In danger — a barrel lines up, a bullet is on course, or some move is
 * forecast to die — each of the policy's top actions is played out against
 * the L2 sandbox (a fresh Laika and an independent random stream: nothing
 * hidden is read) by the shielded policy itself, and the action whose
 * futures go best is taken. Too slow for the 40 ms frame (about 0.2–0.6 s per
 * search); it is the teacher that rl/teacher-labels.mjs distils from.
 */

import { applyAction, makeSandbox } from "../lib/killfield-runtime/src/killfield/sandbox.js";
import { CANDIDATES } from "../lib/killfield-runtime/src/killfield/score.js";
import { Rng } from "../lib/engine/rng.js";
import {
  AIM_OPPONENT_OFFSET, AIM_SELF_OFFSET, BULLET_SLOTS, DODGE_OFFSET, OBS_DIM, SeatHistory,
  THREAT_OFFSET, encodeObservation,
} from "./duel-env.js";
import { PpoLeagueAgent, loadedPpoNetwork, shieldProbabilities } from "./ppo-agent.js";

function copyHistory(history) {
  const copy = new SeatHistory();
  copy.actions = history.actions.slice();
  copy.changes = history.changes;
  copy.idleStreak = history.idleStreak;
  return copy;
}

const poses = (game) => game.tanks.map((t) => [t.x, t.y, t.rotation]);

export class PpoMpcAgent extends PpoLeagueAgent {
  constructor({ candidates = 4, rollouts = 4, horizon = 32, doubleScore = -0.5, ...options } = {}) {
    super(options);
    this.candidates = candidates;
    this.rollouts = rollouts;
    this.horizon = horizon;
    this.doubleScore = doubleScore;
    this.searches = 0;
    this.switches = 0;
    this.rolloutObs = new Float32Array(OBS_DIM);
    this.rolloutMask = new Uint8Array(BULLET_SLOTS);
  }

  /** From this frame's observation: is an exchange on or a move deadly? */
  danger(game) {
    const obs = this.obs;
    if (!game.tanks[0].alive || !game.tanks[1].alive || game.frozen) return false;
    if (obs[AIM_OPPONENT_OFFSET] > 0.5 || obs[AIM_SELF_OFFSET] > 0.5 || obs[THREAT_OFFSET] > 0) return true;
    for (let move = 0; move < 9; move++) if (obs[DODGE_OFFSET + move] < 0) return true;
    return false;
  }

  /** Play `first`, then the shielded policy, for the horizon in an L2 sandbox. */
  rollout(game, first, seed) {
    const net = loadedPpoNetwork();
    const sb = makeSandbox(game, "L2", seed);
    const rng = new Rng((seed ^ 0x9e3779b9) >>> 0);
    const history = copyHistory(this.history);
    const obs = this.rolloutObs;
    const mask = this.rolloutMask;
    let prevPose = this.prevPose;
    let frames = this.frames;
    const me = sb.tanks[0];
    const them = sb.tanks[1];
    let action = first;
    for (let t = 0; t < this.horizon && !sb.frozen; t++) {
      if (t > 0) {
        if (!me.alive) break;
        encodeObservation(sb, 0, prevPose, this.boxes, history, frames, obs, 0, mask, 0);
        const p = net.logits(obs, mask);
        let max = -Infinity;
        for (const v of p) if (v > max) max = v;
        let total = 0;
        for (let i = 0; i < p.length; i++) { p[i] = Math.exp(p[i] - max); total += p[i]; }
        for (let i = 0; i < p.length; i++) p[i] /= total;
        if (this.shield) shieldProbabilities(p, obs);
        let u = rng.random();
        for (let i = 0; i < p.length; i++) {
          if (p[i] <= 0) continue;
          action = i;
          u -= p[i];
          if (u < 0) break;
        }
      }
      history.record(action);
      frames += 1;
      if (me.alive) applyAction(sb, CANDIDATES[action]);
      prevPose = poses(sb);
      sb.step();
    }
    if (!me.alive) return them.alive ? -1 : this.doubleScore;
    return them.alive ? 0 : 1;
  }

  /**
   * Score the policy's top actions (and `sampled`) by rollouts: a Map from
   * action to the summed outcome over `rollouts` common-random-number futures.
   */
  evaluate(game, probs, sampled) {
    const ranked = Array.from(probs.keys()).filter((a) => probs[a] > 0)
      .sort((x, y) => probs[y] - probs[x]).slice(0, this.candidates);
    if (!ranked.includes(sampled)) ranked.push(sampled);
    // Common random numbers: every candidate faces the same futures.
    const seeds = Array.from({ length: this.rollouts }, () => this.rng.randrange(1 << 30));
    const scores = new Map();
    for (const action of ranked) {
      let score = 0;
      for (const seed of seeds) score += this.rollout(game, action, seed);
      scores.set(action, score);
    }
    return scores;
  }

  choose(game, probs) {
    const sampled = this.sample(probs);
    if (!this.danger(game)) return sampled;
    this.searches += 1;
    const scores = this.evaluate(game, probs, sampled);
    let best = sampled;
    let bestScore = -Infinity;
    for (const [action, score] of scores) {
      if (score > bestScore) { best = action; bestScore = score; }
    }
    // Keep the policy's own choice unless something is strictly better.
    if (bestScore <= scores.get(sampled)) return sampled;
    this.switches += 1;
    return best;
  }

  telemetry() {
    return { ...super.telemetry(), mpcSearches: this.searches, mpcSwitches: this.switches };
  }
}
