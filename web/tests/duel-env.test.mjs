import assert from "node:assert/strict";
import test from "node:test";

import * as C from "../lib/engine/constants.js";
import { Game } from "../lib/engine/game.js";
import {
  ACTIONS, BULLET_OFFSET, BULLET_SLOTS, BULLET_DIM, DuelEpisode,
  DUEL_FRAMES, OBS_DIM, OpponentKind, Outcome, PHASE_OFFSET, SELF_THREAT_COUNT_OFFSET,
  dodgeSafety, dodgeSafetyReference, outcomeReward, styleBonus, turnsToHit, winReward,
} from "../rl/duel-env.js";

function lcg(seed) {
  let s = seed;
  return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
}

/** Play `episodes` rounds with a sticky random policy, calling `visit` every frame. */
function playRandom({ seeds, kind, visit }) {
  const random = lcg(17);
  for (const seed of seeds) {
    const ep = new DuelEpisode(seed, kind);
    let action = 0;
    while (!ep.outcome) {
      if (random() < 0.2) action = Math.floor(random() * ACTIONS);
      const reward = ep.step(action, kind === OpponentKind.FROZEN
        ? Math.floor(random() * ACTIONS) : null);
      visit(ep, reward);
    }
  }
}

test("the observation is 1030 values, all finite and inside [-1, 1]", () => {
  assert.equal(OBS_DIM, 1030);
  const obs = new Float32Array(OBS_DIM);
  const mask = new Uint8Array(BULLET_SLOTS);
  playRandom({
    seeds: [1, 2, 3, 4], kind: OpponentKind.FROZEN,
    visit(ep) {
      for (const seat of [0, 1]) {
        ep.observe(seat, obs, 0, mask);
        for (let i = 0; i < OBS_DIM; i++) {
          assert.ok(Number.isFinite(obs[i]) && obs[i] >= -1 && obs[i] <= 1,
            `seat ${seat} channel ${i} = ${obs[i]} at frame ${ep.frames}`);
        }
        const phase = obs[PHASE_OFFSET] + obs[PHASE_OFFSET + 1] + obs[PHASE_OFFSET + 2];
        assert.equal(phase, 1, "phase must be one-hot");
      }
    },
  });
});

/** turnsToHit spelled out with the engine's own hit tests and a real barrel turn. */
function turnsToHitReference(game, shooter, target) {
  if (!shooter.alive || !target.alive) return -1;
  const hits = (angle) => {
    const saved = shooter.rotation;
    shooter.rotation = angle;
    const rad = ((angle - 90) * Math.PI) / 180;
    const scale = game.scale;
    let x = shooter.x + Math.cos(rad) * scale * 4.5 / 16;
    let y = shooter.y + Math.sin(rad) * scale * 4.5 / 16;
    let xs = Math.cos(rad) * C.BULLETSPEED * (scale / 50);
    let ys = Math.sin(rad) * C.BULLETSPEED * (scale / 50);
    let result = false;
    for (let life = C.BULLETLIFETIME / 3; life > 0; life -= 1) {
      const px = x;
      const py = y;
      x += xs;
      y += ys;
      if (game.wallHit(x, y)) {
        const a = game.wallHit(px - xs, py + ys);
        const b = game.wallHit(px + xs, py - ys);
        if (a && !b) ys = -ys;
        else if (b && !a) xs = -xs;
        else { xs = -xs; ys = -ys; }
        x = px + xs;
        y = py + ys;
      }
      if (target.pointInBbox(x, y) && target.pointInShape(x, y)) { result = true; break; }
      if (shooter.pointInBbox(x, y) && shooter.pointInShape(x, y)) break;
    }
    shooter.rotation = saved;
    return result;
  };
  for (let k = 0; k <= 18; k++) {
    if (hits(shooter.rotation - 10 * k)) return k;
    if (k > 0 && k < 18 && hits(shooter.rotation + 10 * k)) return k;
  }
  return -1;
}

test("first strike matches a turn-and-shoot reference, both seats", () => {
  const seen = new Set();
  playRandom({
    seeds: [21, 22, 23, 24, 25], kind: OpponentKind.LAIKA,
    visit(ep) {
      const [a, b] = ep.game.tanks;
      for (const [s, t] of [[a, b], [b, a]]) {
        const got = turnsToHit(ep.game, s, t);
        assert.equal(got, turnsToHitReference(ep.game, s, t), `frame ${ep.frames}`);
        seen.add(got);
      }
    },
  });
  assert.ok(seen.has(0) && seen.has(-1) && seen.size > 8, `values seen: ${[...seen]}`);
});

test("fast dodgeSafety is bit-identical to the nine-rollout reference", () => {
  const fast = new Float32Array(9);
  const slow = new Float32Array(9);
  let compared = 0;
  playRandom({
    seeds: [11, 12, 13, 14, 15, 16], kind: OpponentKind.FROZEN,
    visit(ep) {
      for (const seat of [0, 1]) {
        dodgeSafety(ep.game, seat, fast, 0);
        dodgeSafetyReference(ep.game, seat, slow, 0);
        assert.deepEqual(Array.from(fast), Array.from(slow),
          `seat ${seat} frame ${ep.frames}`);
        compared += 1;
      }
    },
  });
  assert.ok(compared > 200, `only ${compared} states compared`);
});

test("the engine's wall grid answers exactly like testing every wall", () => {
  // The bucketed index must equal the brute-force definition: a point hits a
  // wall when it lies in some segment's box grown by the half thickness.
  const random = lcg(5);
  for (let seed = 1; seed <= 20; seed++) {
    const game = new Game({ seed });
    const t = game.wallHalfT;
    const brute = (x, y) => game.walls.some(([x1, y1, x2, y2]) => (
      Math.min(x1, x2) - t <= x && x <= Math.max(x1, x2) + t
      && Math.min(y1, y2) - t <= y && y <= Math.max(y1, y2) + t));
    for (let i = 0; i < 5000; i++) {
      const x = random() * 760 - 30;
      const y = random() * 520 - 30;
      assert.equal(game.wallHit(x, y), brute(x, y), `seed ${seed} at ${x},${y}`);
    }
  }
});

test("my own unbounced round is never flagged as a threat to me", () => {
  const ep = new DuelEpisode(19, OpponentKind.FROZEN);
  ep.step(9, 8); // stand still and fire; the opponent stands still
  const obs = new Float32Array(OBS_DIM);
  const mask = new Uint8Array(BULLET_SLOTS);
  ep.observe(0, obs, 0, mask);
  assert.ok(mask[0], "the shot should occupy slot 0");
  const base = BULLET_OFFSET;
  assert.equal(obs[base + 4], 1, "the bullet is mine");
  assert.equal(obs[base + 5], 0, "and has not bounced");
  assert.equal(obs[base + 7], 0, "so it cannot be coming for me");
  assert.equal(obs[SELF_THREAT_COUNT_OFFSET], 0);
  assert.equal(BULLET_SLOTS * BULLET_DIM, 100);
});

test("nothing is paid before the round's true end", () => {
  let terminals = 0;
  playRandom({
    seeds: [21, 22, 23], kind: OpponentKind.LAIKA,
    visit(ep, reward) {
      if (ep.outcome === Outcome.RUNNING) {
        assert.equal(reward, 0, `frame ${ep.frames} paid ${reward}`);
      } else {
        terminals += 1;
        assert.ok(ep.frames <= DUEL_FRAMES + 125);
        assert.ok(Math.abs(reward - (outcomeReward(ep.outcome, ep.frames)
          + styleBonus(ep.history[0].changes, ep.frames))) < 1e-12);
      }
    },
  });
  assert.equal(terminals, 3);
});

test("the reward scale orders win > trade > loss = draw", () => {
  assert.equal(winReward(0), 1);
  assert.equal(winReward(250), 1);
  assert.ok(Math.abs(winReward(DUEL_FRAMES) - 0.5) < 1e-12);
  assert.ok(winReward(400) < winReward(300));
  const best = styleBonus(0, 300);
  assert.equal(best, 0.25);
  assert.ok(styleBonus(299, 300) < 1e-12, "changing every frame earns nothing");
  assert.ok(winReward(DUEL_FRAMES) > outcomeReward(Outcome.DOUBLE_DEATH, 0) + best);
  assert.ok(outcomeReward(Outcome.DOUBLE_DEATH, 0) > outcomeReward(Outcome.LOSS, 0) + best);
  assert.equal(outcomeReward(Outcome.DRAW, 0), outcomeReward(Outcome.LOSS, 0));
});

test("the browser PPO forward pass reproduces PyTorch's logits", async () => {
  const { readFileSync } = await import("node:fs");
  const { loadPpoNetwork } = await import("../rl/ppo-agent.js");
  const net = await loadPpoNetwork();
  const parity = JSON.parse(readFileSync(new URL("../rl/models/ppo-league.parity.json", import.meta.url)));
  parity.obs.forEach((obs, row) => {
    const logits = net.logits(Float32Array.from(obs), Uint8Array.from(parity.mask[row]));
    logits.forEach((value, action) => {
      assert.ok(Math.abs(value - parity.logits[row][action]) < 1e-4,
        `row ${row} action ${action}: ${value} vs ${parity.logits[row][action]}`);
    });
  });
});

test("the PPO safety shield removes movements dodge_safety predicts die", async () => {
  const { shieldProbabilities } = await import("../rl/ppo-agent.js");
  const { AIM_SELF_OFFSET, DODGE_OFFSET } = await import("../rl/duel-env.js");
  const obs = new Float32Array(OBS_DIM);
  const uniform = () => new Float64Array(ACTIONS).fill(1 / ACTIONS);

  // Moves 0..3 die, 4..8 survive: every action of a dying move is dropped.
  for (let m = 0; m < 9; m++) obs[DODGE_OFFSET + m] = m < 4 ? -0.8 : 0.5;
  let probs = uniform();
  shieldProbabilities(probs, obs);
  for (let a = 0; a < ACTIONS; a++) assert.equal(probs[a] > 0, (a >> 1) >= 4, `action ${a}`);
  assert.ok(Math.abs(probs.reduce((x, y) => x + y, 0) - 1) < 1e-12);

  // Nothing survives: keep only the movement that lasts longest.
  for (let m = 0; m < 9; m++) obs[DODGE_OFFSET + m] = m === 6 ? -0.2 : -0.9;
  probs = uniform();
  shieldProbabilities(probs, obs);
  for (let a = 0; a < ACTIONS; a++) assert.equal(probs[a] > 0, (a >> 1) === 6, `action ${a}`);

  // A shot forecast to hit the shooter drops every fire action.
  for (let m = 0; m < 9; m++) obs[DODGE_OFFSET + m] = 1;
  obs[AIM_SELF_OFFSET + 1] = 1;
  probs = uniform();
  const overridden = shieldProbabilities(probs, obs);
  for (let a = 0; a < ACTIONS; a++) assert.equal(probs[a] > 0, (a & 1) === 0, `action ${a}`);
  assert.equal(overridden, false, "the argmax of a uniform distribution is action 0, a no-fire action");
});

test("a run can price the mutual kill without moving the rest of the scale", () => {
  assert.equal(outcomeReward(Outcome.DOUBLE_DEATH, 100), -0.1);
  assert.equal(outcomeReward(Outcome.DOUBLE_DEATH, 100, -0.6), -0.6);
  assert.equal(outcomeReward(Outcome.LOSS, 100, -0.6), -1);
  assert.equal(outcomeReward(Outcome.WIN, 100, -0.6), 1);
  const ep = new DuelEpisode(1, OpponentKind.LAIKA, { doubleDeathReward: -0.6 });
  assert.equal(ep.doubleDeathReward, -0.6);
});
