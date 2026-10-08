#!/usr/bin/env node
/**
 * A batch of duel environments behind a binary stdin/stdout protocol, driven
 * in lockstep by `training/js_league_ppo.py`.
 *
 * Requests (little-endian), each starting with a u8 command:
 *   1 INIT    u32 length + JSON {envs, seed, weights, wallSliding}  -> STEP reply
 *   2 STEP    u8[E] learner actions, u8[E] frozen-opponent actions  -> STEP reply
 *   3 WEIGHTS u32 length + JSON {weights}                           -> u8 ack
 *   4 CLOSE
 *
 * `weights[k]` is the chance a new episode's opponent is slot k: 0 Laika,
 * 1 Tactical, k >= 2 the trainer's frozen checkpoint k - 2. Episodes reset
 * themselves: an env that just finished reports the finished episode's
 * result and the new episode's first observation in the same reply.
 *
 * STEP reply, sections in this order so every one is naturally aligned
 * (seat 1 is only filled when the opponent is a frozen checkpoint):
 *   f32 rest[2][E][OBS_DIM - MAP_DIM]   f32 reward[E]
 *   i16 slot[E]  (the episode now running)   i16 endedSlot[E]
 *   u16 frames[E], shots[E], hits[E]     (of the episode that just ended)
 *   u8  map[2][E][MAP_DIM]   u8 mask[2][E][BULLET_SLOTS]
 *   u8  done[E]   u8 outcome[E]
 */

import { readSync, writeSync } from "node:fs";
import {
  BULLET_SLOTS, DuelEpisode, MAP_DIM, OBS_DIM, OpponentKind,
} from "./duel-env.js";
import { Rng } from "../lib/engine/rng.js";

const REST = OBS_DIM - MAP_DIM;

function readExactly(n) {
  const buf = Buffer.allocUnsafe(n);
  let got = 0;
  while (got < n) {
    let r;
    try {
      r = readSync(0, buf, got, n - got, null);
    } catch (error) {
      if (error.code === "EAGAIN") continue;
      throw error;
    }
    if (r === 0) process.exit(0);
    got += r;
  }
  return buf;
}

function writeAll(bytes) {
  let sent = 0;
  while (sent < bytes.length) {
    try {
      sent += writeSync(1, bytes, sent, bytes.length - sent);
    } catch (error) {
      if (error.code !== "EAGAIN") throw error;
    }
  }
}

function readJson() {
  const len = readExactly(4).readUInt32LE(0);
  return JSON.parse(readExactly(len).toString("utf8"));
}

/** Typed views over one reply buffer, laid out as documented above. */
function replyLayout(n) {
  const sizes = [
    ["rest", Float32Array, 2 * n * REST], ["reward", Float32Array, n],
    ["slot", Int16Array, n], ["endedSlot", Int16Array, n],
    ["frames", Uint16Array, n], ["shots", Uint16Array, n], ["hits", Uint16Array, n],
    ["map", Uint8Array, 2 * n * MAP_DIM], ["mask", Uint8Array, 2 * n * BULLET_SLOTS],
    ["done", Uint8Array, n], ["outcome", Uint8Array, n],
  ];
  const bytes = sizes.reduce((sum, [, T, len]) => sum + T.BYTES_PER_ELEMENT * len, 0);
  const buffer = new ArrayBuffer(bytes);
  const views = { bytes: new Uint8Array(buffer) };
  let at = 0;
  for (const [name, T, len] of sizes) {
    views[name] = new T(buffer, at, len);
    at += T.BYTES_PER_ELEMENT * len;
  }
  return views;
}

class Batch {
  constructor({ envs, seed, weights, wallSliding = false }) {
    this.n = envs;
    this.rng = new Rng(seed >>> 0);
    this.nextSeed = seed >>> 0;
    this.weights = weights;
    this.wallSliding = wallSliding;
    this.episodes = new Array(envs);
    this.slots = new Int16Array(envs);
    this.obs = new Float32Array(OBS_DIM);
    this.mask = new Uint8Array(BULLET_SLOTS);
    this.reply = replyLayout(envs);
    for (let i = 0; i < envs; i++) this.reset(i);
  }

  sampleSlot() {
    const total = this.weights.reduce((a, b) => a + b, 0);
    let u = this.rng.random() * total;
    for (let k = 0; k < this.weights.length; k++) {
      u -= this.weights[k];
      if (u < 0) return k;
    }
    return this.weights.length - 1;
  }

  reset(i) {
    const slot = this.sampleSlot();
    const kind = slot === 0 ? OpponentKind.LAIKA
      : slot === 1 ? OpponentKind.TACTICAL : OpponentKind.FROZEN;
    // Consecutive seeds from this worker's own range: every episode a new maze.
    this.nextSeed = (this.nextSeed + 1) >>> 0;
    this.episodes[i] = new DuelEpisode(this.nextSeed, kind, { wallSliding: this.wallSliding });
    this.slots[i] = slot;
  }

  writeSeat(i, seat) {
    const { rest, map, mask } = this.reply;
    const row = seat * this.n + i;
    const ep = this.episodes[i];
    if (seat === 1 && ep.opponent !== OpponentKind.FROZEN) {
      rest.fill(0, row * REST, (row + 1) * REST);
      map.fill(0, row * MAP_DIM, (row + 1) * MAP_DIM);
      mask.fill(0, row * BULLET_SLOTS, (row + 1) * BULLET_SLOTS);
      return;
    }
    ep.observe(seat, this.obs, 0, this.mask, 0);
    map.set(this.obs.subarray(0, MAP_DIM), row * MAP_DIM); // channels are exactly 0/1
    rest.set(this.obs.subarray(MAP_DIM), row * REST);
    mask.set(this.mask, row * BULLET_SLOTS);
  }

  /** Write every env's observations and current slot into the reply. */
  finish() {
    for (let i = 0; i < this.n; i++) {
      this.writeSeat(i, 0);
      this.writeSeat(i, 1);
    }
    this.reply.slot.set(this.slots);
    return this.reply.bytes;
  }

  initial() {
    const r = this.reply;
    for (const key of ["reward", "endedSlot", "frames", "shots", "hits", "done", "outcome"]) {
      r[key].fill(0);
    }
    return this.finish();
  }

  step(actions, opponentActions) {
    const r = this.reply;
    for (let i = 0; i < this.n; i++) {
      const ep = this.episodes[i];
      r.reward[i] = ep.step(actions[i],
        ep.opponent === OpponentKind.FROZEN ? opponentActions[i] : null);
      const ended = ep.outcome !== 0;
      r.done[i] = ended ? 1 : 0;
      r.outcome[i] = ep.outcome;
      r.endedSlot[i] = ended ? this.slots[i] : 0;
      r.frames[i] = ended ? Math.min(65535, ep.frames) : 0;
      r.shots[i] = ended ? ep.shots : 0;
      r.hits[i] = ended ? ep.hits : 0;
      if (ended) this.reset(i);
    }
    return this.finish();
  }
}

let batch = null;
for (;;) {
  const cmd = readExactly(1)[0];
  if (cmd === 1) {
    batch = new Batch(readJson());
    writeAll(batch.initial());
  } else if (cmd === 2) {
    const actions = readExactly(batch.n);
    const opponentActions = readExactly(batch.n);
    writeAll(batch.step(actions, opponentActions));
  } else if (cmd === 3) {
    batch.weights = readJson().weights;
    writeAll(Buffer.from([1]));
  } else {
    process.exit(0);
  }
}
