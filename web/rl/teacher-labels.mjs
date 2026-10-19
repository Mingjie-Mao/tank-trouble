#!/usr/bin/env node
/**
 * Teacher labels for distilling PPO-MPC into the network.
 *
 *   node rl/teacher-labels.mjs <outDir> <seedStart> <rounds> [threads] [--k 4 --m 4 --h 24] [--calm 0.25]
 *
 * The student (the exported `ppo-league` network with its shield) plays
 * native Laika and its own sampled actions are executed, so the states are
 * the ones the student actually reaches (DAgger). On every danger frame the
 * MPC teacher scores the policy's top actions by rollouts against the L2
 * sandbox; a fraction of calm frames is kept unlabelled to anchor the rest of
 * the policy. Per worker, raw little-endian arrays:
 *
 *   obs f32[N][OBS_DIM]  mask u8[N][BULLET_SLOTS]  probs f32[N][18] (shielded)
 *   scores f32[N][18] (mean rollout outcome, NaN where unscored)  danger u8[N]
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { join } from "node:path";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

const ACTIONS = 18;

async function label({ seeds, k, m, h, calm }) {
  const { BULLET_SLOTS, DuelEpisode, OBS_DIM, OpponentKind } = await import("./duel-env.js");
  const { loadPpoNetwork } = await import("./ppo-agent.js");
  const { PpoMpcAgent } = await import("./ppo-mpc.js");
  const { Rng } = await import("../lib/engine/rng.js");
  await loadPpoNetwork();
  const rows = { obs: [], mask: [], probs: [], scores: [], danger: [] };
  const tally = { win: 0, other: 0, labelled: 0, unlabelled: 0 };
  for (const seed of seeds) {
    const ep = new DuelEpisode(seed, OpponentKind.LAIKA);
    const agent = new PpoMpcAgent({ seed: seed + 101, candidates: k, rollouts: m, horizon: h });
    const keep = new Rng((seed ^ 0x2545f491) >>> 0);
    while (!ep.outcome) {
      if (!ep.game.frozen) {
        const probs = agent.policy(ep.game);
        const action = agent.sample(probs);
        const danger = agent.danger(ep.game);
        if (danger || keep.random() < calm) {
          const scores = new Float32Array(ACTIONS).fill(NaN);
          if (danger) {
            for (const [a, s] of agent.evaluate(ep.game, probs, action)) scores[a] = s / m;
          }
          rows.obs.push(Float32Array.from(agent.obs));
          rows.mask.push(Uint8Array.from(agent.mask));
          rows.probs.push(Float32Array.from(probs));
          rows.scores.push(scores);
          rows.danger.push(danger ? 1 : 0);
          tally[danger ? "labelled" : "unlabelled"] += 1;
        }
        agent.commit(ep.game, action);
      }
      ep.stepControlled();
    }
    tally[ep.outcome === 1 ? "win" : "other"] += 1;
  }
  const flat = (list, Type, width) => {
    const out = new Type(list.length * width);
    list.forEach((row, i) => out.set(row, i * width));
    return out;
  };
  return {
    tally,
    obs: flat(rows.obs, Float32Array, OBS_DIM),
    mask: flat(rows.mask, Uint8Array, BULLET_SLOTS),
    probs: flat(rows.probs, Float32Array, ACTIONS),
    scores: flat(rows.scores, Float32Array, ACTIONS),
    danger: Uint8Array.from(rows.danger),
  };
}

if (!isMainThread) {
  label(workerData).then((r) => parentPort.postMessage(r));
} else {
  const args = process.argv.slice(2);
  const opt = (name, fallback) => {
    const at = args.indexOf(`--${name}`);
    return at >= 0 ? Number(args.splice(at, 2)[1]) : fallback;
  };
  const k = opt("k", 4);
  const m = opt("m", 4);
  const h = opt("h", 24);
  const calm = opt("calm", 0.25);
  const [outDir, seedStart, rounds, threads] = args;
  if (!outDir || !seedStart || !rounds) {
    console.error("usage: teacher-labels.mjs <outDir> <seedStart> <rounds> [threads] [--k --m --h --calm]");
    process.exit(2);
  }
  const n = Number(rounds);
  const t = Math.min(n, Number(threads ?? Math.max(1, availableParallelism() - 1)));
  const seeds = Array.from({ length: n }, (_, i) => Number(seedStart) + i);
  mkdirSync(outDir, { recursive: true });
  const started = Date.now();
  const parts = await Promise.all(Array.from({ length: t }, (_, w) => new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { seeds: seeds.filter((_, i) => i % t === w), k, m, h, calm },
    });
    worker.once("message", (r) => {
      for (const key of ["obs", "mask", "probs", "scores", "danger"]) {
        writeFileSync(join(outDir, `w${w}.${key}.bin`), Buffer.from(r[key].buffer));
      }
      resolve(r.tally);
    });
    worker.once("error", reject);
  })));
  const tally = parts.reduce((a, b) => Object.fromEntries(Object.keys(b).map((key) => [key, (a[key] ?? 0) + b[key]])), {});
  const { loadPpoNetwork } = await import("./ppo-agent.js");
  const net = await loadPpoNetwork();
  const meta = {
    seedStart: Number(seedStart), rounds: n, workers: t, k, m, h, calm, ...tally,
    student: `${net.meta.run}/${net.meta.checkpoint}`, schema: net.meta.schema,
    seconds: (Date.now() - started) / 1000,
  };
  writeFileSync(join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
  console.log(JSON.stringify(meta));
}
