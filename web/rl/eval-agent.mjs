#!/usr/bin/env node
/**
 * Grade any browser policy against Laika under the league-PPO protocol, so
 * search and network agents are compared on identical terms.
 *
 *   node rl/eval-agent.mjs <policy> <seedStart> <count> [threads] [--wall-sliding] [--out file]
 *
 * One round per seed (seed, seed+1, ...), the same maze and spawns
 * `training/js_league_ppo.py eval` uses for that seed; the policy drives tank
 * 0 and native Laika tank 1; a round both tanks survive for 30 s is a draw;
 * only surviving the settlement window counts as a win. Each round's
 * bullet-tank hits are kept for failure attribution.
 */

import { writeFileSync } from "node:fs";
import { availableParallelism } from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";

const OUTCOMES = ["running", "win", "loss", "double", "draw"];

async function playSeeds({ policy, seeds, wallSliding }) {
  const { DuelEpisode, OpponentKind } = await import("./duel-env.js");
  const { makeAgent } = await import("../lib/browser-arena.js");
  if (policy.startsWith("ppo-")) {
    const { loadPpoNetwork } = await import("./ppo-agent.js");
    await loadPpoNetwork();
  }
  const rows = [];
  for (const seed of seeds) {
    const ep = new DuelEpisode(seed, OpponentKind.LAIKA, { wallSliding });
    // The arena builds a fresh agent per round with seed + 101 and, against
    // native Laika, the L2 opponent model; mirror that exactly.
    const agent = makeAgent(policy, seed + 101, "L2");
    const ms = [];
    while (!ep.outcome) {
      if (!ep.game.frozen) {
        const started = performance.now();
        agent.drive(ep.game);
        ms.push(performance.now() - started);
      }
      ep.stepControlled();
    }
    ms.sort((a, b) => a - b);
    rows.push({
      seed, outcome: OUTCOMES[ep.outcome], frames: ep.frames, shots: ep.shots,
      kills: ep.kills, p50: ms[ms.length >> 1] ?? 0, p95: ms[Math.floor(ms.length * 0.95)] ?? 0,
    });
  }
  return rows;
}

if (!isMainThread) {
  playSeeds(workerData).then((rows) => parentPort.postMessage(rows));
} else {
  const args = process.argv.slice(2);
  const outAt = args.indexOf("--out");
  const out = outAt >= 0 ? args.splice(outAt, 2)[1] : null;
  const wallSliding = args.includes("--wall-sliding");
  const [policy, seedStart, count, threads] = args.filter((a) => a !== "--wall-sliding");
  if (!policy || !seedStart || !count) {
    console.error("usage: eval-agent.mjs <policy> <seedStart> <count> [threads] [--wall-sliding] [--out file]");
    process.exit(2);
  }
  const n = Number(count);
  const k = Math.min(n, Number(threads ?? Math.max(1, availableParallelism() - 1)));
  const seeds = Array.from({ length: n }, (_, i) => Number(seedStart) + i);
  const started = Date.now();
  const parts = await Promise.all(Array.from({ length: k }, (_, t) => new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { policy, seeds: seeds.filter((_, i) => i % k === t), wallSliding },
    });
    worker.once("message", resolve);
    worker.once("error", reject);
  })));
  const rows = parts.flat().sort((a, b) => a.seed - b.seed);
  const tally = Object.fromEntries(OUTCOMES.slice(1).map((o) => [o, 0]));
  for (const row of rows) tally[row.outcome] += 1;
  const p = tally.win / rows.length;
  const se = Math.sqrt(p * (1 - p) / rows.length);
  const perRound = (key) => rows.map((r) => r[key]).sort((a, b) => a - b)[rows.length >> 1];
  const summary = {
    policy, seedStart: Number(seedStart), games: rows.length, wallSliding, ...tally,
    winRate: p, ci95: [p - 1.96 * se, p + 1.96 * se],
    meanFrames: rows.reduce((s, r) => s + r.frames, 0) / rows.length,
    medianRoundP50Ms: perRound("p50"), medianRoundP95Ms: perRound("p95"),
    seconds: (Date.now() - started) / 1000,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (out) writeFileSync(out, JSON.stringify({ ...summary, rows }));
}
