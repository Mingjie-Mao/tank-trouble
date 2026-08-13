import { writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";

const rounds = Number(process.argv[2] ?? 60);
const seed = Number(process.argv[3] ?? 970000);
const maxFrames = Number(process.argv[4] ?? 3000);
const policy = process.argv[5] ?? "killfield-js";
const output = process.argv[6] ?? null;
const counts = { win: 0, loss: 0, double_death: 0, draw: 0 };
let totalFrames = 0;
const decisionMs = [];
const games = [];
const started = performance.now();

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
}

for (let index = 0; index < rounds; index += 1) {
  const arena = new BrowserArena({ seed: seed + index });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  let result = "draw";
  let frames = 0;
  for (; frames < maxFrames; frames += 1) {
    arena.step(frames * 40);
    if (arena.lastDecisionMs[0] > 0) decisionMs.push(arena.lastDecisionMs[0]);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      result = roundEnd[1] === 0 ? "win" : roundEnd[1] === 1 ? "loss" : "double_death";
      frames += 1;
      break;
    }
  }
  counts[result] += 1;
  totalFrames += frames;
  games.push({ seed: seed + index, result, frames });
  if ((index + 1) % 10 === 0 || index + 1 === rounds) {
    console.log(`  ${index + 1}/${rounds}  wins ${counts.win}`);
  }
}

const elapsed = (performance.now() - started) / 1000;
const pct = (value) => `${(100 * value / rounds).toFixed(1)}%`;
console.log(`===== Browser JS ${policy}: ${rounds} games @${seed} (${elapsed.toFixed(1)}s) =====`);
console.log(`  true win ${pct(counts.win)}  loss ${pct(counts.loss)}  double death ${pct(counts.double_death)}  draw ${pct(counts.draw)}`);
console.log(`  avg length ${(totalFrames / rounds / 25).toFixed(1)}s  throughput ${(totalFrames / elapsed).toFixed(0)} frames/s`);
console.log(`  decision p50 ${percentile(decisionMs, 0.5).toFixed(3)}ms  p95 ${percentile(decisionMs, 0.95).toFixed(3)}ms  p99 ${percentile(decisionMs, 0.99).toFixed(3)}ms`);

if (output !== null) {
  const report = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    policy,
    rounds,
    seed,
    maxFrames,
    counts,
    winRate: counts.win / rounds,
    doubleDeathRate: counts.double_death / rounds,
    drawRate: counts.draw / rounds,
    averageFrames: totalFrames / rounds,
    decisionP50Ms: percentile(decisionMs, 0.5),
    decisionP95Ms: percentile(decisionMs, 0.95),
    decisionP99Ms: percentile(decisionMs, 0.99),
    elapsedSeconds: elapsed,
    games,
  };
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  console.log(`  wrote ${output}`);
}
