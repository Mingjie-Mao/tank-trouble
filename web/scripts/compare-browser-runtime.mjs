import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";

const rounds = Number(process.argv[2] ?? 20);
const seed = Number(process.argv[3] ?? 970000);
const challenger = process.argv[4] ?? "p27-js-shield";
const maxFrames = Number(process.argv[5] ?? 3000);
const baselinePolicy = process.argv[6] ?? "killfield-js";

function play(policy, roundSeed) {
  const arena = new BrowserArena({ seed: roundSeed });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  for (let frame = 0; frame < maxFrames; frame += 1) {
    arena.step(frame * 40);
    const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
    if (roundEnd) {
      return roundEnd[1] === 0 ? "win" : roundEnd[1] === 1 ? "loss" : "double_death";
    }
  }
  return "draw";
}

const started = performance.now();
const matrix = new Map();
const rows = [];
for (let index = 0; index < rounds; index += 1) {
  const roundSeed = seed + index;
  const baseline = play(baselinePolicy, roundSeed);
  const candidate = play(challenger, roundSeed);
  const key = `${baseline}->${candidate}`;
  matrix.set(key, (matrix.get(key) ?? 0) + 1);
  rows.push({ seed: roundSeed, baseline, candidate });
  if (baseline !== candidate) console.log(`  changed ${roundSeed}: ${key}`);
}
const wins = (key) => rows.filter((row) => row[key] === "win").length;
console.log(`===== paired Browser JS: ${rounds} seeds @${seed} (${((performance.now() - started) / 1000).toFixed(1)}s) =====`);
console.log(`  ${baselinePolicy} ${wins("baseline")}/${rounds} (${(100 * wins("baseline") / rounds).toFixed(1)}%)`);
console.log(`  ${challenger} ${wins("candidate")}/${rounds} (${(100 * wins("candidate") / rounds).toFixed(1)}%)`);
console.log(`  win delta ${wins("candidate") - wins("baseline") >= 0 ? "+" : ""}${wins("candidate") - wins("baseline")} games`);
console.log(`  outcome matrix ${JSON.stringify(Object.fromEntries(matrix))}`);
