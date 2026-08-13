import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";

const rounds = Number(process.argv[2] ?? 60);
const seed = Number(process.argv[3] ?? 980000);
const maxFrames = Number(process.argv[4] ?? 3000);
const policy = process.argv[5] ?? "killfield-js";
const failures = [];
const started = performance.now();

for (let index = 0; index < rounds; index += 1) {
  const gameSeed = seed + index;
  const arena = new BrowserArena({ seed: gameSeed });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  let result = "draw";
  let frames = 0;
  for (; frames < maxFrames; frames += 1) {
    arena.step(frames * 40);
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (!end) continue;
    result = end[1] === 0 ? "win" : end[1] === 1 ? "loss" : "double_death";
    frames += 1;
    break;
  }
  if (result !== "win") failures.push({ seed: gameSeed, result, frames });
  console.log(`${gameSeed} ${result} ${frames}`);
}

console.log(`===== ${policy}: ${failures.length}/${rounds} non-wins in ${((performance.now() - started) / 1000).toFixed(1)}s =====`);
console.log(JSON.stringify(failures));
