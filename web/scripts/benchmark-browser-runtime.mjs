import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";

const frames = Number(process.argv[2] ?? 2000);
const mode = process.argv[3] ?? "watch";
const policy = process.argv[4] ?? "killfield-js";
const arena = new BrowserArena({ seed: 970000 });
if (mode === "watch") {
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
}
if (mode === "selfplay") {
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: "killfield-js",
    right_policy: "killfield-js",
  });
}
const samples = [];
for (let frame = 0; frame < frames; frame += 1) {
  const started = performance.now();
  arena.step(frame * 40);
  samples.push(performance.now() - started);
}
samples.sort((a, b) => a - b);
const at = (q) => samples[Math.min(samples.length - 1, Math.floor(q * samples.length))];
const total = samples.reduce((sum, value) => sum + value, 0);
const state = arena.state();
console.log(`browser-native ${policy} (${mode}): ${frames} frames`);
console.log(`mean ${(total / frames).toFixed(3)} ms  p50 ${at(0.50).toFixed(3)} ms  p95 ${at(0.95).toFixed(3)} ms  p99 ${at(0.99).toFixed(3)} ms  max ${at(1).toFixed(3)} ms`);
console.log(`search rate ${(state.telemetry.search_rate * 100).toFixed(1)}%  last ${state.telemetry.last_reason}`);
