import { performance } from "node:perf_hooks";

import { BrowserArena } from "../lib/browser-arena.js";

const policy = process.argv[2] ?? "p27-js-tactical";
const seeds = process.argv.slice(3).map(Number);
if (!seeds.length) seeds.push(980005, 980007, 980028, 980029, 980036, 980042, 980049);

const results = { win: 0, loss: 0, double_death: 0, draw: 0 };
const started = performance.now();
for (const seed of seeds) {
  const arena = new BrowserArena({ seed });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  let result = "draw";
  for (let frame = 0; frame < 3000; frame += 1) {
    arena.step(frame * 40);
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (!end) continue;
    result = end[1] === 0 ? "win" : end[1] === 1 ? "loss" : "double_death";
    break;
  }
  results[result] += 1;
  const telemetry = arena.leftAgent?.telemetry?.() ?? {};
  console.log(JSON.stringify({
    seed,
    result,
    tacticalAudits: telemetry.tacticalAudits ?? 0,
    tacticalOverrides: telemetry.tacticalOverrides ?? 0,
    settlementPlans: telemetry.settlementPlans ?? 0,
    planP95Ms: telemetry.planP95Ms ?? 0,
    tacticalP95Ms: telemetry.tacticalP95Ms ?? 0,
  }));
}
console.log(`===== ${policy}: ${results.win}/${seeds.length} wins in ${((performance.now() - started) / 1000).toFixed(1)}s =====`);
console.log(JSON.stringify(results));
