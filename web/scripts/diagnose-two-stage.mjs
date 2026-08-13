import { BrowserArena } from "../lib/browser-arena.js";

const policy = process.argv[2]?.startsWith("p27-js-two-stage")
  ? process.argv[2]
  : "p27-js-two-stage";
const seedArgs = policy === process.argv[2] ? process.argv.slice(3) : process.argv.slice(2);
const seeds = seedArgs.map(Number);
if (!seeds.length) seeds.push(970012, 970018);

for (const seed of seeds) {
  const arena = new BrowserArena({ seed });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: policy,
    right_policy: "laika-js",
  });
  let winner = "draw";
  for (let frame = 0; frame < 3000; frame += 1) {
    arena.step(frame * 40);
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (end) {
      winner = end[1] === 0 ? "win" : end[1] === 1 ? "loss" : "double_death";
      break;
    }
  }
  const telemetry = arena.leftAgent?.telemetry?.() ?? {};
  console.log(JSON.stringify({
    seed,
    winner,
    twoStageCalls: telemetry.twoStageCalls,
    twoStageChanges: telemetry.twoStageChanges,
    changes: telemetry.changeLog,
  }, null, 2));
}
