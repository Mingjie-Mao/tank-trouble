import { BrowserArena } from "../lib/browser-arena.js";
import { incomingRisk } from "../lib/killfield-runtime/src/killfield/risk.js";

const policy = process.argv[2]?.includes("js") ? process.argv[2] : "killfield-js";
const seedArgs = policy === process.argv[2] ? process.argv.slice(3) : process.argv.slice(2);
const seeds = seedArgs.map(Number);
if (!seeds.length) seeds.push(980005, 980007, 980028, 980029, 980036, 980042, 980049);

for (const seed of seeds) {
  const arena = new BrowserArena({ seed });
  arena.command({ action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js" });
  const recent = [];
  const deaths = [];
  let result = "draw";
  for (let step = 0; step < 3000; step += 1) {
    const agent = arena.leftAgent;
    const riskBefore = agent?.boxes?.length ? incomingRisk(arena.game, agent.boxes) : 0;
    const aliveBefore = arena.game.tanks.map((tank) => tank.alive);
    arena.step(step * 40);
    const row = {
      frame: arena.game.frame,
      action: agent?.lastAction,
      decision: agent?.lastDecisionKind,
      riskBefore,
      aliveBefore,
      aliveAfter: arena.game.tanks.map((tank) => tank.alive),
      events: arena.lastEvents,
    };
    recent.push(row);
    if (recent.length > 15) recent.shift();
    if (arena.lastEvents.some((event) => event[0] === "destroy")) {
      deaths.push({ frame: arena.game.frame, window: recent.slice() });
    }
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (!end) continue;
    result = end[1] === 0 ? "win" : end[1] === 1 ? "loss" : "double_death";
    break;
  }
  console.log(JSON.stringify({ seed, policy, result, telemetry: arena.leftAgent?.telemetry?.(), deaths }));
}
