import { BrowserArena } from "../lib/browser-arena.js";
import { incomingRisk } from "../lib/killfield-runtime/src/killfield/risk.js";

const seeds = process.argv.slice(2).map(Number);
if (!seeds.length) seeds.push(980005, 980007, 980028, 980029, 980036, 980042, 980049);

for (const seed of seeds) {
  const arena = new BrowserArena({ seed });
  arena.command({ action: "mode", mode: "watch", left_policy: "killfield-js", right_policy: "laika-js" });
  const trace = [];
  let result = "draw";
  for (let step = 0; step < 3000; step += 1) {
    arena.step(step * 40);
    const agent = arena.leftAgent;
    const events = arena.lastEvents.map((event) => Array.from(event));
    const important = events.some((event) => ["fire", "hit", "destroy", "round_end"].includes(event[0]));
    const risk = agent?.boxes?.length ? incomingRisk(arena.game, agent.boxes) : 0;
    if (important || risk > 0 || agent?.actionNoEffect) {
      trace.push({
        frame: arena.game.frame,
        action: agent?.lastAction,
        decision: agent?.lastDecisionKind,
        risk,
        noEffect: agent?.actionNoEffect ?? false,
        alive: arena.game.tanks.map((tank) => tank.alive),
        bulletsFired: arena.game.tanks.map((tank) => tank.bulletsFired),
        bullets: arena.game.bullets.map((bullet) => ({
          owner: bullet.owner.number,
          x: Number(bullet.x.toFixed(1)),
          y: Number(bullet.y.toFixed(1)),
          life: bullet.lifetime,
          bounced: bullet.hasBounced,
        })),
        events,
      });
    }
    const end = events.find((event) => event[0] === "round_end");
    if (!end) continue;
    result = end[1] === 0 ? "win" : end[1] === 1 ? "loss" : "double_death";
    break;
  }
  const lethal = trace.filter((row) => row.events.some(
    (event) => event[0] === "hit" || event[0] === "destroy" || event[0] === "round_end",
  ));
  const fires = trace.filter((row) => row.events.some((event) => event[0] === "fire"));
  console.log(JSON.stringify({ seed, result, fires, lethal, tail: trace.slice(-25) }, null, 2));
}
