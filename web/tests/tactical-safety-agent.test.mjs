import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("tactical safety repairs at least one frozen H36 failure seed", () => {
  const arena = new BrowserArena({ seed: 980049 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-tactical",
    right_policy: "laika-js",
  });
  let winner;
  for (let frame = 0; frame < 1000; frame += 1) {
    arena.step(frame * 40);
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (end) {
      winner = end[1];
      break;
    }
  }
  assert.equal(winner, 0);
  assert.ok(arena.leftAgent.telemetry().tacticalOverrides > 0);
});
