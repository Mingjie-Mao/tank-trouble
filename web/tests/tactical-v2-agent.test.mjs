import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { movingAimInterceptPlan } from "../lib/tactical-v2-agent.js";

test("Tactical v2 activates topology chase only after a sustained stall", () => {
  const arena = new BrowserArena({ seed: 1300000 });
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: "dodger-js",
    right_policy: "p27-js-tactical-v2",
  });
  for (let frame = 0; frame < 180; frame += 1) arena.step(frame * 40);
  assert.ok(arena.rightAgent.topologyOverrides > 0);
  assert.ok(arena.rightAgent.topologyBursts > 0);
});

test("moving-target intercept search proposes only physics-verified plans", () => {
  const arena = new BrowserArena({ seed: 1400004 });
  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: "p27-js-tactical-v2",
    right_policy: "dodger-js",
  });
  let plan = null;
  for (let frame = 0; frame < 900 && plan === null; frame += 1) {
    arena.step(frame * 40);
    if (arena.game.bullets.length === 0) plan = movingAimInterceptPlan(arena.game);
  }
  assert.notEqual(plan, null);
  assert.ok(plan.frame > 0);
  assert.ok(plan.flightFrame > 0);
});
