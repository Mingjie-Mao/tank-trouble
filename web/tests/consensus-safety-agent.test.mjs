import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("consensus safety candidate runs without changing the public state contract", () => {
  const arena = new BrowserArena({ seed: 970001 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-consensus",
    right_policy: "laika-js",
  });
  for (let frame = 0; frame < 20; frame += 1) arena.step(frame * 40);

  const state = arena.state();
  assert.equal(state.left_policy, "p27-js-consensus");
  assert.equal(state.tanks.length, 2);
  assert.ok(Number.isFinite(state.runtime.left_decision_ms));
});
