import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("two-stage search candidate executes with finite telemetry", () => {
  const arena = new BrowserArena({ seed: 970001 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-two-stage",
    right_policy: "laika-js",
  });
  for (let frame = 0; frame < 40; frame += 1) arena.step(frame * 40);

  const state = arena.state();
  assert.equal(state.left_policy, "p27-js-two-stage");
  assert.equal(state.frame, 40);
  assert.ok(Number.isFinite(state.runtime.left_decision_ms));
  assert.ok(state.telemetry.p95_ms >= 0);
});
