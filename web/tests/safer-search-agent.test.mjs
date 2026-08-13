import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("experimental JS safety agent runs in watch mode", () => {
  const arena = new BrowserArena({ seed: 970000 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-shield",
    right_policy: "laika-js",
  });
  for (let frame = 0; frame < 20; frame += 1) arena.step(frame * 40);

  const state = arena.state();
  assert.equal(state.left_policy, "p27-js-shield");
  assert.ok(Number.isFinite(state.runtime.left_decision_ms));
  assert.ok(state.frame >= 20);
});
