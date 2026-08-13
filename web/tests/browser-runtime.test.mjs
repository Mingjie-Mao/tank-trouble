import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("browser runtime exposes the renderer state contract", () => {
  const arena = new BrowserArena({ seed: 970000 });
  const state = arena.state();

  assert.equal(state.connected, true);
  assert.equal(state.fps, 25);
  assert.equal(state.left_policy, "p27-js-tactical-v2");
  assert.equal(state.available_policies[0].value, "p27-js-tactical-v2");
  assert.equal(state.right_policy, "laika-js");
  assert.equal(state.tanks.length, 2);
  assert.ok(state.walls.length > 0);
  assert.ok(state.world_width > 0);
  assert.ok(state.tanks[0].display_scale > 0);
});

test("browser runtime advances physics and supports all three modes", () => {
  const arena = new BrowserArena({ seed: 970001 });
  for (let frame = 0; frame < 12; frame += 1) arena.step(frame * 40);
  assert.equal(arena.state().frame, 12);

  arena.command({ action: "mode", mode: "play", right_policy: "killfield-js" });
  arena.command({ action: "input", controls: { forward: true, fire: true } });
  arena.step(520);
  assert.equal(arena.state().left_policy, "human");

  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: "random-js",
    right_policy: "idle-js",
  });
  arena.step(560);
  assert.equal(arena.state().mode, "selfplay");
  assert.equal(arena.state().right_policy, "idle-js");
});

test("browser runtime exposes fair scripted league opponents", () => {
  const arena = new BrowserArena({ seed: 970002 });
  const policies = new Set(arena.state().available_policies.map((policy) => policy.value));
  assert.ok(policies.has("laika-js"));
  assert.ok(policies.has("hunter-js"));
  assert.ok(policies.has("dodger-js"));

  for (const rightPolicy of ["laika-js", "hunter-js", "dodger-js"]) {
    arena.command({
      action: "mode",
      mode: "selfplay",
      left_policy: "random-js",
      right_policy: rightPolicy,
    });
    arena.step();
    assert.equal(arena.state().right_policy, rightPolicy);
  }
});
