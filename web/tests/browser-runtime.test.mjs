import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("browser runtime exposes the renderer state contract", () => {
  const arena = new BrowserArena({ seed: 970000 });
  const state = arena.state();

  assert.equal(state.connected, true);
  assert.equal(state.fps, 25);
  assert.equal(state.left_policy, "ppo-league");
  assert.equal(state.available_policies[0].value, "ppo-league");
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

  arena.command({ action: "mode", mode: "play", right_policy: "p27-js-tactical-v2" });
  arena.command({ action: "input", controls: { forward: true, fire: true } });
  arena.step(520);
  assert.equal(arena.state().left_policy, "human");

  arena.command({
    action: "mode",
    mode: "selfplay",
    left_policy: "random-js",
    right_policy: "dodger-js",
  });
  arena.step(560);
  assert.equal(arena.state().mode, "selfplay");
  assert.equal(arena.state().right_policy, "dodger-js");
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

test("every policy runs under the engine's original collision model", () => {
  // lib/engine/ implements the decompiled Flash collision model only, so no
  // policy may switch the world to another one.
  const arena = new BrowserArena({ seed: 970000 });
  for (const policy of ["ppo-league", "p27-js-tactical-v2", "ppo-league"]) {
    arena.command({ action: "mode", mode: "watch", left_policy: policy });
    assert.equal(arena.wallSliding, false);
    assert.equal(arena.game.wallSliding, false);
  }
  assert.throws(() => new BrowserArena({ seed: 970000, wallSliding: true }));
});

test("the AI view reports tank 0's lookahead only while it is on", () => {
  const arena = new BrowserArena({ seed: 970003 });
  arena.command({ action: "mode", mode: "watch", left_policy: "p27-js-tactical-v2" });
  arena.step(0);
  assert.equal(arena.state().vision, null);
  arena.command({ action: "vision", enabled: true });
  let vision = null;
  for (let frame = 1; frame < 40 && !vision; frame += 1) {
    arena.step(frame * 40);
    vision = arena.state().vision;
  }
  assert.ok(vision, "vision is reported while tank 0 is alive");
  assert.equal(vision.dodge.length, 9);
  assert.ok(vision.shot.points.length >= 2);
  assert.ok(["opponent", "self", "none"].includes(vision.shot.hit));
  assert.ok(Array.isArray(vision.threats));
  arena.command({ action: "vision", enabled: false });
  assert.equal(arena.state().vision, null);
});
