import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { BrowserArena, loadArena } from "../lib/browser-arena.js";

await loadArena(readFileSync(new URL("../lib/wasm/tank_bg.wasm", import.meta.url)));

test("browser runtime exposes the renderer state contract", () => {
  const state = new BrowserArena({ seed: 970000 }).state();
  assert.equal(state.connected, true);
  assert.equal(state.fps, 25);
  assert.equal(state.left_policy, "ppo-league");
  assert.equal(state.right_policy, "laika-js");
  assert.deepEqual(state.available_policies.map((p) => p.value), ["ppo-league", "laika-js", "random-js"]);
  assert.equal(state.tanks.length, 2);
  assert.ok(state.walls.length > 0);
  assert.ok(state.world_width > 0);
  assert.ok(state.tanks[0].display_scale > 0);
});

test("browser runtime advances physics and supports all three modes", () => {
  const arena = new BrowserArena({ seed: 970001 });
  for (let frame = 0; frame < 12; frame += 1) arena.step(frame * 40);
  assert.equal(arena.state().frame, 12);

  arena.command({ action: "mode", mode: "play", right_policy: "ppo-league" });
  arena.command({ action: "input", controls: { forward: true, fire: true } });
  arena.step(520);
  assert.equal(arena.state().left_policy, "human");

  arena.command({ action: "mode", mode: "selfplay", left_policy: "random-js", right_policy: "laika-js" });
  arena.step(560);
  assert.equal(arena.state().mode, "selfplay");
  assert.equal(arena.state().right_policy, "laika-js");
  assert.throws(() => arena.command({ action: "mode", mode: "watch", left_policy: "p27-js-tactical-v2" }));
});

test("the AI view reports moves, shot path and the network's choices", () => {
  const arena = new BrowserArena({ seed: 970002 });
  arena.command({ action: "vision", enabled: true });
  let vision = null;
  for (let frame = 0; frame < 30 && !vision?.actions; frame += 1) vision = arena.step(frame * 40).vision;
  assert.equal(vision.dodge.length, 9);
  assert.ok(vision.shot.points.length >= 2);
  assert.ok(["self", "opponent", "none"].includes(vision.shot.hit));
  assert.ok(vision.actions.length >= 1 && vision.actions[0].length === 4);
});

test("new maze keeps the score, reset clears it", () => {
  const arena = new BrowserArena({ seed: 970003 });
  let state = arena.state();
  for (let frame = 0; frame < 3000 && state.scores[0] + state.scores[1] === 0; frame += 1) state = arena.step(frame * 40);
  const scores = state.scores;
  assert.ok(scores[0] + scores[1] > 0, "a round finished");
  state = arena.command({ action: "new_maze", seed: 12345 });
  assert.equal(state.seed, 12345);
  assert.deepEqual(state.scores, scores);
  assert.deepEqual(arena.command({ action: "reset_score" }).scores, [0, 0]);
});
