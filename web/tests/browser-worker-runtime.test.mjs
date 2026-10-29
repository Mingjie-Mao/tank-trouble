import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { BrowserArenaWorkerRuntime } from "../lib/browser-arena-worker-runtime.js";
import { SearchHelper } from "../lib/search-pool.js";

const wasm = readFileSync(new URL("../lib/wasm/tank_bg.wasm", import.meta.url));

function comparable(state) {
  const { mode, paused, seed, frame, round, frozen, scores, streak, tanks, bullets, walls } = state;
  return { mode, paused, seed, frame, round, frozen, scores, streak, tanks, bullets, walls,
    left: state.left_policy, right: state.right_policy };
}

// The search runs on a wall-clock budget, so the deterministic drivers are compared.
test("worker-owned arena preserves the direct trajectory", async () => {
  const emitted = [];
  const runtime = new BrowserArenaWorkerRuntime({ emit: (message) => emitted.push(message) });
  await runtime.handle({ type: "init", seed: 1010042, autostart: false, wasm });
  const direct = new BrowserArena({ seed: 1010042 });
  const command = { action: "mode", mode: "selfplay", left_policy: "random-js", right_policy: "laika-js" };
  await runtime.handle({ type: "command", id: 1, payload: command });
  direct.command(command);
  for (let frame = 0; frame < 400; frame += 1) {
    assert.deepEqual(comparable(await runtime.advance(frame * 40)), comparable(direct.step(frame * 40)),
      `worker trajectory diverged at frame ${frame}`);
  }
  assert.ok(emitted.some((message) => message.type === "response" && message.id === 1));
});

// With no deadline the shared search is the single-worker search, move for move.
test("helpers share the search without changing a move", async (t) => {
  const helpers = Array.from({ length: 6 }, () => {
    const channel = new MessageChannel();
    new SearchHelper(channel.port2);
    return channel.port1;
  });
  t.after(() => helpers.forEach((port) => port.close()));
  const runtime = new BrowserArenaWorkerRuntime();
  await runtime.handle({ type: "init", seed: 1010043, autostart: false, wasm, helpers, budget_ms: 1e9 });
  const direct = new BrowserArena({ seed: 1010043 });
  direct.command({ action: "search", candidates: 4, rollouts: 4, budget_ms: 1e9 });
  let state = null;
  for (let frame = 0; frame < 240; frame += 1) {
    if (frame === 120) {
      const command = { action: "new_maze", seed: 4242 };
      await runtime.handle({ type: "command", id: 1, payload: command });
      direct.command(command);
    }
    state = await runtime.advance(frame * 40);
    const expected = direct.step(frame * 40);
    assert.deepEqual(comparable(state), comparable(expected), `shared search diverged at frame ${frame}`);
    assert.equal(state.telemetry.search_changes, expected.telemetry.search_changes);
  }
  assert.equal(state.telemetry.search, "4×4");
  assert.equal(state.runtime.search_workers, 7);
  assert.ok(state.telemetry.searches > 10, `only ${state.telemetry.searches} searches`);
  assert.equal(runtime.pool.broken, false);
});
