import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { BrowserArenaWorkerRuntime } from "../lib/browser-arena-worker-runtime.js";
import { SearchHelper, SearchSizer } from "../lib/search-pool.js";

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

// Slowed (most rollouts unplayed) the search narrows to 2×2; played again, it widens back.
test("the search narrows while the page is slowed", async (t) => {
  const sizer = new SearchSizer(8);
  const searches = (played, n) => Array.from({ length: n }, () => sizer.record(Array.from({ length: 16 }, (_, k) => (k < played ? 0 : null))));
  assert.deepEqual(searches(15, 40).filter(Boolean), []);
  assert.deepEqual(searches(5, 16).filter(Boolean), [{ candidates: 2, rollouts: 2 }]);
  assert.deepEqual(searches(13, 40).filter(Boolean), []);
  assert.deepEqual(searches(16, 16).filter(Boolean), [{ candidates: 4, rollouts: 4 }]);
  assert.equal(new SearchSizer(4).record([null, null]), null);

  // Helpers that never answer: each search plays only this worker's share.
  const helpers = Array.from({ length: 6 }, () => new MessageChannel());
  t.after(() => helpers.forEach((channel) => channel.port1.close()));
  const runtime = new BrowserArenaWorkerRuntime();
  await runtime.handle({ type: "init", seed: 1010043, autostart: false, wasm, helpers: helpers.map((c) => c.port1), budget_ms: 1 });
  let state = null;
  for (let frame = 0; frame < 600 && !(state?.telemetry.search === "2×2"); frame += 1) state = await runtime.advance(frame * 40);
  assert.equal(state.telemetry.search, "2×2", `still ${state.telemetry.search} after ${state.telemetry.searches} searches`);
  assert.equal(state.runtime.search_workers, 7);
});
