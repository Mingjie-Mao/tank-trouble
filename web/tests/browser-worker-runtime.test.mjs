import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { BrowserArenaWorkerRuntime } from "../lib/browser-arena-worker-runtime.js";

function comparable(state) {
  return {
    mode: state.mode,
    paused: state.paused,
    seed: state.seed,
    frame: state.frame,
    round: state.round,
    frozen: state.frozen,
    scores: state.scores,
    streak: state.streak,
    leftPolicy: state.left_policy,
    rightPolicy: state.right_policy,
    tanks: state.tanks,
    bullets: state.bullets,
    walls: state.walls,
    lastReason: state.telemetry.last_reason,
  };
}

test("worker-owned arena preserves direct BrowserArena trajectory", () => {
  const emitted = [];
  const runtime = new BrowserArenaWorkerRuntime({ emit: (message) => emitted.push(message) });
  runtime.handle({ type: "init", seed: 1010042, autostart: false });
  const direct = new BrowserArena({ seed: 1010042 });

  const command = {
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-tactical-v2",
    right_policy: "laika-js",
  };
  runtime.handle({ type: "command", id: 1, payload: command });
  direct.command(command);
  for (let frame = 0; frame < 400; frame += 1) {
    const workerState = runtime.advance(frame * 40);
    const directState = direct.step(frame * 40);
    assert.deepEqual(comparable(workerState), comparable(directState),
      `worker trajectory diverged at frame ${frame}`);
  }
  assert.ok(emitted.some((message) => message.type === "response" && message.id === 1));
});
