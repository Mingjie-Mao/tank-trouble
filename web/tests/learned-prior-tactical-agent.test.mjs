import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("learned action prior prunes rollouts while exact physics retains authority", () => {
  const arena = new BrowserArena({ seed: 2200000 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-tactical-prior-k6",
    right_policy: "laika-js",
  });
  for (let frame = 0; frame < 160; frame += 1) arena.step(frame * 40);
  const telemetry = arena.leftAgent.telemetry();
  assert.ok(telemetry.priorCalls > 0);
  assert.ok(telemetry.meanPriorCandidates >= 6);
  assert.ok(telemetry.meanPriorCandidates < 10);
  assert.equal(telemetry.priorGateSuggestions, 0);
});
