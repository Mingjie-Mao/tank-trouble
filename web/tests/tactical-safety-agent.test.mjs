import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";

test("tactical safety repairs at least one frozen H36 failure seed", () => {
  const arena = new BrowserArena({ seed: 980049 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-tactical",
    right_policy: "laika-js",
  });
  let winner;
  for (let frame = 0; frame < 1000; frame += 1) {
    arena.step(frame * 40);
    const end = arena.lastEvents.find((event) => event[0] === "round_end");
    if (end) {
      winner = end[1];
      break;
    }
  }
  assert.equal(winner, 0);
  assert.ok(arena.leftAgent.telemetry().tacticalOverrides > 0);
});

test("an unspent evasion budget leaves the champion bit-for-bit unchanged", () => {
  // The whole point of a budget trigger over a fixed breadth cut is that it
  // costs nothing while the frame can afford the full search. A budget large
  // enough never to fire must therefore reproduce the champion exactly.
  const trace = (policy) => {
    const arena = new BrowserArena({ seed: 970000 });
    arena.command({
      action: "mode", mode: "watch", left_policy: policy, right_policy: "laika-js",
    });
    const poses = [];
    for (let frame = 0; frame < 400; frame += 1) {
      arena.step(frame * 40);
      const tank = arena.game.tanks[0];
      poses.push(`${tank.x.toFixed(6)},${tank.y.toFixed(6)},${tank.rotation.toFixed(6)}`);
    }
    return { poses, telemetry: arena.leftAgent.telemetry() };
  };
  const frozen = trace("p27-js-tactical-v3");
  const budgeted = trace("p27-js-tactical-v3-b100000");
  assert.equal(budgeted.telemetry.evasionBudgetNarrowed, 0);
  assert.deepEqual(budgeted.poses, frozen.poses);
});

test("a spent evasion budget narrows the search and still returns a plan", () => {
  // Policy name rather than a direct field write: the arena rebuilds both
  // agents on every `new_round` event, so anything set on the instance is
  // discarded after the first round. A zero budget is already spent by the
  // time the audit starts, which makes this independent of machine speed.
  const arena = new BrowserArena({ seed: 990000 });
  arena.command({
    action: "mode",
    mode: "watch",
    left_policy: "p27-js-tactical-v3-b0",
    right_policy: "laika-js",
  });
  assert.equal(arena.leftAgent.evasionBudgetMs, 0);

  let narrowed = 0;
  let dropped = 0;
  let audits = 0;
  for (let frame = 0; frame < 3000; frame += 1) {
    arena.step(frame * 40);
    // Counters reset with the agent each round, so they must be accumulated.
    const telemetry = arena.leftAgent.telemetry();
    if (arena.lastEvents.some((event) => event[0] === "new_round")) {
      narrowed += telemetry.evasionBudgetNarrowed;
      dropped += telemetry.evasionRootsDropped;
      audits += telemetry.tacticalAudits;
    }
  }
  const last = arena.leftAgent.telemetry();
  audits += last.tacticalAudits;
  narrowed += last.evasionBudgetNarrowed;
  dropped += last.evasionRootsDropped;

  assert.ok(audits > 0, "expected the evasion audit to run on this seed");
  assert.ok(narrowed > 0, "a zero budget must narrow every audited frame");
  assert.ok(dropped > 0);
  // Narrowing degrades the search; it must not break it.
  assert.ok(arena.game.roundNumber > 0);
});
