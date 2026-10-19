import assert from "node:assert/strict";
import test from "node:test";

import { BrowserArena } from "../lib/browser-arena.js";
import { TacticalCandidateAgent } from "../lib/tactical-candidate-agent.js";

test("shot settlement audit repairs a real double death without a seed rule", () => {
  // Tactical v2 as the arena builds it (selfplay, L1 opponent model), with the
  // audit's per-round allowance either at its default or at zero. The round
  // ends before the arena would rebuild the agent.
  const play = (maximumUnsafeAuditsPerRound) => {
    const seed = 2700185;
    const arena = new BrowserArena({ seed });
    arena.command({
      action: "mode",
      mode: "selfplay",
      left_policy: "p27-js-tactical-v2",
      right_policy: "dodger-js",
    });
    arena.leftAgent = new TacticalCandidateAgent({
      seed: seed + 101,
      oppModel: "L1",
      enableShotSettlementAudit: true,
      maximumUnsafeAuditsPerRound,
    });
    for (let frame = 0; frame < 3000; frame += 1) {
      arena.step(frame * 40);
      const roundEnd = arena.lastEvents.find((event) => event[0] === "round_end");
      if (roundEnd) return { roundEnd, telemetry: arena.leftAgent.telemetry() };
    }
    throw new Error("round did not finish");
  };

  assert.deepEqual(play(0).roundEnd, ["round_end", null]);
  const repaired = play(4);
  assert.deepEqual(repaired.roundEnd, ["round_end", 0]);
  assert.ok(repaired.telemetry.unsafeShotsSuppressed >= 1);
  assert.ok(repaired.telemetry.shotAuditP95Ms < 15);
});
