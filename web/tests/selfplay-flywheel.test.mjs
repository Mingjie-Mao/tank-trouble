import assert from "node:assert/strict";
import test from "node:test";

import { summarizeSelfplayFlywheel } from "../lib/selfplay-flywheel.js";

function report(candidate, outcomes, p95DecisionMs = 8) {
  const games = outcomes.map((outcome, index) => ({
    candidate,
    opponent: "laika-js",
    candidateSide: index % 2,
    seed: 10 + Math.floor(index / 2),
    outcome,
    frames: 100,
  }));
  const win = outcomes.filter((outcome) => outcome === "win").length;
  const doubleDeath = outcomes.filter((outcome) => outcome === "double_death").length;
  const draw = outcomes.filter((outcome) => outcome === "draw").length;
  const overall = {
    games: outcomes.length,
    win,
    loss: outcomes.length - win - doubleDeath - draw,
    double_death: doubleDeath,
    draw,
    winRate: win / outcomes.length,
    colorGap: 0,
    p95DecisionMs,
  };
  return {
    candidate,
    seed: 10,
    roundsPerSide: outcomes.length / 2,
    overall,
    opponents: { "laika-js": overall },
    games,
  };
}

test("flywheel identifies repaired and regressed paired seeds", () => {
  const summary = summarizeSelfplayFlywheel({
    championReport: report("champion", ["win", "loss", "win", "win"]),
    candidateReport: report("candidate", ["win", "win", "double_death", "win"]),
  });
  assert.equal(summary.recoveries.length, 1);
  assert.equal(summary.regressions.length, 1);
  assert.equal(summary.hardCases.length, 2);
  assert.equal(summary.gates.pairedCorrections, true);
  assert.equal(summary.gates.doubleDeath, false);
  assert.equal(summary.readyForLaikaBlind, false);
});

test("flywheel rejects color or latency degradation relative to its paired champion", () => {
  const champion = report("champion", ["win", "win", "win", "win"], 8);
  const candidate = report("candidate", ["win", "win", "win", "win"], 8.3);
  candidate.opponents["laika-js"] = { ...candidate.opponents["laika-js"], colorGap: 0.25 };
  const summary = summarizeSelfplayFlywheel({
    championReport: champion,
    candidateReport: candidate,
  });
  assert.equal(summary.gates.opponentPool, true);
  assert.equal(summary.gates.color, false);
  assert.equal(summary.gates.latency, false);
  assert.equal(summary.readyForLaikaBlind, false);
});
